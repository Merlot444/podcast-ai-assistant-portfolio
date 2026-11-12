"""AST-based allow-list validation for untrusted SQL candidates."""

from __future__ import annotations

import re
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from app.text2sql.catalog import DAY20_CATALOG, SchemaCatalog
from app.text2sql.schemas import GeneratedQuery, ValidatedQuery


class QueryValidationError(RuntimeError):
    """Safe policy rejection raised before a database can be reached."""


@dataclass(frozen=True)
class ValidationPolicy:
    dialect: str = "sqlite"
    default_limit: int = 20
    max_limit: int = 100
    max_offset: int = 100
    max_sql_length: int = 4_000
    max_ast_nodes: int = 180
    max_ast_depth: int = 20
    max_tables: int = 2
    max_joins: int = 1


_AGGREGATES = (exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max)
_COMPARISONS = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)
_PARAMETER_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,31}$")
_ALLOWED_NODE_TYPES = (
    exp.Select,
    exp.Column,
    exp.Identifier,
    exp.Alias,
    exp.Table,
    exp.TableAlias,
    exp.From,
    exp.Join,
    exp.Where,
    exp.Group,
    exp.Having,
    exp.Order,
    exp.Ordered,
    exp.Limit,
    exp.Offset,
    exp.Placeholder,
    exp.Literal,
    exp.Star,
    exp.Paren,
    exp.And,
    exp.Or,
    *_COMPARISONS,
    *_AGGREGATES,
)


class QueryValidator:
    def __init__(
        self,
        approval_token: object,
        *,
        catalog: SchemaCatalog = DAY20_CATALOG,
        policy: ValidationPolicy = ValidationPolicy(),
    ) -> None:
        self._approval_token = approval_token
        self.catalog = catalog
        self.policy = policy

    def validate(self, candidate: GeneratedQuery) -> ValidatedQuery:
        sql = candidate.sql
        if len(sql) > self.policy.max_sql_length:
            raise QueryValidationError("SQL 超过允许长度。")
        self._reject_comments(sql)

        try:
            statements = sqlglot.parse(sql, read=self.policy.dialect)
        except SqlglotError:
            raise QueryValidationError("SQL 无法按 SQLite 方言解析。") from None
        if len(statements) != 1:
            raise QueryValidationError("只允许一条 SQL 语句。")
        tree = statements[0]
        if not isinstance(tree, exp.Select):
            raise QueryValidationError("只允许单条 SELECT 查询。")

        nodes = list(tree.walk())
        if len(nodes) > self.policy.max_ast_nodes:
            raise QueryValidationError("SQL AST 节点数量超过上限。")
        if _ast_depth(tree) > self.policy.max_ast_depth:
            raise QueryValidationError("SQL AST 深度超过上限。")
        if any(not isinstance(node, _ALLOWED_NODE_TYPES) for node in nodes):
            raise QueryValidationError("SQL 包含未允许的语法结构。")
        if any(isinstance(node, exp.Select) for node in nodes[1:]):
            raise QueryValidationError("Day 20 不允许子查询、CTE 或集合查询。")
        if tree.args.get("into") or tree.args.get("locks") or tree.args.get("with_"):
            raise QueryValidationError("SQL 包含写入、锁定或 CTE 结构。")
        if tree.args.get("distinct"):
            raise QueryValidationError("Day 20 不允许 DISTINCT。")
        if tree.args.get("from_") is None:
            raise QueryValidationError("SELECT 必须读取 Catalog 中的表。")

        tables, aliases = self._validate_tables(tree)
        referenced_columns = self._validate_columns(tree, tables, aliases)
        self._validate_functions(tree)
        self._validate_projections(tree)
        self._validate_predicates(tree)
        self._validate_group_and_order(tree)
        self._validate_joins(tree, aliases)
        self._validate_parameters(tree, candidate)
        effective_limit = self._rewrite_limit(tree)

        validated = ValidatedQuery(
            normalized_sql=tree.sql(dialect=self.policy.dialect),
            parameters=candidate.parameters,
            referenced_tables=tuple(sorted(tables)),
            referenced_columns=tuple(sorted(referenced_columns)),
            effective_limit=effective_limit,
        )
        validated._approve(self._approval_token)
        return validated

    @staticmethod
    def _reject_comments(sql: str) -> None:
        try:
            statements = sqlglot.parse(sql, read="sqlite")
        except SqlglotError:
            return
        if any(node.comments for statement in statements for node in statement.walk()):
            raise QueryValidationError("SQL 注释和 Hint 不被允许。")

    def _validate_tables(
        self,
        tree: exp.Select,
    ) -> tuple[set[str], dict[str, str]]:
        table_nodes = list(tree.find_all(exp.Table))
        if not table_nodes or len(table_nodes) > self.policy.max_tables:
            raise QueryValidationError("查询表数量不在允许范围内。")
        tables: set[str] = set()
        aliases: dict[str, str] = {}
        for table in table_nodes:
            if table.db or table.catalog:
                raise QueryValidationError("不允许跨 Schema 或 Catalog 查询。")
            name = table.name
            if name not in self.catalog.table_names:
                raise QueryValidationError("查询引用了未授权表。")
            alias = table.alias_or_name
            if alias in aliases:
                raise QueryValidationError("表别名必须唯一。")
            aliases[alias] = name
            aliases.setdefault(name, name)
            tables.add(name)
        return tables, aliases

    def _validate_columns(
        self,
        tree: exp.Select,
        tables: set[str],
        aliases: dict[str, str],
    ) -> set[str]:
        referenced: set[str] = set()
        for column in tree.find_all(exp.Column):
            if isinstance(column.this, exp.Star):
                raise QueryValidationError("SELECT * 和 table.* 不被允许。")
            name = column.name
            if column.table:
                table_name = aliases.get(column.table)
                if table_name is None:
                    raise QueryValidationError("字段使用了未知表别名。")
            else:
                owners = [
                    table_name
                    for table_name in tables
                    if name in self.catalog.table(table_name).columns  # type: ignore[union-attr]
                ]
                if len(owners) != 1:
                    raise QueryValidationError("未限定字段必须唯一对应一个授权表。")
                table_name = owners[0]
            table = self.catalog.table(table_name)
            if table is None or name not in table.columns:
                raise QueryValidationError("查询引用了未授权字段。")
            referenced.add(f"{table_name}.{name}")
        return referenced

    def _validate_functions(self, tree: exp.Select) -> None:
        for function in tree.find_all(exp.Func):
            if not isinstance(function, _AGGREGATES):
                raise QueryValidationError("查询使用了未授权函数。")
            if function.sql_name().upper() not in self.catalog.aggregate_functions:
                raise QueryValidationError("查询使用了未授权聚合函数。")
            if isinstance(function, (exp.Sum, exp.Avg)):
                argument = function.this
                if not (
                    isinstance(argument, exp.Column)
                    and argument.name == "schema_version"
                ):
                    raise QueryValidationError("SUM/AVG 仅允许用于 schema_version。")
            if isinstance(function, exp.Count) and isinstance(function.this, exp.Star):
                if function.this.parent is not function:
                    raise QueryValidationError("只允许 COUNT(*) 使用星号。")
        for star in tree.find_all(exp.Star):
            if not isinstance(star.parent, exp.Count):
                raise QueryValidationError("SELECT * 和 table.* 不被允许。")

    @staticmethod
    def _validate_projections(tree: exp.Select) -> None:
        for projection in tree.expressions:
            value = projection.this if isinstance(projection, exp.Alias) else projection
            if not isinstance(value, (exp.Column, *_AGGREGATES)):
                raise QueryValidationError("SELECT 列表只允许字段或授权聚合。")
            if isinstance(value, _AGGREGATES):
                argument = value.this
                if not isinstance(argument, (exp.Column, exp.Star)):
                    raise QueryValidationError("聚合函数参数必须是字段或 COUNT(*)。")

    @staticmethod
    def _validate_predicates(tree: exp.Select) -> None:
        where = tree.args.get("where")
        if where is not None:
            QueryValidator._validate_boolean_expression(
                where.this,
                allow_aggregates=False,
            )
        having = tree.args.get("having")
        if having is not None:
            QueryValidator._validate_boolean_expression(
                having.this,
                allow_aggregates=True,
            )

    @staticmethod
    def _validate_boolean_expression(
        node: exp.Expression,
        *,
        allow_aggregates: bool,
    ) -> None:
        if isinstance(node, exp.Paren):
            QueryValidator._validate_boolean_expression(
                node.this,
                allow_aggregates=allow_aggregates,
            )
            return
        if isinstance(node, (exp.And, exp.Or)):
            QueryValidator._validate_boolean_expression(
                node.this,
                allow_aggregates=allow_aggregates,
            )
            QueryValidator._validate_boolean_expression(
                node.expression,
                allow_aggregates=allow_aggregates,
            )
            return
        if not isinstance(node, _COMPARISONS):
            raise QueryValidationError("WHERE/HAVING 仅允许受限比较表达式。")
        if not isinstance(node.expression, exp.Placeholder):
            raise QueryValidationError("查询条件值必须使用命名参数。")
        allowed_left = (exp.Column, *_AGGREGATES) if allow_aggregates else exp.Column
        if not isinstance(node.this, allowed_left):
            raise QueryValidationError("查询条件左侧必须是字段或授权聚合。")

    @staticmethod
    def _validate_group_and_order(tree: exp.Select) -> None:
        group = tree.args.get("group")
        if group and any(not isinstance(item, exp.Column) for item in group.expressions):
            raise QueryValidationError("GROUP BY 只允许授权字段。")
        order = tree.args.get("order")
        if order:
            for item in order.expressions:
                if not isinstance(item, exp.Ordered) or not isinstance(
                    item.this, exp.Column
                ):
                    raise QueryValidationError("ORDER BY 只允许授权字段。")

    def _validate_joins(self, tree: exp.Select, aliases: dict[str, str]) -> None:
        joins = list(tree.args.get("joins") or [])
        if len(joins) > self.policy.max_joins:
            raise QueryValidationError("JOIN 数量超过上限。")
        for join in joins:
            if join.args.get("side") or join.args.get("kind") not in (
                None,
                "",
                "INNER",
            ):
                raise QueryValidationError("只允许 INNER JOIN。")
            on = join.args.get("on")
            if not isinstance(on, exp.EQ):
                raise QueryValidationError("JOIN 必须使用显式授权关系。")
            if not isinstance(on.this, exp.Column) or not isinstance(
                on.expression, exp.Column
            ):
                raise QueryValidationError("JOIN 条件必须连接两个字段。")
            endpoints = frozenset(
                {
                    self._resolved_endpoint(on.this, aliases),
                    self._resolved_endpoint(on.expression, aliases),
                }
            )
            if not any(
                endpoints == relationship.endpoints
                for relationship in self.catalog.relationships
            ):
                raise QueryValidationError("JOIN 未使用 Catalog 授权关系。")

    @staticmethod
    def _resolved_endpoint(
        column: exp.Column,
        aliases: dict[str, str],
    ) -> tuple[str, str]:
        if not column.table or column.table not in aliases:
            raise QueryValidationError("JOIN 字段必须使用有效表别名限定。")
        return aliases[column.table], column.name

    @staticmethod
    def _validate_parameters(
        tree: exp.Select,
        candidate: GeneratedQuery,
    ) -> None:
        placeholders = list(tree.find_all(exp.Placeholder))
        if any(
            not placeholder.name
            or not _PARAMETER_NAME.fullmatch(placeholder.name)
            for placeholder in placeholders
        ):
            raise QueryValidationError("只允许命名参数。")
        names = {placeholder.name for placeholder in placeholders}
        if names != set(candidate.parameters):
            raise QueryValidationError("SQL 参数与 parameters 必须完全对应。")
        for placeholder in placeholders:
            if not _has_ancestor(placeholder, exp.Where, exp.Having):
                raise QueryValidationError("命名参数只能出现在 WHERE/HAVING。")

    def _rewrite_limit(self, tree: exp.Select) -> int:
        limit = tree.args.get("limit")
        if limit is None:
            effective = self.policy.default_limit
        else:
            effective = _positive_integer_literal(
                limit.expression,
                name="LIMIT",
            )
            effective = min(effective, self.policy.max_limit)
        offset = tree.args.get("offset")
        if offset is not None:
            value = _non_negative_integer_literal(offset.expression, name="OFFSET")
            if value > self.policy.max_offset:
                raise QueryValidationError("OFFSET 超过上限。")
        tree.set("limit", exp.Limit(expression=exp.Literal.number(effective)))
        return effective


def _positive_integer_literal(node: exp.Expression, *, name: str) -> int:
    value = _non_negative_integer_literal(node, name=name)
    if value == 0:
        raise QueryValidationError(f"{name} 必须大于 0。")
    return value


def _non_negative_integer_literal(node: exp.Expression, *, name: str) -> int:
    if not isinstance(node, exp.Literal) or node.is_string:
        raise QueryValidationError(f"{name} 必须是受控整数。")
    try:
        value = int(node.this)
    except (TypeError, ValueError):
        raise QueryValidationError(f"{name} 必须是受控整数。") from None
    if str(value) != str(node.this) or value < 0:
        raise QueryValidationError(f"{name} 必须是非负整数。")
    return value


def _has_ancestor(node: exp.Expression, *types: type[exp.Expression]) -> bool:
    parent = node.parent
    while parent is not None:
        if isinstance(parent, types):
            return True
        parent = parent.parent
    return False


def _ast_depth(node: exp.Expression) -> int:
    children = list(node.iter_expressions())
    if not children:
        return 1
    return 1 + max(_ast_depth(child) for child in children)

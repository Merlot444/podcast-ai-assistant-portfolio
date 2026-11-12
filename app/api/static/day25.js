"use strict";

const NETWORK_FAILURE_MESSAGE = "暂时无法连接服务，请稍后再试。";
const INVALID_RESPONSE_MESSAGE = "服务返回了无法识别的结果，请稍后再试。";
// Deliberate allowlist: a contract test keeps these fixed public messages in
// sync with app/api/application.py. Unknown backend text is never echoed.
const SAFE_API_MESSAGES = new Set([
  "请求格式无效。",
  "请求体过大。",
  "请求的节目不存在。",
  "问答服务暂时不可用。",
  "问答服务无法完成请求。",
  "服务暂时无法完成请求。",
]);
const REQUEST_ID_PATTERN = /^req_[0-9a-f]{32}$/;

const form = document.querySelector("#ask-form");
const submitButton = document.querySelector("#submit-button");
const submitLabel = document.querySelector("#submit-label");
const formStatus = document.querySelector("#form-status");
const answerRegion = document.querySelector("#answer-region");
const answerText = document.querySelector("#answer-text");
const answerRequestId = document.querySelector("#answer-request-id");
const citationRegion = document.querySelector("#citation-region");
const citationList = document.querySelector("#citation-list");
const errorRegion = document.querySelector("#error-region");
const errorText = document.querySelector("#error-text");
const errorRequestId = document.querySelector("#error-request-id");
const apiStatus = document.querySelector("#api-status");
const serviceStatus = document.querySelector(".service-status");

function setRequestId(target, requestId) {
  target.textContent = REQUEST_ID_PATTERN.test(requestId || "")
    ? `请求编号：${requestId}`
    : "";
}

function hideResults() {
  answerRegion.hidden = true;
  errorRegion.hidden = true;
  answerText.textContent = "";
  errorText.textContent = "";
  answerRequestId.textContent = "";
  errorRequestId.textContent = "";
  citationList.replaceChildren();
  citationRegion.hidden = true;
}

function citationRow(label, value) {
  const row = document.createElement("p");
  row.className = "citation-row";

  const labelNode = document.createElement("span");
  labelNode.className = "citation-label";
  labelNode.textContent = label;

  const valueNode = document.createElement("span");
  valueNode.className = "citation-value";
  valueNode.textContent = typeof value === "string" ? value : "—";

  row.append(labelNode, valueNode);
  return row;
}

function renderCitations(citations) {
  citationList.replaceChildren();
  if (!Array.isArray(citations) || citations.length === 0) {
    citationRegion.hidden = true;
    return;
  }

  citations.forEach((citation) => {
    const card = document.createElement("article");
    card.className = "citation-card";
    card.append(
      citationRow("类型", citation && citation.kind),
      citationRow("引用 ID", citation && citation.reference_id),
      citationRow("节目 ID", citation && citation.episode_id),
    );
    citationList.append(card);
  });
  citationRegion.hidden = false;
}

function showAnswer(payload, requestId) {
  errorRegion.hidden = true;
  answerText.textContent = payload.answer;
  setRequestId(answerRequestId, requestId);
  renderCitations(payload.citations);
  answerRegion.hidden = false;
  answerRegion.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function showError(message, requestId) {
  answerRegion.hidden = true;
  errorText.textContent = message;
  setRequestId(errorRequestId, requestId);
  errorRegion.hidden = false;
}

function safeApiMessage(payload) {
  const candidate = payload && payload.error && payload.error.message;
  return SAFE_API_MESSAGES.has(candidate) ? candidate : INVALID_RESPONSE_MESSAGE;
}

function setLoading(isLoading) {
  submitButton.disabled = isLoading;
  submitButton.setAttribute("aria-busy", String(isLoading));
  submitLabel.textContent = isLoading ? "正在生成回答…" : "开始提问";
  formStatus.textContent = isLoading ? "请求处理中，请稍候。" : "";
}

async function checkHealth() {
  try {
    const response = await fetch("/health", {
      method: "GET",
      headers: { Accept: "application/json" },
    });
    const payload = await response.json();
    if (response.ok && payload && payload.ok === true && payload.status === "ok") {
      apiStatus.textContent = "API 可用";
      serviceStatus.dataset.state = "available";
      return;
    }
  } catch (_error) {
    // The visible status below is intentionally fixed and contains no error text.
  }
  apiStatus.textContent = "API 暂不可用";
  serviceStatus.dataset.state = "unavailable";
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (submitButton.disabled || !form.reportValidity()) {
    return;
  }

  hideResults();
  setLoading(true);

  const sessionId = document.querySelector("#session-id").value.trim();
  const payload = {
    question: document.querySelector("#question").value.trim(),
    episode_id: document.querySelector("#episode-id").value.trim(),
    top_k: Number(document.querySelector("#top-k").value),
  };
  if (sessionId) {
    payload.session_id = sessionId;
  }

  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
    const requestId = response.headers.get("X-Request-ID");
    let responsePayload;
    try {
      responsePayload = await response.json();
    } catch (_error) {
      showError(INVALID_RESPONSE_MESSAGE, requestId);
      return;
    }

    if (
      response.ok &&
      responsePayload &&
      responsePayload.ok === true &&
      typeof responsePayload.answer === "string" &&
      Array.isArray(responsePayload.citations)
    ) {
      showAnswer(responsePayload, requestId);
      return;
    }
    showError(safeApiMessage(responsePayload), requestId);
  } catch (_error) {
    showError(NETWORK_FAILURE_MESSAGE, null);
  } finally {
    setLoading(false);
  }
});

checkHealth();

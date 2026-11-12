"""Day 2 application entry point."""

import sys

from app.core.config import ConfigError, Settings
from app.core.errors import safe_error_message
from app.core.llm import ModelServiceError, create_client
from app.generation.analyzer import AnalysisOutputError, analyze_transcript


SAMPLE_TRANSCRIPT = """Dwarkesh Patel: Is scaling existing systems enough for future AI?
Demis Hassabis: Scaling remains important, but new architectural ideas may also be needed.
Dwarkesh Patel: So progress may require both stronger systems and further research.
"""


def main() -> None:
    """Analyze a short demonstration excerpt with validated JSON output."""
    try:
        settings = Settings.from_env()
        client = create_client(settings)
        analysis = analyze_transcript(client, settings, SAMPLE_TRANSCRIPT)
    except (ConfigError, ModelServiceError, AnalysisOutputError, ValueError) as error:
        print(f"运行失败：{safe_error_message(error)}", file=sys.stderr)
        raise SystemExit(1) from None

    print(analysis.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

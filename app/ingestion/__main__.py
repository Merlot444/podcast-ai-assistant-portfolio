"""Allow ``python -m app.ingestion`` to run the Day 3 pipeline."""

from app.ingestion.cli import main


if __name__ == "__main__":
    main()

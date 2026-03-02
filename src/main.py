"""Main entry point for the PDF translator."""

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.agents.graph import run_translation
from src.config import get_settings, setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Translate PDF documents using AI while preserving layout.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main data/book.pdf --language pt-BR
  python -m src.main data/manual.pdf --language es
  python -m src.main data/article.pdf --language fr --verbose
  python -m src.main data/book.pdf --language es --format docx
        """,
    )

    parser.add_argument(
        "input_pdf",
        type=Path,
        help="Path to the input PDF file",
    )

    parser.add_argument(
        "-l", "--language",
        type=str,
        default=None,
        help="Target language code (e.g., pt-BR, es, fr, de). "
             "Defaults to TARGET_LANGUAGE from .env",
    )

    parser.add_argument(
        "-f", "--format",
        type=str,
        choices=["pdf", "docx"],
        default="pdf",
        help="Output format: 'pdf' or 'docx' (default: pdf)",
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output file path (optional, auto-generated if not specified)",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


async def main() -> int:
    """Main entry point."""
    load_dotenv()

    args = parse_args()

    settings = get_settings()
    log_level = "DEBUG" if args.verbose else settings.log_level
    setup_logging(log_level)

    import logging
    logger = logging.getLogger(__name__)

    input_path = args.input_pdf
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    if not input_path.suffix.lower() == ".pdf":
        logger.error(f"Input file must be a PDF: {input_path}")
        return 1

    target_language = args.language or settings.target_language
    output_format = args.format

    logger.info(f"Starting translation of {input_path}")
    logger.info(f"Target language: {target_language}")
    logger.info(f"Output format: {output_format.upper()}")

    try:
        output_path = await run_translation(
            input_path=input_path,
            target_language=target_language,
            output_format=output_format,
        )

        logger.info(f"Translation completed successfully!")
        logger.info(f"Output saved to: {output_path}")
        print(f"\nTranslated {output_format.upper()} saved to: {output_path}")

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Translation failed: {e}")
        return 1


def cli() -> None:
    """CLI wrapper for main."""
    sys.exit(asyncio.run(main()))


if __name__ == "__main__":
    cli()

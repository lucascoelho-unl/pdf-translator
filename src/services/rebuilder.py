"""PDF reconstruction service using HTML intermediate format and WeasyPrint."""

import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML

from src.models.document import ExtractedDocument, ImageBlock, TranslatedChunk

logger = logging.getLogger(__name__)


class PDFRebuilder:
    """Rebuild translated PDFs using HTML/CSS and WeasyPrint."""

    def __init__(self) -> None:
        """Initialize the PDF rebuilder with Jinja2 templates."""
        template_dir = Path(__file__).parent.parent / "templates"
        self.env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=True,
        )

    def rebuild(
        self,
        document: ExtractedDocument,
        translated_chunks: list[TranslatedChunk],
        output_path: str | Path,
        target_language: str = "pt-BR",
    ) -> Path:
        """
        Rebuild the PDF with translated content.

        Args:
            document: Original extracted document with structure
            translated_chunks: Translated text chunks
            output_path: Path to save the output PDF
            target_language: Target language code for HTML lang attribute

        Returns:
            Path to the generated PDF
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Rebuilding PDF with {len(translated_chunks)} translated blocks...")

        translation_map = {chunk.id: chunk.translated_text for chunk in translated_chunks}

        blocks = self._prepare_blocks(document, translation_map)

        page_width, page_height = self._get_page_dimensions(document)

        html_content = self._render_html(
            blocks=blocks,
            page_width=page_width,
            page_height=page_height,
            language=target_language,
        )

        self._generate_pdf(html_content, output_path)

        logger.info(f"PDF successfully rebuilt: {output_path}")
        return output_path

    def _prepare_blocks(
        self,
        document: ExtractedDocument,
        translation_map: dict[str, str],
    ) -> list[dict]:
        """Prepare blocks for HTML rendering, merging text and images."""
        blocks = []

        for text_block in document.blocks:
            translated_text = translation_map.get(text_block.id, text_block.text)

            blocks.append({
                "type": "text",
                "block_type": text_block.block_type,
                "text": translated_text,
                "page_number": text_block.page_number,
                "reading_order": text_block.reading_order,
            })

        for image in document.images:
            blocks.append({
                "type": "image",
                "block_type": "image",
                "image_data": image.image_data,
                "image_format": image.image_format,
                "alt_text": image.alt_text,
                "page_number": image.page_number,
                "reading_order": image.reading_order,
            })

        blocks.sort(key=lambda b: (b["page_number"], b["reading_order"]))

        return blocks

    def _get_page_dimensions(
        self,
        document: ExtractedDocument,
    ) -> tuple[float, float]:
        """Get the most common page dimensions or default to A4."""
        if not document.page_dimensions:
            return (210.0, 297.0)

        dims = list(document.page_dimensions.values())
        if dims:
            return dims[0]

        return (210.0, 297.0)

    def _render_html(
        self,
        blocks: list[dict],
        page_width: float,
        page_height: float,
        language: str,
    ) -> str:
        """Render blocks to HTML using Jinja2 template."""
        template = self.env.get_template("document.html")

        return template.render(
            blocks=blocks,
            page_width=page_width,
            page_height=page_height,
            language=language,
        )

    def _generate_pdf(self, html_content: str, output_path: Path) -> None:
        """Generate PDF from HTML using WeasyPrint."""
        html = HTML(string=html_content)
        html.write_pdf(str(output_path))


def rebuild_pdf(
    document: ExtractedDocument,
    translated_chunks: list[TranslatedChunk],
    output_path: str | Path,
    target_language: str = "pt-BR",
) -> Path:
    """
    Convenience function to rebuild a PDF.

    Args:
        document: Original extracted document
        translated_chunks: Translated chunks
        output_path: Output PDF path
        target_language: Target language code

    Returns:
        Path to generated PDF
    """
    rebuilder = PDFRebuilder()
    return rebuilder.rebuild(
        document=document,
        translated_chunks=translated_chunks,
        output_path=output_path,
        target_language=target_language,
    )

"""Document extraction service using Docling for layout analysis."""

import base64
import logging
import uuid
from pathlib import Path

import fitz  # PyMuPDF
from docling.document_converter import DocumentConverter

from src.models.document import ExtractedDocument, ImageBlock, TextBlock

logger = logging.getLogger(__name__)


class DocumentExtractor:
    """Extract structured content from PDFs using Document Layout Analysis."""

    def __init__(self) -> None:
        """Initialize the document extractor with Docling converter."""
        self.converter = DocumentConverter()

    def extract(self, pdf_path: str | Path) -> ExtractedDocument:
        """
        Extract text blocks and images from a PDF with layout information.

        Args:
            pdf_path: Path to the PDF file to extract

        Returns:
            ExtractedDocument containing all blocks, images, and metadata
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"Extracting document: {pdf_path}")

        result = self.converter.convert(str(pdf_path))
        doc = result.document

        text_blocks: list[TextBlock] = []
        images: list[ImageBlock] = []
        page_dimensions: dict[int, tuple[float, float]] = {}

        pdf_doc = fitz.open(str(pdf_path))
        try:
            for page_num, page in enumerate(pdf_doc, start=1):
                rect = page.rect
                page_dimensions[page_num] = (
                    rect.width * 0.352778,
                    rect.height * 0.352778,
                )

            page_images = self._extract_images(pdf_doc)
            images.extend(page_images)
        finally:
            pdf_doc.close()

        reading_order = 0
        for item in doc.iterate_items():
            element = item
            if hasattr(item, "text") and item.text:
                block_type = self._map_element_type(element)

                page_num = 1
                bbox = None
                if hasattr(element, "prov") and element.prov:
                    prov = element.prov[0] if element.prov else None
                    if prov:
                        page_num = getattr(prov, "page_no", 1) or 1
                        if hasattr(prov, "bbox"):
                            bbox_obj = prov.bbox
                            if bbox_obj:
                                bbox = (
                                    float(bbox_obj.l),
                                    float(bbox_obj.t),
                                    float(bbox_obj.r),
                                    float(bbox_obj.b),
                                )

                text_block = TextBlock(
                    id=str(uuid.uuid4()),
                    text=element.text.strip(),
                    block_type=block_type,
                    page_number=page_num,
                    bbox=bbox,
                    reading_order=reading_order,
                )
                text_blocks.append(text_block)
                reading_order += 1

        logger.info(
            f"Extracted {len(text_blocks)} text blocks and {len(images)} images "
            f"from {len(page_dimensions)} pages"
        )

        return ExtractedDocument(
            source_path=str(pdf_path),
            total_pages=len(page_dimensions),
            blocks=text_blocks,
            images=images,
            page_dimensions=page_dimensions,
        )

    def _map_element_type(self, element) -> str:
        """Map Docling element types to our block types."""
        element_type = type(element).__name__.lower()

        type_mapping = {
            "title": "title",
            "sectionheader": "heading",
            "section_header": "heading",
            "heading": "heading",
            "paragraph": "paragraph",
            "text": "paragraph",
            "listitem": "list_item",
            "list_item": "list_item",
            "caption": "caption",
            "table": "table",
            "footer": "footer",
            "header": "header",
            "pagenumber": "footer",
        }

        for key, value in type_mapping.items():
            if key in element_type:
                return value

        return "paragraph"

    def _extract_images(self, pdf_doc: fitz.Document) -> list[ImageBlock]:
        """Extract images from PDF pages."""
        images: list[ImageBlock] = []

        for page_num, page in enumerate(pdf_doc, start=1):
            image_list = page.get_images()

            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = pdf_doc.extract_image(xref)

                    if base_image:
                        image_data = base64.b64encode(base_image["image"]).decode("utf-8")
                        image_format = base_image.get("ext", "png")

                        bbox = None
                        for img_rect in page.get_image_rects(xref):
                            bbox = (
                                float(img_rect.x0),
                                float(img_rect.y0),
                                float(img_rect.x1),
                                float(img_rect.y1),
                            )
                            break

                        image_block = ImageBlock(
                            id=str(uuid.uuid4()),
                            page_number=page_num,
                            bbox=bbox,
                            image_data=image_data,
                            image_format=image_format,
                            reading_order=img_index,
                        )
                        images.append(image_block)
                except Exception as e:
                    logger.warning(f"Failed to extract image on page {page_num}: {e}")

        return images

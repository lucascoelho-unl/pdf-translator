"""Document models for extracted PDF content."""

from typing import Literal

from pydantic import BaseModel, Field


class TextBlock(BaseModel):
    """A block of text extracted from a PDF with layout metadata."""

    id: str = Field(description="Unique identifier for this block")
    text: str = Field(description="The text content of this block")
    block_type: Literal[
        "title",
        "heading",
        "section_header",
        "paragraph",
        "caption",
        "list_item",
        "table",
        "footer",
        "header",
    ] = Field(description="The semantic type of this text block")
    page_number: int = Field(description="Page number where this block appears (1-indexed)")
    bbox: tuple[float, float, float, float] | None = Field(
        default=None,
        description="Bounding box coordinates (x0, y0, x1, y1) in points",
    )
    font_size: float | None = Field(
        default=None,
        description="Approximate font size in points",
    )
    reading_order: int = Field(
        default=0,
        description="Order in which this block should be read",
    )


class ImageBlock(BaseModel):
    """An image extracted from a PDF."""

    id: str = Field(description="Unique identifier for this image")
    page_number: int = Field(description="Page number where this image appears")
    bbox: tuple[float, float, float, float] | None = Field(
        default=None,
        description="Bounding box coordinates (x0, y0, x1, y1) in points",
    )
    image_data: str = Field(description="Base64-encoded image data")
    image_format: str = Field(default="png", description="Image format (png, jpeg, etc.)")
    alt_text: str = Field(default="", description="Alternative text description")
    reading_order: int = Field(
        default=0,
        description="Order in which this image appears in reading flow",
    )


class ExtractedDocument(BaseModel):
    """A complete document extracted from a PDF with layout information."""

    source_path: str = Field(description="Path to the original PDF file")
    total_pages: int = Field(description="Total number of pages in the document")
    blocks: list[TextBlock] = Field(
        default_factory=list,
        description="All text blocks extracted from the document",
    )
    images: list[ImageBlock] = Field(
        default_factory=list,
        description="All images extracted from the document",
    )
    page_dimensions: dict[int, tuple[float, float]] = Field(
        default_factory=dict,
        description="Page dimensions (width, height) in mm for each page number",
    )

    def get_full_text(self) -> str:
        """Return all text content concatenated in reading order."""
        sorted_blocks = sorted(self.blocks, key=lambda b: (b.page_number, b.reading_order))
        return "\n\n".join(block.text for block in sorted_blocks)

    def get_text_sample(self, num_blocks: int = 20) -> str:
        """Return a representative sample of text from the document."""
        sorted_blocks = sorted(self.blocks, key=lambda b: (b.page_number, b.reading_order))
        if len(sorted_blocks) <= num_blocks:
            return "\n\n".join(block.text for block in sorted_blocks)

        sample_indices = [
            0,
            len(sorted_blocks) // 4,
            len(sorted_blocks) // 2,
            3 * len(sorted_blocks) // 4,
            len(sorted_blocks) - 1,
        ]
        blocks_per_section = num_blocks // len(sample_indices)

        sampled_text = []
        for idx in sample_indices:
            start = max(0, idx)
            end = min(len(sorted_blocks), start + blocks_per_section)
            for block in sorted_blocks[start:end]:
                sampled_text.append(block.text)

        return "\n\n".join(sampled_text)


class TranslationChunk(BaseModel):
    """A chunk of text prepared for translation."""

    id: str = Field(description="Unique identifier matching the original TextBlock")
    text: str = Field(description="Original text to translate")
    block_type: str = Field(description="Type of block for context")
    previous_context: str = Field(
        default="",
        description="Previously translated text for context continuity",
    )


class TranslatedChunk(BaseModel):
    """A translated chunk of text."""

    id: str = Field(description="Unique identifier matching the original TextBlock")
    original_text: str = Field(description="Original text before translation")
    translated_text: str = Field(description="Translated text")
    block_type: str = Field(description="Type of block")

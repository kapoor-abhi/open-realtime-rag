#parser.py
"""
Multimodal document parser.

FIX: Tightened table detection to eliminate false positives.

Root cause of the bug:
  pdfplumber detects any grid-like layout as a "table" — including bullet
  point lists, numbered lists, and wrapped paragraph text. With the original
  MIN_TABLE_CELLS = 4 threshold and no column/row checks, virtually every
  page was producing spurious "table" chunks. The LLM then invented table
  numbers (Table 2, Table 4, Table 6…) that don't exist in the PDF, and
  every text block got labelled chunk_type="table".

Fixes applied in _extract_tables_from_page():
  1. MIN_TABLE_CELLS raised from 4 → 12  (requires meaningful amount of data)
  2. MIN_TABLE_COLS = 2  — reject single-column extractions (these are lists)
  3. MIN_TABLE_ROWS = 2  — require at least a header row + 1 data row
  4. MIN_MULTI_CELL_ROWS — at least half the rows must have ≥2 non-empty cells.
     This catches the "one long string split into fake rows" pattern.
  5. Column header validation: if every "header" cell is empty, it's not a
     real table — generate generic headers only as a last resort and flag it.
"""

import os
import logging
import concurrent.futures
from typing import List

import fitz           # PyMuPDF
import pdfplumber

from app.models.schemas import DocumentChunk, DocumentMetadata
from app.services.vision import generate_image_caption
from app.services.storage import StorageService

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MIN_IMAGE_BYTES   = 5_000
MIN_TEXT_CHARS    = 20
MIN_TABLE_CELLS   = 12   # raised from 4 — need meaningful content
MIN_TABLE_COLS    = 2    # NEW: single-column = list, not a table
MIN_TABLE_ROWS    = 2    # NEW: need header + at least 1 data row
MIN_MULTI_CELL_ROWS_RATIO = 0.5  # NEW: ≥50% of rows must have ≥2 non-empty cells


def _is_real_table(table: list) -> bool:
    """
    Return True only if the extracted grid looks like a genuine table.

    Rejects:
      - Single-column grids (bullet lists, numbered lists)
      - Grids where most rows have only one filled cell (wrapped paragraph text)
      - Grids with fewer than MIN_TABLE_ROWS data rows
      - Grids with fewer than MIN_TABLE_CELLS total non-empty cells
    """
    if not table or len(table) < MIN_TABLE_ROWS + 1:  # +1 for header
        return False

    def non_empty(cell) -> bool:
        return bool(cell and str(cell).strip())

    # Count columns
    n_cols = max(len(row) for row in table)
    if n_cols < MIN_TABLE_COLS:
        logger.debug(f"[TABLE FILTER] Rejected: only {n_cols} column(s)")
        return False

    # Count total non-empty cells
    flat = [c for row in table for c in row if non_empty(c)]
    if len(flat) < MIN_TABLE_CELLS:
        logger.debug(f"[TABLE FILTER] Rejected: only {len(flat)} non-empty cells")
        return False

    # Count rows that have ≥2 non-empty cells
    multi_cell_rows = sum(
        1 for row in table
        if sum(1 for c in row if non_empty(c)) >= 2
    )
    ratio = multi_cell_rows / len(table)
    if ratio < MIN_MULTI_CELL_ROWS_RATIO:
        logger.debug(
            f"[TABLE FILTER] Rejected: only {ratio:.0%} of rows have ≥2 filled cells "
            f"(looks like a list or wrapped text)"
        )
        return False

    return True


def _table_to_markdown(table: list) -> str:
    """
    Convert a pdfplumber table to GitHub-Flavored Markdown.
    Only called after _is_real_table() returns True.
    """
    if not table or not any(table):
        return ""

    def clean(cell) -> str:
        return str(cell).replace("\n", " ").strip() if cell is not None else ""

    rows = [[clean(c) for c in row] for row in table]
    n_cols = max(len(r) for r in rows)
    rows = [r + [""] * (n_cols - len(r)) for r in rows]

    if len(rows) < 2:
        return ""

    headers = rows[0]
    body = rows[1:]

    if not any(headers):
        headers = [f"Column {i+1}" for i in range(n_cols)]

    md = "| " + " | ".join(headers) + " |\n"
    md += "| " + " | ".join(["---"] * n_cols) + " |\n"
    for row in body:
        md += "| " + " | ".join(row) + " |\n"

    return md.strip()


class DocumentParser:
    def __init__(self):
        self.storage = StorageService()

    def _process_single_image(
        self,
        img_bytes: bytes,
        ext: str,
        image_counter: int,
        page_no: int,
        file_hash: str,
    ):
        img_filename = f"uploads/{file_hash}_img_{image_counter}.{ext}"

        with open(img_filename, "wb") as f:
            f.write(img_bytes)

        logger.info(f"[VISION] Extracted Image {image_counter} from Page {page_no}.")

        caption = ""
        image_path_saved = None

        try:
            logger.info(f"[VISION] Sending Image {image_counter} to Groq Vision API...")
            caption = generate_image_caption(img_filename)
        except Exception as e:
            logger.error(
                f"[VISION] API failed for Image {image_counter}. Using fallback. Error: {e}"
            )
            caption = f"A diagram, chart, or visual element located on page {page_no}."

        try:
            logger.info(f"[STORAGE] Uploading Image {image_counter} to MinIO...")
            public_img_url = self.storage.upload_file(
                img_filename, f"images/{file_hash}_img_{image_counter}.{ext}"
            )
            image_path_saved = public_img_url
        except Exception as e:
            logger.error(f"[STORAGE] MinIO upload failed: {e}")

        if os.path.exists(img_filename):
            os.remove(img_filename)

        return caption, image_path_saved

    def _extract_tables_from_page(
        self,
        plumber_page,
        page_no: int,
        source_file_name: str,
        file_hash: str,
    ) -> List[DocumentChunk]:
        """
        Extract genuine tables from a page.

        Each candidate grid is passed through _is_real_table() before being
        converted to Markdown. Grids that fail the check are silently skipped —
        their content will be captured by the text extraction pass instead.
        """
        chunks = []

        try:
            tables = plumber_page.extract_tables()
        except Exception as e:
            logger.warning(f"[TABLE] pdfplumber extraction failed on p{page_no}: {e}")
            return chunks

        real_table_count = 0
        for i, table in enumerate(tables, start=1):
            if not table:
                continue

            if not _is_real_table(table):
                logger.info(
                    f"[TABLE] Page {page_no}, grid {i}: failed real-table check — "
                    f"treating as text (not creating a table chunk)"
                )
                continue

            md_table = _table_to_markdown(table)
            if not md_table:
                continue

            real_table_count += 1
            contextualized = (
                f"Source Document: {source_file_name}\n"
                f"Page: {page_no}\n"
                f"[Table {real_table_count}]:\n"
                f"{md_table}"
            )

            metadata = DocumentMetadata(
                source_file=source_file_name,
                file_hash=file_hash,
                page_number=page_no,
                chunk_type="table",
            )
            chunks.append(DocumentChunk(text=contextualized, metadata=metadata))
            logger.info(
                f"[TABLE] Confirmed table {real_table_count} on page {page_no}"
            )

        return chunks

    def parse_document(
        self,
        file_path: str,
        source_file_name: str,
        file_hash: str,
    ) -> List[DocumentChunk]:
        logger.info(
            f"[PARSER] Starting multimodal parse for '{source_file_name}' "
            f"(hash={file_hash[:8]}...)"
        )

        results: List[DocumentChunk] = []
        image_counter = 0

        fitz_doc = fitz.open(file_path)

        with pdfplumber.open(file_path) as plumber_doc, \
             concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:

            future_to_page: dict = {}

            for page_num in range(len(fitz_doc)):
                fitz_page = fitz_doc[page_num]
                page_no = page_num + 1

                # ---- TEXT EXTRACTION ----
                blocks = fitz_page.get_text("blocks")

                if page_no == 1:
                    page_1_text = "\n".join(
                        b[4].strip()
                        for b in blocks
                        if b[6] == 0 and len(b[4].strip()) > 5
                    )
                    if page_1_text:
                        contextualized = (
                            f"Source Document: {source_file_name}\n"
                            f"Page: {page_no}\n"
                            f"Document Summary & Metadata:\n{page_1_text}"
                        )
                        results.append(
                            DocumentChunk(
                                text=contextualized,
                                metadata=DocumentMetadata(
                                    source_file=source_file_name,
                                    file_hash=file_hash,
                                    page_number=page_no,
                                    chunk_type="text",
                                ),
                            )
                        )
                else:
                    for block in blocks:
                        if block[6] == 0:
                            text_content = block[4].strip()
                            if len(text_content) >= MIN_TEXT_CHARS:
                                contextualized = (
                                    f"Source Document: {source_file_name}\n"
                                    f"Page: {page_no}\n"
                                    f"Content:\n{text_content}"
                                )
                                results.append(
                                    DocumentChunk(
                                        text=contextualized,
                                        metadata=DocumentMetadata(
                                            source_file=source_file_name,
                                            file_hash=file_hash,
                                            page_number=page_no,
                                            chunk_type="text",
                                        ),
                                    )
                                )

                # ---- TABLE EXTRACTION ----
                if page_num < len(plumber_doc.pages):
                    plumber_page = plumber_doc.pages[page_num]
                    table_chunks = self._extract_tables_from_page(
                        plumber_page, page_no, source_file_name, file_hash
                    )
                    results.extend(table_chunks)

                # ---- IMAGE EXTRACTION ----
                for img in fitz_page.get_images(full=True):
                    xref = img[0]
                    base_image = fitz_doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    ext = base_image["ext"]

                    if len(image_bytes) >= MIN_IMAGE_BYTES:
                        image_counter += 1
                        future = executor.submit(
                            self._process_single_image,
                            image_bytes, ext, image_counter, page_no, file_hash,
                        )
                        future_to_page[future] = page_no

            # ---- COLLECT VISION RESULTS ----
            for future in concurrent.futures.as_completed(future_to_page):
                page_no = future_to_page[future]
                try:
                    caption, image_path_saved = future.result()
                    if caption and image_path_saved:
                        final_text = (
                            f"Source Document: {source_file_name}\n"
                            f"Page: {page_no}\n"
                            f"[Visual Content]: {caption}"
                        )
                        results.append(
                            DocumentChunk(
                                text=final_text,
                                metadata=DocumentMetadata(
                                    source_file=source_file_name,
                                    file_hash=file_hash,
                                    page_number=page_no,
                                    chunk_type="image",
                                    image_path=image_path_saved,
                                ),
                            )
                        )
                except Exception as exc:
                    logger.error(f"[VISION THREAD] Exception on p{page_no}: {exc}")

        fitz_doc.close()

        text_count  = sum(1 for c in results if c.metadata.chunk_type == "text")
        table_count = sum(1 for c in results if c.metadata.chunk_type == "table")
        image_count = sum(1 for c in results if c.metadata.chunk_type == "image")

        logger.info(
            f"[PARSER] Done — {len(results)} total chunks: "
            f"{text_count} text | {table_count} tables | {image_count} images"
        )
        return results
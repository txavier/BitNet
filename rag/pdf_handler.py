"""
PDF handler for the RAG pipeline.
Extracts text from PDFs using PyMuPDF (text-based) with OCR fallback
(pytesseract + pdf2image) for scanned/image-based pages.

Can be used standalone or imported by index_data.py and scrape_web.py.

Usage:
    # Extract text from a PDF and save as MD
    python rag/pdf_handler.py document.pdf -o rag/sampling_data/

    # Process a directory of PDFs
    python rag/pdf_handler.py /path/to/pdfs/ -o rag/sampling_data/

    # Then index
    python rag/index_data.py rag/sampling_data/
"""

import argparse
import os
import sys


def extract_text_from_pdf(pdf_path, ocr_threshold=30, workers=None):
    """
    Extract text from a PDF file.

    Uses PyMuPDF for direct text extraction. Falls back to OCR
    (pytesseract + pdf2image) for pages with less text than ocr_threshold chars.

    Args:
        pdf_path: Path to the PDF file.
        ocr_threshold: Minimum chars per page before OCR fallback triggers.
        workers: Parallel OCR threads (default: min(pages_needing_ocr, 4)).

    Returns:
        List of (page_number, text) tuples.
    """
    import fitz  # pymupdf

    pages = []
    doc = fitz.open(pdf_path)

    ocr_needed = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()

        if len(text) >= ocr_threshold:
            pages.append((page_num + 1, text))
        else:
            ocr_needed.append(page_num + 1)  # 1-indexed

    doc.close()

    # OCR fallback for pages with little/no text
    if ocr_needed:
        ocr_pages = _ocr_pages(pdf_path, ocr_needed, workers=workers)
        pages.extend(ocr_pages)

    # Sort by page number
    pages.sort(key=lambda x: x[0])
    return pages


def _ocr_pages(pdf_path, page_numbers, workers=None):
    """OCR specific pages of a PDF using pytesseract + pdf2image.

    Args:
        pdf_path: Path to the PDF file.
        page_numbers: List of 1-indexed page numbers to OCR.
        workers: Number of parallel threads (default: min(len(pages), 4)).
    """
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except ImportError:
        print("  Warning: pytesseract/pdf2image not installed. Skipping OCR for scanned pages.")
        return []

    from concurrent.futures import ThreadPoolExecutor, as_completed

    pages = []
    print(f"  OCR needed for {len(page_numbers)} page(s): {page_numbers}")

    effective_workers = workers or min(len(page_numbers), 4)

    def _ocr_one(page_num):
        images = convert_from_path(
            pdf_path,
            first_page=page_num,
            last_page=page_num,
            dpi=300,
        )
        if images:
            text = pytesseract.image_to_string(images[0]).strip()
            return page_num, text
        return page_num, ""

    if effective_workers > 1 and len(page_numbers) > 1:
        print(f"  OCR with {effective_workers} parallel workers...")
        with ThreadPoolExecutor(max_workers=effective_workers) as pool:
            futures = {pool.submit(_ocr_one, pn): pn for pn in page_numbers}
            for future in as_completed(futures):
                try:
                    page_num, text = future.result()
                    if text:
                        pages.append((page_num, text))
                        print(f"    Page {page_num}: OCR extracted {len(text)} chars")
                    else:
                        print(f"    Page {page_num}: OCR found no text")
                except Exception as e:
                    pn = futures[future]
                    print(f"    Page {pn}: OCR failed: {e}")
    else:
        for page_num in page_numbers:
            try:
                _, text = _ocr_one(page_num)
                if text:
                    pages.append((page_num, text))
                    print(f"    Page {page_num}: OCR extracted {len(text)} chars")
                else:
                    print(f"    Page {page_num}: OCR found no text")
            except Exception as e:
                print(f"    Page {page_num}: OCR failed: {e}")

    return pages


def pdf_to_markdown(pdf_path, ocr_threshold=30, workers=None):
    """
    Convert a PDF to a markdown string.

    Args:
        pdf_path: Path to the PDF file.
        ocr_threshold: Min chars per page before OCR kicks in.
        workers: Parallel OCR threads.

    Returns:
        Markdown string with page sections, or None if no text extracted.
    """
    filename = os.path.basename(pdf_path)
    pages = extract_text_from_pdf(pdf_path, ocr_threshold, workers=workers)

    if not pages:
        return None

    sections = []
    sections.append(f"# {filename}\n")

    for page_num, text in pages:
        sections.append(f"## Page {page_num}\n")
        sections.append(text)
        sections.append("")

    return "\n".join(sections)


def pdf_to_chunks(pdf_path, ocr_threshold=30, workers=None):
    """
    Convert a PDF to a list of text chunks suitable for RAG indexing.
    Each page becomes a chunk prefixed with the source filename.

    Args:
        pdf_path: Path to the PDF file.
        ocr_threshold: Min chars per page before OCR kicks in.
        workers: Parallel OCR threads.

    Returns:
        List of text strings (one per page).
    """
    filename = os.path.basename(pdf_path)
    pages = extract_text_from_pdf(pdf_path, ocr_threshold, workers=workers)

    chunks = []
    for page_num, text in pages:
        if len(text.strip()) > 20:
            chunks.append(f"[{filename} p.{page_num}] {text}")
    return chunks


def process_pdf(pdf_path, output_dir, workers=None):
    """Process a single PDF and save as markdown file."""
    filename = os.path.basename(pdf_path)
    print(f"  Processing: {filename}")

    md_content = pdf_to_markdown(pdf_path, workers=workers)
    if not md_content:
        print(f"  No text extracted from {filename}")
        return False

    md_filename = os.path.splitext(filename)[0] + ".md"
    md_path = os.path.join(output_dir, md_filename)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    char_count = len(md_content)
    print(f"  Saved: {md_filename} ({char_count} chars)")
    return True


def process_directory(dir_path, output_dir, workers=None):
    """Process all PDFs in a directory."""
    pdf_files = sorted(
        f for f in os.listdir(dir_path)
        if f.lower().endswith(".pdf") and os.path.isfile(os.path.join(dir_path, f))
    )

    if not pdf_files:
        print(f"No PDF files found in {dir_path}")
        return 0

    print(f"Found {len(pdf_files)} PDF(s) in {dir_path}")
    processed = 0
    for filename in pdf_files:
        pdf_path = os.path.join(dir_path, filename)
        if process_pdf(pdf_path, output_dir, workers=workers):
            processed += 1

    print(f"\nProcessed {processed}/{len(pdf_files)} PDFs -> {output_dir}/")
    return processed


def main():
    parser = argparse.ArgumentParser(
        description="Extract text from PDFs for RAG indexing (with OCR fallback)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single PDF
  python rag/pdf_handler.py manual.pdf -o rag/sampling_data/

  # Directory of PDFs
  python rag/pdf_handler.py /path/to/pdfs/ -o rag/sampling_data/

  # Then index the extracted text
  python rag/index_data.py rag/sampling_data/
        """,
    )
    parser.add_argument("input", help="PDF file or directory of PDFs")
    parser.add_argument("-o", "--output", default="rag/sampling_data",
                        help="Output directory for MD files (default: rag/sampling_data)")
    parser.add_argument("--ocr-threshold", type=int, default=30,
                        help="Min chars per page before OCR fallback (default: 30)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel OCR threads per PDF (default: min(pages, 4))")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if os.path.isdir(args.input):
        count = process_directory(args.input, args.output, workers=args.workers)
    elif os.path.isfile(args.input):
        if not args.input.lower().endswith(".pdf"):
            print(f"Error: Not a PDF file: {args.input}")
            sys.exit(1)
        count = 1 if process_pdf(args.input, args.output, workers=args.workers) else 0
    else:
        print(f"Error: Path not found: {args.input}")
        sys.exit(1)

    if count:
        print(f"\nIndex with: python rag/index_data.py {args.output}")


if __name__ == "__main__":
    main()

"""Extract physics-of-llm-inference.pdf into per-chapter markdown files under book/.

Uses the PDF's top-level bookmarks (Introduction, Chapter 1..10, Appendix) as
chapter boundaries. Each chapter's pages run from its bookmark to the next
top-level bookmark (exclusive); the last chapter runs to the final page.
"""

import re
import sys
from pathlib import Path

from pypdf import PdfReader

PDF_PATH = Path(r"C:\Users\barry\Desktop\ml-infra-journey\physics-of-llm-inference.pdf")
OUT_DIR = Path(__file__).resolve().parent.parent / "book"


def slugify(title: str) -> str:
    name = re.sub(r"^Chapter\s+\d+\s*:\s*", "", title)
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "-", name).strip("-")
    return name


def top_level_outline(reader: PdfReader):
    """Flatten the outline to (title, page_index) for depth-0 entries only."""
    entries = []
    # pypdf nests children as lists following their parent item; keep depth 0 only
    for item in reader.outline:
        if isinstance(item, list):
            continue
        entries.append((item.title, reader.get_destination_page_number(item)))
    return entries


def main() -> int:
    reader = PdfReader(str(PDF_PATH))
    entries = top_level_outline(reader)
    if not entries:
        print("ERROR: no top-level outline entries found", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(exist_ok=True)
    n_pages = len(reader.pages)
    report = []

    for i, (title, start) in enumerate(entries):
        end = entries[i + 1][1] if i + 1 < len(entries) else n_pages
        text = "\n\n".join(
            reader.pages[p].extract_text() or "" for p in range(start, end)
        )
        fname = f"{i:02d}-{slugify(title)}.md"
        header = (
            f"# {title}\n\n"
            f"> Source: physics-of-llm-inference.pdf, pages {start + 1}-{end} "
            f"({end - start} pages). Verbatim text extraction.\n\n"
        )
        (OUT_DIR / fname).write_text(header + text + "\n", encoding="utf-8")
        words = len(text.split())
        report.append(f"{fname}  pages {start + 1}-{end}  {words} words")

    for line in report:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())

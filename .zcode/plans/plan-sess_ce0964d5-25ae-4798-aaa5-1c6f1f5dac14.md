Ingest `C:\Users\barry\Desktop\ml-infra-journey\physics-of-llm-inference.pdf` into this repo as per-chapter verbatim reference files.

## Steps

1. **Extract chapters** — small Python script using `pypdf` (pip-install if missing) that:
   - Reads the PDF's outline/bookmarks for chapter boundaries; falls back to detecting "Chapter N" headings in page text if no outline exists.
   - Writes one markdown file per chapter to `book/NN-chapter-slug.md` — a short header (chapter title, page range) followed by the verbatim extracted text.

2. **Build `book/INDEX.md`** — table listing each chapter's number, title, page range, file, and the matching hands-on folder already in this repo (e.g., flash-attention → `flash-attention/`, tensor-cores → `tensor-cores/`), plus a note on extraction quality.

3. **Keep it out of git** — create `.gitignore` in the repo root containing `book/`.

4. **Make it discoverable** — add a short "Book reference" section to `README.md` pointing at `book/INDEX.md`, so any future session knows the full chapter texts are available to read on demand.

5. **Ingest in this session** — after extraction, read `INDEX.md` and spot-check 2–3 chapter files for extraction quality. From then on I can pull any chapter verbatim while working on its matching folder.

## Caveats
- PDF text extraction loses figures and can garble code/math formatting; I'll flag quality issues in the index.
- This makes the chapters durably available as context without pasting a whole book into one conversation (which wouldn't fit) — files are read per-chapter as needed.

## Permissions needed
- Run Python scripts to extract PDF text (pip install pypdf if not present)
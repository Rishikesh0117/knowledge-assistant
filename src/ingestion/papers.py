import re
import pymupdf4llm
from src import config


def _title_from_filename(path):
    return path.stem.replace("_", " ").title()


def _split_sections(md):
    """Split markdown on headings -> list of (heading, body)."""
    parts = re.split(r"(?m)^(#{1,4} .+)$", md)
    sections = []
    if parts[0].strip():
        sections.append(("Preamble", parts[0]))
    for i in range(1, len(parts), 2):
        heading = parts[i].lstrip("# ").strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        sections.append((heading, body))
    return sections


def _pack(text, max_words):
    """Pack paragraphs into chunks of at most ~max_words words."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur, n = [], [], 0
    for p in paras:
        w = len(p.split())
        if cur and n + w > max_words:
            chunks.append("\n\n".join(cur))
            cur, n = [], 0
        cur.append(p)
        n += w
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def load_papers(max_words=450):
    chunks = []
    for pdf in sorted(config.PAPERS_DIR.glob("*.pdf")):
        md = pymupdf4llm.to_markdown(str(pdf))
        for heading, body in _split_sections(md):
            for i, c in enumerate(_pack(body, max_words)):
                if len(c.split()) < 30:  # drop tiny fragments
                    continue
                chunks.append({
                    "text": c,
                    "metadata": {
                        "source_type": "paper",
                        "source": pdf.stem,
                        "title": _title_from_filename(pdf),
                        "section": heading,
                        "chunk": i,
                    },
                })
    return chunks


if __name__ == "__main__":
    cs = load_papers()
    print(len(cs), "chunks")
    for c in cs[:3] + cs[-2:]:
        print(c["metadata"], "\n", c["text"][:300], "\n---")
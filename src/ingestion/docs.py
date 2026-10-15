import re
from src import config


def _blocks(text):
    """Paragraph blocks; a fenced code block is kept whole."""
    blocks, cur, in_code = [], [], False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            cur.append(line)
            if not in_code:
                blocks.append("\n".join(cur)); cur = []
            continue
        if in_code:
            cur.append(line)
        elif line.strip() == "":
            if cur:
                blocks.append("\n".join(cur)); cur = []
        else:
            cur.append(line)
    if cur:
        blocks.append("\n".join(cur))
    return blocks


def _sections(text):
    """Split on markdown headings that sit outside code fences."""
    sections, heading, buf, in_code = [], "Intro", [], False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        if not in_code and re.match(r"^#{1,4} ", line):
            if buf:
                sections.append((heading, "\n".join(buf)))
            heading, buf = line.lstrip("# ").strip(), []
        else:
            buf.append(line)
    if buf:
        sections.append((heading, "\n".join(buf)))
    return sections


def _pack(blocks, max_words):
    chunks, cur, n = [], [], 0
    for b in blocks:
        w = len(b.split())
        if cur and n + w > max_words:
            chunks.append("\n\n".join(cur)); cur, n = [], 0
        cur.append(b); n += w
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def load_docs(max_words=450):
    chunks = []
    for md in sorted(config.DOCS_DIR.rglob("*.md")):
        text = md.read_text(encoding="utf-8", errors="ignore")
        for heading, body in _sections(text):
            for i, c in enumerate(_pack(_blocks(body), max_words)):
                if len(c.split()) < 25:
                    continue
                chunks.append({
                    "text": c,
                    "metadata": {
                        "source_type": "docs",
                        "source": md.stem,
                        "title": md.stem.replace("_", " ").replace("-", " ").title(),
                        "section": heading,
                        "chunk": i,
                    },
                })
    return chunks


if __name__ == "__main__":
    cs = load_docs()
    print(len(cs), "chunks")
from src.ingestion.papers import _pack, _split_sections


def test_pack_respects_max_words():
    text = "\n\n".join(["word " * 100] * 6)
    chunks = _pack(text, 250)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 250 for c in chunks)


def test_split_sections_finds_headings():
    md = "intro text\n\n## Method\nbody one\n\n## Results\nbody two"
    assert [h for h, _ in _split_sections(md)] == ["Preamble", "Method", "Results"]
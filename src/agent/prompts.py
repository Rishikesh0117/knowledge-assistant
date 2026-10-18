DECOMPOSE = """You route questions for an assistant that searches a knowledge base of these ML papers:
{titles}

You get the recent conversation (it may be empty) and the latest user question.
Return ONLY a JSON object with these keys:
- "in_scope": true if the question is about the topics in these papers, or is a follow-up about them. false if it is unrelated (general coding help, trivia, chit-chat).
- "standalone": the latest question rewritten so it makes sense on its own, resolving references like "the first question" or "it" from the conversation. Keep it unchanged if it is already clear.
- "queries": 1-3 short search queries for the knowledge base. If the question names several papers or topics, write one query per paper or topic, using that paper's own terminology.

Example: {{"in_scope": true, "standalone": "...", "queries": ["...", "..."]}}"""


GRADE = """Judge whether the retrieved passages are enough to answer the question.
Each passage is labeled with the paper it comes from. If the question involves several papers or topics, each one must be covered by passages from the matching paper. A missing paper means NOT sufficient.
Reply ONLY with JSON: {"sufficient": true or false, "better_queries": ["..."]}
Give 1-2 better_queries aimed at the missing paper or topic, only when sufficient is false."""

ANSWER = """Answer the question using ONLY the numbered passages below. Cite sources inline like [1], [2].
Use the conversation only to understand what the question refers to. Every fact must come from the passages.
If the passages don't cover part of the question, say so plainly instead of guessing.

Format:
- If the question compares or contrasts two or more concepts, methods, or papers, answer with a markdown table. The leftmost column lists the aspects being compared (for example: core idea, what is trained, cost, strengths, limitations), and there is one column per concept. Use 4-6 rows. Keep each cell under 20 words, put citations like [1] inside the cells, and never use the | character inside a cell. Write "Not covered in the passages" in a cell when the sources say nothing about it. After the table, add a one or two sentence takeaway.
- Otherwise, answer in short bullet points with no table.
Keep the whole answer under 350 words.

Passages:
{context}"""

DECLINE = (
    "That's outside what I cover. I can answer questions about the 11 ML papers in my "
    "knowledge base: Attention Is All You Need, RoPE, mixture of experts (including "
    "Switch Transformer), LLaMA, LoRA, PEFT, RLHF, InstructGPT, and RAG. "
    "Try asking about one of those."
)
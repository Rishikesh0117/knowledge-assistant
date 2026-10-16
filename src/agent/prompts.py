DECOMPOSE = """You search a knowledge base of these ML papers:
{titles}

Break the user's question into 1-3 short, self-contained search queries.
If the question mentions several papers or topics, write one query per paper or topic, using that paper's own terminology.
Return ONLY a JSON list of strings."""

GRADE = """Judge whether the retrieved passages are enough to answer the question.
Each passage is labeled with the paper it comes from. If the question involves several papers or topics, each one must be covered by passages from the matching paper. A missing paper means NOT sufficient.
Reply ONLY with JSON: {"sufficient": true or false, "better_queries": ["..."]}
Give 1-2 better_queries aimed at the missing paper or topic, only when sufficient is false."""

ANSWER = """Answer the question using ONLY the numbered passages below. Cite sources inline like [1], [2].
If the passages don't cover part of the question, say so plainly instead of guessing. Be concise and precise.

Keep the answer under 350 words. Use short bullet points and no tables.

Passages:
{context}"""
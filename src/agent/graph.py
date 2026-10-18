import json
from typing import List, TypedDict

from langgraph.graph import END, START, StateGraph

from src import config
from src.agent import prompts
from src.agent.llm import chat
from src.retrieval.retriever import retrieve

MAX_RETRIES = 1
TITLES = "\n".join(
    "- " + p.stem.replace("_", " ") for p in sorted(config.PAPERS_DIR.glob("*.pdf"))
)


class State(TypedDict):
    question: str
    history: List[dict]
    standalone: str
    in_scope: bool
    queries: List[str]
    passages: List[dict]
    retries: int
    retry: bool
    answer: str


def _parse(text, opener, closer, default):
    e = text.rfind(closer)
    s = text.rfind(opener, 0, e) if e != -1 else -1
    if s == -1 or e <= s:
        return default
    try:
        return json.loads(text[s:e + 1])
    except Exception:
        return default


def _key(p):
    m = p["metadata"]
    return (m["source"], m["section"], m["chunk"])


def _format(passages, chars=None):
    lines = []
    for i, p in enumerate(passages, 1):
        m = p["metadata"]
        text = p["text"][:chars] if chars else p["text"]
        lines.append(f"[{i}] ({m['title']} - {m['section']})\n{text}")
    return "\n\n".join(lines)


def _history_text(history, max_turns=3, chars=500):
    if not history:
        return "(none)"
    lines = []
    for m in history[-2 * max_turns:]:
        role = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{role}: {m['content'][:chars]}")
    return "\n".join(lines)


def decompose(state):
    out = chat(
        [{"role": "system", "content": prompts.DECOMPOSE.format(titles=TITLES)},
         {"role": "user", "content":
             f"Conversation so far:\n{_history_text(state.get('history', []))}\n\n"
             f"Latest question: {state['question']}"}],
        model=config.FAST_MODEL, max_tokens=1500, think=False)
    data = _parse(out, "{", "}", {})
    standalone = data.get("standalone")
    if not isinstance(standalone, str) or not standalone.strip():
        standalone = state["question"]
    qs = data.get("queries")
    qs = qs if isinstance(qs, list) else []
    queries = [q for q in qs if isinstance(q, str)][:3] or [standalone]
    in_scope = data.get("in_scope", True) is not False
    return {
        "standalone": standalone,
        "in_scope": in_scope,
        "queries": queries if in_scope else [],
        "passages": [],
        "retries": 0,
        "retry": False,
    }


def route_scope(state):
    return "retrieve" if state["in_scope"] else "decline"


def decline(state):
    return {"answer": prompts.DECLINE, "passages": []}


def retrieve_node(state):
    passages = list(state["passages"])
    seen = {_key(p) for p in passages}
    for q in state["queries"]:
        for p in retrieve(q, k=4):
            if _key(p) not in seen:
                seen.add(_key(p))
                passages.append(p)
    return {"passages": passages}


def grade(state):
    out = chat(
        [{"role": "system", "content": prompts.GRADE},
         {"role": "user", "content":
             f"Question: {state['standalone']}\n\nPassages:\n{_format(state['passages'], 350)}"}],
        model=config.FAST_MODEL, max_tokens=1500, think=False)
    verdict = _parse(out, "{", "}", {"sufficient": True})
    better = [q for q in (verdict.get("better_queries") or []) if isinstance(q, str)][:2]
    if verdict.get("sufficient", True) or not better or state["retries"] >= MAX_RETRIES:
        return {"retry": False}
    return {"retry": True, "queries": better, "retries": state["retries"] + 1}


def route(state):
    return "retrieve" if state["retry"] else "answer"


def answer(state):
    top = sorted(state["passages"], key=lambda p: p["distance"])[:10]
    user = (f"Conversation so far:\n{_history_text(state.get('history', []))}\n\n"
            f"Question: {state['standalone']}")
    reply = chat(
        [{"role": "system", "content": prompts.ANSWER.format(context=_format(top))},
         {"role": "user", "content": user}],
        model=config.REASONING_MODEL, max_tokens=2500)
    return {"answer": reply, "passages": top}


def build_graph():
    g = StateGraph(State)
    g.add_node("decompose", decompose)
    g.add_node("retrieve", retrieve_node)
    g.add_node("grade", grade)
    g.add_node("answer", answer)
    g.add_node("decline", decline)
    g.add_edge(START, "decompose")
    g.add_conditional_edges("decompose", route_scope,
                            {"retrieve": "retrieve", "decline": "decline"})
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route, {"retrieve": "retrieve", "answer": "answer"})
    g.add_edge("answer", END)
    g.add_edge("decline", END)
    return g.compile()


graph = build_graph()


def ask(question, history=None):
    return graph.invoke({"question": question, "history": history or []})


if __name__ == "__main__":
    r = ask("How does LoRA differ from the adapter approach in the PEFT survey?")
    print("queries:", r["queries"], "| retries:", r["retries"])
    print(r["answer"])
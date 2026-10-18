import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.agent.graph import graph

import streamlit.components.v1 as components


def scroll(where):
    components.html(
        f"""
        <script>
        setTimeout(function () {{
            const doc = window.parent.document;
            const box = doc.querySelector('[data-testid="stMain"]')
                || doc.querySelector('section.main')
                || doc.querySelector('[data-testid="stAppViewContainer"]');
            if (!box) return;
            if ("{where}" === "bottom") {{
                box.scrollTo({{top: box.scrollHeight, behavior: "smooth"}});
            }} else {{
                const msgs = doc.querySelectorAll('[data-testid="stChatMessage"]');
                if (msgs.length) {{
                    msgs[msgs.length - 1].scrollIntoView({{behavior: "smooth", block: "start"}});
                }}
            }}
        }}, 150);
        </script>
        """,
        height=0,
    )

st.set_page_config(page_title="Technical Knowledge Assistant", page_icon="📚")

MAX_QUESTIONS = 10
EXAMPLES = [
    "How does Switch Transformer's routing differ from the sparsely-gated MoE layer?",
    "Compare LoRA with the adapter methods surveyed in the PEFT paper.",
    "How does InstructGPT build on the earlier human-preference RLHF work?",
    "How does RoPE work, and which model here uses it?",
]
STEP_LABELS = {
    "decompose": "Planning search queries",
    "retrieve": "Searching the papers",
    "grade": "Checking that every paper is covered",
    "answer": "Writing the answer",
}

if "messages" not in st.session_state:
    st.session_state.messages = []
if "n" not in st.session_state:
    st.session_state.n = 0

with st.sidebar:
    st.header("About")
    st.write(
        "An agentic RAG assistant over 11 foundational ML papers. "
        "It plans targeted searches, checks that every paper in your question "
        "is covered, retries if not, then answers with citations."
    )
    st.markdown(
        "**Models (Nebius Token Factory)**\n"
        "- Nemotron-3-Ultra: final answers\n"
        "- Nemotron-3.5-Lightning: planning and checking\n"
        "- Qwen3-Embedding-8B: retrieval"
    )
    st.markdown(
        "**Papers**\n"
        "- Attention Is All You Need\n"
        "- RoFormer (RoPE)\n"
        "- Sparsely-Gated MoE\n"
        "- Switch Transformers\n"
        "- LLaMA\n"
        "- LoRA\n"
        "- PEFT survey\n"
        "- Deep RL from Human Preferences\n"
        "- Learning to Summarize from Human Feedback\n"
        "- InstructGPT\n"
        "- RAG"
    )
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
    counter_box = st.empty()
    counter_box.caption(f"Questions this session: {st.session_state.n}/{MAX_QUESTIONS}")

st.title("Technical Knowledge Assistant")
st.caption("Ask across papers on transformers, LoRA, RLHF, mixture of experts, and more.")


def render_assistant(msg):
    st.write(msg["content"])
    if msg["sources"]:
        with st.expander("Sources"):
            for i, p in enumerate(msg["sources"], 1):
                m = p["metadata"]
                st.markdown(f"**[{i}] {m['title']}**: {m['section']}")
                st.caption(p["text"][:300] + "...")
    with st.expander("Agent trace"):
        if msg["queries"]:
            st.write("Search queries: " + " | ".join(msg["queries"]))
            st.write(f"Retrieval retries: {msg['retries']}")
        else:
            st.write("Outside the knowledge base, so no search was run.")


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "user":
            st.write(m["content"])
        else:
            render_assistant(m)

examples_box = st.empty()
pending = None
if not st.session_state.messages:
    with examples_box.container():
        st.caption("Try one of these:")
        for i, ex in enumerate(EXAMPLES):
            if st.button(ex, key=f"ex{i}", use_container_width=True):
                pending = ex

typed = st.chat_input("Ask about transformers, LoRA, RLHF, mixture of experts...")
question = typed or pending

if question:
    examples_box.empty()
    if st.session_state.n >= MAX_QUESTIONS:
        st.warning("Demo limit reached for this session.")
        st.stop()
    st.session_state.n += 1
    counter_box.caption(f"Questions this session: {st.session_state.n}/{MAX_QUESTIONS}")

    history = [{"role": m["role"], "content": m["content"]}
               for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    scroll("bottom")

    with st.chat_message("assistant"):
        state, queries, retries = {}, [], 0
        with st.status("Working...", expanded=True) as status:
            try:
                for update in graph.stream(
                    {"question": question, "history": history}, stream_mode="updates"
                ):
                    for node, out in update.items():
                        state.update(out)
                        if out.get("queries"):
                            queries.extend(out["queries"])
                        retries = out.get("retries", retries)
                        label = STEP_LABELS.get(node, node)
                        if node == "decompose":
                            if out.get("in_scope", True):
                                st.write("✓ " + label + ": " + " | ".join(out["queries"]))
                            else:
                                st.write("✓ Checked scope: outside the knowledge base")
                        elif node == "retrieve":
                            st.write(f"✓ {label}: {len(out['passages'])} passages found")
                        elif node == "grade":
                            if out.get("retry"):
                                st.write("✓ " + label + ": gaps found, searching again")
                            else:
                                st.write("✓ " + label + ": coverage is enough")
                        elif node == "answer":
                            st.write("✓ " + label)
                status.update(label="Done", state="complete", expanded=False)
            except Exception as e:
                status.update(label="Something went wrong", state="error")
                st.error(f"The request failed: {e}")
                st.stop()
        msg = {
            "role": "assistant",
            "content": state.get("answer", ""),
            "sources": state.get("passages", []),
            "queries": queries,
            "retries": retries,
        }
        render_assistant(msg)
        st.session_state.messages.append(msg)
        scroll("answer")
import streamlit as st

from src.agent.graph import ask

st.set_page_config(page_title="Technical Knowledge Assistant")
st.title("Technical Knowledge Assistant")
st.caption("RAG over 11 ML papers: Qwen3-Embedding retrieval, Nemotron reasoning on Nebius Token Factory")

q = st.chat_input("Ask about transformers, LoRA, RLHF, mixture of experts...")
if q:
    st.chat_message("user").write(q)
    with st.chat_message("assistant"):
        with st.spinner("Searching and reasoning..."):
            r = ask(q)
        st.write(r["answer"])
        with st.expander("Sources"):
            for i, p in enumerate(r["passages"], 1):
                m = p["metadata"]
                st.markdown(f"**[{i}] {m['title']}**: {m['section']}")
                st.caption(p["text"][:300] + "...")
        with st.expander("Agent trace"):
            st.write("Search queries:", r["queries"])
            st.write("Retrieval retries:", r["retries"])
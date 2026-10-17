# Technical Knowledge Assistant

An agentic RAG app that answers questions across 11 foundational ML papers, with cited sources and a visible reasoning trace. Built for the **Nebius x NVIDIA Global AI Hackathon** (Best Apps and Agents track).

- **Live demo:** [ADD LIVE DEMO URL]
- **Demo video:** [ADD YOUTUBE URL]
- **License:** MIT

## What it does

Ask a question like *"How does Switch Transformer's routing differ from the sparsely-gated MoE layer?"* and the assistant:

1. Breaks the question into targeted search queries (one per paper or topic it names).
2. Retrieves matching passages from a vector index of the papers.
3. Checks whether the passages actually cover every paper the question mentions, and searches again if they don't.
4. Writes a concise answer that cites the passages it used.

The UI shows the answer, the source passages, and the agent trace (queries used, retries), so you can see how it got there.

## Why

Reading papers one at a time is slow, and the interesting questions span several of them (LoRA vs. adapters, RLHF vs. InstructGPT, MoE vs. Switch Transformer). In testing, plain single-query RAG answered a LoRA-vs-PEFT-survey question almost entirely from the LoRA paper, because "LoRA" dominated the query. The decompose-and-check loop in this project fixes that failure mode.

## How it works

```
OFFLINE (run once)
PDFs -> papers.py (parse + chunk) -> chunks_cache.json
     -> build_index.py -> embeddings -> Chroma collection "knowledge_base"

ONLINE (every question)
Streamlit UI -> LangGraph agent
   decompose -> retrieve -> grade --(not enough, one retry)--> retrieve
                              \--(enough)--> answer -> UI (answer + sources + trace)
```

- **Ingestion:** PDFs are converted to markdown with `pymupdf4llm`, split at section headings, and packed into chunks of roughly 450 words. References and appendix sample outputs are skipped. Each chunk keeps its paper, section, and position as metadata.
- **Retrieval:** the query is embedded and matched against Chroma by cosine similarity (top 4 per query).
- **Agent (LangGraph):** a four-node state machine: `decompose`, `retrieve`, `grade`, `answer`, with a conditional edge from `grade` back to `retrieve`. The loop is capped at one retry.
- **Answering:** the best 10 passages go to the reasoning model, which is told to use only those passages, cite them as `[n]`, and say so when something isn't covered.

## NVIDIA models and Nebius Token Factory

Everything runs through the Token Factory OpenAI-compatible API (`https://api.tokenfactory.nebius.com/v1/`).

| Role | Model | Notes |
|---|---|---|
| Answer synthesis | `nvidia/Nemotron-3-Ultra-550b-a55b` | Called once per question |
| Planning and grading | `nvidia/Nemotron-3_5-Lightning` | Fast and cheap; thinking disabled for these short JSON calls |
| Embeddings | `Qwen/Qwen3-Embedding-8B` | The only embedding model on the public endpoints when I checked; not an NVIDIA model |

Where Token Factory helped:

- The OpenAI-compatible API meant the standard `openai` client worked with only a base URL change, for both chat and embeddings.
- One key and one gateway module (`src/agent/llm.py`) cover every model, so swapping a model is a one-line config change.
- Model routing is cheap: Lightning handles the frequent small calls and Ultra is used once per question. Building the full index (about 174k tokens) cost a fraction of a cent.

No other Nebius services (AI Cloud, Serverless) are used.

## Knowledge base

| Topic | Papers |
|---|---|
| Architecture | Attention Is All You Need; RoFormer (rotary position embeddings); Outrageously Large Neural Networks (sparsely-gated MoE); Switch Transformers; LLaMA |
| Efficient adaptation | LoRA; a survey of parameter-efficient fine-tuning (PEFT) |
| Alignment | Deep Reinforcement Learning from Human Preferences; Learning to Summarize from Human Feedback; Training Language Models to Follow Instructions (InstructGPT) |
| Grounding | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks |

The PDFs are in `data/papers/`.

## Quickstart

Requires Python 3.10+ and a Nebius Token Factory API key.

```bash
git clone https://github.com/Rishikesh0117/[REPO-NAME].git
cd [REPO-NAME]
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root (see `.env.example`):

```
NEBIUS_API_KEY=your_key_here
```

Model names are set in `src/config.py`. If a model name returns a 404, list the IDs your key can use:

```bash
python -c "from src.agent.llm import client; print(sorted(m.id for m in client.models.list()))"
```

Build the index (skip this if `vectorstore/` is already included in the repo). Parsing the PDFs takes about 10 minutes the first time, and embedding costs a fraction of a cent:

```bash
python -m src.ingestion.build_index
```

Check retrieval, then run the agent and the app:

```bash
python -m src.retrieval.retriever
python -m src.agent.graph
python -m streamlit run src/app.py
```

## Example questions

- How does Switch Transformer's routing differ from the sparsely-gated MoE layer?
- Compare LoRA with the adapter methods surveyed in the PEFT paper.
- How does InstructGPT build on the earlier human-preference RLHF work?
- How does RoPE work, and which model here uses it?

## Project structure

```
src/
  config.py              paths, model names, constants
  agent/
    llm.py               the only code that talks to Token Factory (chat + embed)
    graph.py             LangGraph state machine
    prompts.py           decompose / grade / answer prompts
  ingestion/
    papers.py            PDF -> section-aware chunks
    docs.py              markdown docs -> chunks (not used in this build)
    build_index.py       embed chunks into Chroma
  retrieval/retriever.py query -> top-k passages
  app.py                 Streamlit chat UI
data/papers/             the 11 PDFs
```

## Limitations

- Retrieval is vector-only: no keyword search and no reranker, so one paper can occasionally dominate the results.
- Each question is independent; there is no conversation memory.
- PDF parsing leaves some noise (figure text, author blocks).
- There is no automated evaluation yet. An evaluation harness for this system is planned as follow-up work.

## What was built during the hackathon

This is a new project, built entirely during the submission period.

## Feedback on Nebius Token Factory and NVIDIA tools

*(Based on building this project. Edit freely.)*

**What worked well**

- The OpenAI-compatible API made integration quick, and the Playground made it easy to try a model before writing code.
- Nemotron-3.5-Lightning is very fast for planning and grading calls, and the Ultra model gave grounded, well-cited answers.
- Embedding the whole knowledge base was extremely cheap.

**Friction I ran into**

- **Model naming:** the catalog displays `Nemotron-3.5-Lightning`, but the API ID is `nvidia/Nemotron-3_5-Lightning`, so my first call returned a 404. Showing the copyable API ID on each model card would save time. `client.models.list()` was how I found the right one.
- **Reasoning output in the reply:** Lightning writes its thinking process into the normal response. With a small `max_tokens`, the visible reply was cut off before any answer appeared, and my JSON parser silently fell back. A clearly documented switch to disable thinking (and a note on how it interacts with `max_tokens`) would help.
- **No NVIDIA embedding model** was available on the public endpoints, so retrieval had to use a non-NVIDIA embedding model.
- **Billing setup:** onboarding required adding a card even though the hackathon credits were enough, and I found no way to remove the card afterward (only edit it). The docs I read were inconsistent about whether a card is required to apply a promo code. A clearer note on this, plus a spending cap, would help students working on a tight budget.
- **Credits flow:** the hackathon promo code and the Builder Program arrive through separate forms and emails, and it was unclear which account each credit should be applied to.

## License

MIT. See [LICENSE](LICENSE).
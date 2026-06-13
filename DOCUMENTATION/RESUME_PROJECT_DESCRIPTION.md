# 🤖 Agentic RAG Policy Analyst Platform
**Hybrid RAG + Autonomous Multi-Turn Reasoning for Insurance Policy Intelligence**

> **Role**: Full-Stack AI/ML Engineer | **Stack**: Python, LangGraph, Gemini 2.0, FAISS, BM25, Streamlit, PostgreSQL  
> **Codebase**: ~1,650 lines across 8 production modules

---

## 📋 EXECUTIVE SUMMARY

Built an **autonomous, multi-turn agentic RAG system** that interprets complex insurance policies through intelligent retrieval and reasoning. Unlike traditional RAG (`Retrieve → Generate`), this implements a **LangGraph ReAct agent** with autonomous planning, multi-iteration search, verification, and cited synthesis. Achieved **100× throughput improvement** in embedding pipeline via batch optimization, identified and fixed **5 bugs** (including critical FAISS index corruption and broken model fallback), and catalogued **5 dead code items** for transparent maintenance.

---

## 🎯 FIELDS OF EXPERTISE

| Field | Specific Competencies | Scale |
|-------|----------------------|-------|
| **Agentic AI / LLM Orchestration** | LangGraph StateGraph, ReAct pattern, multi-turn reasoning, model fallback strategies | 4 nodes, 5 max iterations |
| **Information Retrieval** | Hybrid search fusion (FAISS dense + BM25 sparse), 70/30 weighting, SHA-256 caching | 100-entry bounded cache |
| **LLM Engineering** | Gemini API integration, prompt engineering, rate limiting (500ms), 2-level failover | 2.0 Flash → 1.5 Flash |
| **Software Architecture** | MCP-inspired decoupling, modular design, dependency injection, resource abstraction | 8 loosely-coupled modules |
| **Database Engineering** | PostgreSQL schema (5 tables, 3NF), thread-safe connection pool, JSONB storage | 1–10 connections |
| **Document Processing** | Multi-format parsing with intelligent chunking (500-word windows, 50-word overlap) | 5 formats (PDF, DOCX, XLSX, XLS, TXT) |
| **Actuarial Modeling** | Premium computation, risk multiplier, claim payout, loss frequency estimation | 4 policy types, 4 functions |
| **Semantic Analysis** | Cross-document contradiction detection, negation polarity, severity classification | 6 categories, 4 severity levels |
| **UI/UX Engineering** | Streamlit dashboard, 12+ interactive widgets, session management, real-time metrics | 4 tabbed tool panels |

---

## 📊 KEY METRICS

| Category | Key Metrics |
|----------|------------|
| **Codebase** | 1,650 lines · 8 modules · 25+ error handlers · 100% syntax compilation |
| **Agent** | 4-node LangGraph · 5 max reasoning iterations · conditional edge routing |
| **Search** | FAISS 768-dim + BM25 · 70/30 fusion · 100× batch throughput · SHA-256 cache (100 entries) |
| **Database** | 5 PostgreSQL tables · thread-safe pool (1–10) · 3 foreign key constraints |
| **Documents** | 5 formats · 500-word chunks · 50-word overlap · 4 encoding fallbacks |
| **Contradictions** | 6 categories · 4 severity levels · O(n²) pairwise · 30% overlap threshold |
| **API** | 500ms rate limit · 2-level model fallback · real-time monitoring |

---

## 🏗️ ARCHITECTURE (7 Layers)

```
Streamlit UI (545 loc) → LangGraph Agent (392 loc) → MCP Server (85 loc)
→ Hybrid Search: FAISS (70%) + BM25 (30%) (210 loc) → Document Processor (186 loc)
→ PostgreSQL (160 loc) ← Performance Monitor (82 loc)
```

---

## 🔧 KEY BUG FIXES & DEAD CODE

| Type | Item | Impact |
|------|------|--------|
| 🐛 **Bug** | Fallback model was identical to primary | Now uses Gemini 1.5 Flash for meaningful failover |
| 🐛 **Bug** | system_prompt dropped on fallback | Preserved in fallback API call |
| 🐛 **Bug** | FAISS index corrupted with duplicate vectors | Incrementally adds only new embeddings |
| 🐛 **Bug** | False positive contradiction detection (substring matching) | Word-boundary regex (`\bnot\b`) |
| 🐛 **Bug** | Error count double-incremented | Single source of truth in `record_error()` |
| 🗑️ **Dead Code** | RiskLevel enum (never instantiated), get_recent_queries() (never called), analytics/contradictions tables (never written) | All annotated with recommendations |

---

## 🛠️ TECH STACK

| Layer | Technology |
|-------|------------|
| Runtime | Python ≥3.11 |
| LLM | Google Gemini 2.0 Flash / 1.5 Flash |
| Agent | LangGraph ≥1.0.3 + LangChain Core ≥1.0.7 |
| Dense Search | FAISS CPU (768-dim embeddings) |
| Sparse Search | rank-bm25 |
| UI | Streamlit ≥1.51.0 |
| Database | PostgreSQL + psycopg2 (connection pool) |
| Documents | PyPDF2, python-docx, pandas |

---

> **Differentiator**: Traditional RAG = `Retrieve → Generate` (linear). This system = `Plan → Search → Verify → Synthesize` (iterative with autonomous reasoning, 2-level fallback, and 25+ error handlers).
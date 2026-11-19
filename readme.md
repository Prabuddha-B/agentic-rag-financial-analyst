# Agentic RAG Financial Document Analysis System (LangGraph & Qwen2.5)

[](https://colab.research.google.com/github/Prabuddha-B/agentic-rag-financial-analyst/blob/main/Notebook/apple_financial_agent.ipynb)

An end-to-end **Agentic Retrieval-Augmented Generation (RAG)** system that autonomously analyzes Apple Inc.'s multi-year SEC Form 10-K filings (FY2023–FY2025), executes deterministic Python financial calculations, generates visualisations, and compiles a structured, citation-backed PDF report.

## Architecture

```text
User Question
      │
      ▼
┌─────────────┐  RETRIEVE   ┌───────────────┐
│   Planner   │────────────►│ Retrieve Node │ (FAISS IndexFlatIP + all-MiniLM-L6-v2)
│ (Qwen 2.5   │◄────────────└───────────────┘
│ 1.5B-Inst.) │  CALCULATE  ┌────────────────┐
│             │────────────►│ Calculate Node │ (Deterministic Python Extractors & Math)
│             │◄────────────└────────────────┘
│             │   REPORT    ┌───────────────┐
│             │────────────►│  Report Node  │ (Grounded Qwen Narrative + Citations)
│             │◄────────────└───────────────┘
└──────┬──────┘
       │ FINISH
       ▼
  Charts + Verified PDF Report (fpdf2)
```

## Key Engineering Highlights

- **Stateful LangGraph Agent**: Implements `AgentState`, an LLM-driven `planner_node` (`Qwen/Qwen2.5-1.5B-Instruct`), conditional routing (`RETRIEVE -> CALCULATE -> REPORT -> FINISH`), and a duplicate-step guard.
- **Semantic Retrieval Pipeline**: Parses 281 pages of Apple FY2023–FY2025 Form 10-K filings via PyMuPDF into 958 overlapping chunks (220 words / 40-word overlap) indexed in `faiss.IndexFlatIP` with cosine similarity and year-filtered search.
- **Hallucination-Free Tool Execution**: Offloads all numerical extraction and financial formulas (YoY revenue growth, product category shares, Products vs. Services comparison) to deterministic Python tools, keeping LLM narratives grounded in verified figures and exact PDF page citations.
- **Automated Reporting**: Generates Matplotlib trend and composition charts (`Outputs/`) and compiles a multi-section PDF report via `fpdf2` with 12 automated verification assertions.

## Verified Analysis Results (Apple Inc. FY2023–FY2025)

| Analysis Question | Verified Result | Source |
| :--- | :--- | :--- |
| **Q1:** Total revenue change (FY2023–FY2025) | `$383,285M` → `$416,161M` (`+$32,876M`, `+8.58%`) | FY2025 Form 10-K, p.25 |
| **Q2:** Top product category (FY2025) | **iPhone** (`$209,586M`, `50.36%` of total net sales) | FY2025 Form 10-K, p.26 |
| **Q3:** Products vs. Services growth (FY2024–FY2025) | **Services** (`+13.51%`) outpaced **Products** (`+4.12%`) | FY2025 Form 10-K, p.32 |

### Generated Visualisations

| Revenue Trend (FY2023–FY2025) | FY2025 Category Revenue Share |
| :---: | :---: |
|  |  |

Full generated report: [Outputs/apple\_financial\_analysis\_report.pdf](https://www.google.com/search?q=Outputs/apple_financial_analysis_report.pdf)

## Repository Structure

```text
├── Datasets/   # Apple Inc. FY2023–FY2025 SEC Form 10-K PDF filings
├── Notebook/   # End-to-end Google Colab LangGraph notebook
└── Outputs/    # Generated PDF financial report and Matplotlib charts
```

## Tech Stack

`Python` · `LangGraph` · `Qwen2.5-1.5B-Instruct` · `Hugging Face Transformers` · `PyTorch` · `SentenceTransformers` · `FAISS` · `PyMuPDF` · `Matplotlib` · `fpdf2` · `Google Colab (T4 GPU)`

## How to Run

1. Click the **Open in Colab** badge above and switch Runtime to **T4 GPU**.
2. Upload the three Form 10-K PDFs from `Datasets/` into `/content/`.
3. Run all cells sequentially to build the FAISS index, run the LangGraph agent on Q1–Q3, and generate both charts and the verified PDF report.

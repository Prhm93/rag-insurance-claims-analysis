# RAG Pipeline for Motor Insurance Claim Analysis

A Retrieval-Augmented Generation (RAG) system that reads motor insurance documents (court packs, rate comparisons, fraud assessments) and answers natural language questions about them using Google Gemini AI and FAISS vector search.

## What This Does

The system takes unstructured insurance PDF documents and makes them queryable through natural language. For example:

- *"What was the total claim value for Mrs Thompson's case?"* → Retrieves and synthesises information from the court pack
- *"What fraud indicators were found?"* → Identifies relevant sections from the fraud assessment report
- *"How does the claimed hire rate compare to market rates?"* → Cross-references rate comparison data

## Architecture

```
PDF Documents → Load & Split → Embed (Google) → Store (FAISS) → Retrieve → Generate (Gemini)
```

1. **Load**: PyPDFLoader extracts text from each PDF, preserving metadata (filename, page number)
2. **Chunk**: RecursiveCharacterTextSplitter breaks text into ~500-character overlapping segments
3. **Embed**: Google's `gemini-embedding-001` model converts each chunk into a high-dimensional vector
4. **Store**: FAISS indexes all vectors for fast similarity search
5. **Retrieve**: User questions are embedded and matched against stored vectors (top-k retrieval)
6. **Generate**: Gemini 2.5 Flash reads retrieved chunks and generates a cited, factual answer

## Tech Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| Orchestration | LangChain | Connects PDF loader, splitter, and vector store |
| Vector Search | FAISS (Facebook AI Similarity Search) | Fast similarity search by meaning |
| LLM | Google Gemini 2.5 Flash | Answer generation from retrieved context |
| Embeddings | Google gemini-embedding-001 | Text-to-vector conversion |
| PDF Parsing | PyPDF | Text extraction from documents |

## Sample Documents

The repository includes 5 realistic motor insurance documents:

| Document | Description |
|----------|-------------|
| `claim_001_court_pack.pdf` | Full court pack with claim summary, hire details, repair costs |
| `claim_002_rate_comparison.pdf` | CHO rate vs BHR/GBR benchmark analysis |
| `claim_003_fraud_assessment.pdf` | Fraud indicator scoring and risk assessment |
| `claim_004_policy_terms.pdf` | Insurance policy terms with rate tables |
| `claim_005_witness_statement.pdf` | Witness statement for court proceedings |

## Getting Started

### Prerequisites

- Google account (for free Gemini API key)
- Google Colab (recommended) or local Python 3.10+

### Quick Start (Google Colab)

1. Open `RAG_Insurance_Claims_Tutorial.ipynb` in Google Colab
2. Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey)
3. Upload the sample PDFs from the `sample_docs/` folder
4. Run each cell sequentially — the notebook is fully annotated with explanations

### Local Setup

```bash
pip install langchain langchain-google-genai langchain-community langchain-text-splitters faiss-cpu pypdf google-generativeai
export GOOGLE_API_KEY="your-key-here"
```

## Key Design Decisions

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| `chunk_size` | 500 chars | Balances context richness with retrieval precision |
| `chunk_overlap` | 50 chars | Prevents splitting mid-sentence |
| `k` (retrieval) | 4 | Returns top 4 most relevant chunks per query |
| `temperature` | 0.2 | Low creativity for factual accuracy in legal/financial context |

## Example Output

```
Q: What was the total claim value for Mrs Sarah Thompson's case?

A: The total claim value for Mrs. Sarah Thompson's case was £8,048.00.
This information is from the "COURT PACK - CLAIM SUMMARY REPORT," section 6.

Sources: claim_001_court_pack.pdf
```

## Relevance to Insurance Domain

This project demonstrates AI capabilities directly applicable to automating motor insurance claim analysis:

- **Document understanding**: Extracting structured data (rates, dates, amounts) from unstructured legal documents
- **Cross-document reasoning**: Comparing claimed rates against benchmark databases
- **Fraud detection support**: Surfacing relevant risk indicators from complex case files
- **Natural language interface**: Enabling non-technical staff to query large document sets

## Author

**Parham Imanzadeh**  
MSc Computer Science (by Research)
[GitHub](https://github.com/Prhm93) | pimanzadeh.ch@gmail.com

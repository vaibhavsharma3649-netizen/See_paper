Research Paper Recommendation & AI Research Assistant

An AI-powered research assistant that helps users discover research papers and understand them more easily.

The project combines semantic paper recommendation, arXiv paper retrieval, RAG, and an LLM-based Feynman-style explanation pipeline.

What the project does

User Query
   ↓
Streamlit Frontend
   ↓
FastAPI
   ↓
ChromaDB + Sentence Transformers
   ↓
Recommended Papers
   ↓
User selects a paper
   ↓
arXiv ID
   ↓
Fetch paper PDF
   ↓
PyMuPDF
   ↓
Recursive Chunking
   ↓
Paper-specific ChromaDB
   ↓
Retriever
   ↓
Relevant Context
   ↓
Feynman-style Prompt
   ↓
LLM
   ↓
Easy-to-understand Paper Explanation

Key Features

Semantic research-paper search using Sentence Transformers and ChromaDB.

arXiv integration using the paper's arxiv_id to retrieve the selected paper automatically.

PDF text extraction with PyMuPDF.

Recursive text chunking for RAG using RecursiveCharacterTextSplitter.

Paper-specific RAG cache so a selected paper can be indexed once and reused.

Feynman-style explanations that focus on intuition, methodology, architecture, experiments, results, and limitations.

FastAPI backend for recommendation and paper-analysis APIs.

Streamlit frontend for the user interface.

Tech Stack

Component

Technology

Frontend

Streamlit

Backend

FastAPI

Recommendation embeddings

sentence-transformers/all-MiniLM-L6-v2

Vector database

ChromaDB

RAG embeddings

sentence-transformers/all-MiniLM-L6-v2

PDF reader

PyMuPDF / PyMuPDFLoader

Text splitting

RecursiveCharacterTextSplitter

LLM orchestration

LangChain

LLM

Groq (ChatGroq)

Paper source

arXiv

Configuration

python-dotenv

Project Structure

Research_papers/
│
├── api/
│   ├── main.py          # FastAPI endpoints
│   ├── arxiv.py         # arXiv/PDF retrieval utilities
│   ├── rag.py           # RAG pipeline
│   └── prompt.py        # Prompt template
│
├── frontend/
│   └── app.py           # Streamlit application
│
├── papers/              # Downloaded PDFs (local, ignored by Git)
├── vector_cache/        # Per-paper RAG vector stores (local, ignored by Git)
├── recommendation_db/   # Recommendation Chroma database (local, ignored by Git)
├── data/                # Dataset files (local, ignored by Git)
├── .env                 # API keys (local, never commit)
├── .gitignore
├── requirements.txt
└── README.md

How the recommendation system works

The recommendation system uses a large research-paper dataset containing metadata such as title, abstract/summary, category, and arXiv ID.

The paper text used for recommendation is embedded with:

sentence-transformers/all-MiniLM-L6-v2

The embeddings are stored in a persistent ChromaDB collection.

A user query is converted into an embedding and searched against the stored paper vectors.

Query
  ↓
Sentence Transformer
  ↓
Query embedding
  ↓
ChromaDB
  ↓
Top similar papers

The Chroma metadata also stores arxiv_id, which is used to retrieve the selected paper later.

How the RAG system works

After a paper is selected:

Read its arxiv_id.

Fetch the paper from arXiv.

Load the PDF with PyMuPDF.

Split the extracted text into overlapping chunks.

Create embeddings for the chunks.

Store the chunks in a paper-specific persistent Chroma collection.

Retrieve relevant chunks for the required explanation.

Combine the retrieved context with the Feynman-style prompt.

Send the prompt to the LLM.

Return the explanation to Streamlit.

Example chunking configuration:

RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

Feynman-style paper explanation

The prompt is designed to teach the paper rather than simply summarize it.

The explanation covers areas such as:

Paper in one sentence

Problem and motivation

Core idea

Step-by-step methodology

Architecture/model

Important concepts

Essential mathematics

Training/algorithm

Experiments

Results

Why the method works

Limitations

Comparison with previous work

Visual mental model

Beginner-friendly analogy

Final takeaway

The prompt also instructs the model not to invent experimental results or unsupported information.

API Endpoints

POST /recommend

Search for relevant research papers.

Request:

{
  "query": "transformer based image classification"
}

The response contains recommended papers with their title, category, arXiv ID, and arXiv URL.

POST /visualize

Analyze a selected research paper.

Request:

{
  "arxiv_id": "1706.03762"
}

The backend retrieves the paper, runs the RAG pipeline, and sends the retrieved context to the LLM.

Local Setup

1. Clone or copy the project

git clone <your-repository-url>
cd Research_papers

2. Create a virtual environment

python -m venv .venv

Windows:

.venv\Scripts\activate

3. Install dependencies

pip install -r requirements.txt

4. Configure environment variables

Create a .env file in the project root:

GROQ_API_KEY=your_groq_api_key

Add any other provider keys required by your local implementation.

Do not commit .env to GitHub.

5. Start the FastAPI backend

From the project root:

uvicorn api.main:app --reload

API documentation will be available at:

http://127.0.0.1:8000/docs

6. Start Streamlit

In another terminal:

streamlit run frontend/app.py

Vector Database and Caching

The project uses two different vector-storage purposes:

Recommendation database

Contains embeddings for the research-paper dataset and is used to find similar papers.

recommendation_db/

Paper-specific RAG cache

Contains chunks for an individual selected paper.

vector_cache/<arxiv_id>/

The RAG cache should be reused after the first indexing operation rather than adding the same chunks repeatedly.

GitHub Notes

Large or sensitive local files should not be committed.

Recommended ignored files/directories include:

.env
papers/
vector_cache/
recommendation_db/
data/
__pycache__/
.venv/

This keeps API keys, downloaded papers, datasets, and local vector databases out of the repository.

Current Development Direction

The current system focuses on:

Research Paper Discovery
        +
Paper-specific RAG
        +
Feynman-style Explanation

Planned improvements include:

Better hybrid retrieval combining title/keyword matching with semantic search.

Retrieval of section-specific context for problem, architecture, experiments, and results.

Extraction and display of important figures from research papers.

Better structured output for visual paper diagrams.

More efficient caching and cleanup of temporary paper files.

Goal

The goal is to turn a difficult research paper into something that a student can read, visualize, question, and understand step-by-step instead of relying on a dense PDF alone.

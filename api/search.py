from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import arxiv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import load_prompt
from arxv import fetch_arxiv_pdf
import os
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()
app = FastAPI()

# ── Recommendation DB (lazy-loaded once) ─────────────────────────────────────
_rec_collection = None
_rec_encoder = None

def get_rec_collection():
    global _rec_collection, _rec_encoder
    if _rec_collection is not None:
        return _rec_collection, _rec_encoder

    db_path = os.path.join(os.path.dirname(__file__), "recommendation_db")
    client = chromadb.PersistentClient(path=db_path)
    _rec_collection = client.get_collection("research_papers_v3")
    _rec_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _rec_collection, _rec_encoder

# ─────────────────────────────────────────────────────────────────────────────

class PaperRequest(BaseModel):
    arxiv_id: str

@app.post('/visualize')
def visualize_paper(request: PaperRequest):
    arxiv_id = request.arxiv_id

    # 1. Fetch metadata using arxiv library
    title = "Unknown Title"
    try:
        client = arxiv.Client()
        search = arxiv.Search(id_list=[arxiv_id])
        paper_metadata = next(client.results(search))
        title = paper_metadata.title
    except Exception as e:
        print(f"Failed to fetch metadata: {e}")

    # 2. Fetch PDF
    try:
        pdf_path = fetch_arxiv_pdf(arxiv_id, save_dir="./papers")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {str(e)}")

    # 3. Processing / RAG pipeline
    try:
        pdfreader = PyMuPDFLoader(pdf_path)
        docs = pdfreader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )
        chunked = splitter.split_documents(docs)

        embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        cache_dir = f'./vector_cache/{arxiv_id}'
        os.makedirs(cache_dir, exist_ok=True)

        vector_store = Chroma(
            embedding_function=embedding,
            persist_directory=cache_dir,
            collection_name=f"paper_{arxiv_id.replace('.', '_')}"
        )
        vector_store.add_documents(chunked)

        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        queries = [
        "What is the core problem and motivation of the paper?",
        "What is the proposed methodology and how does it work?",
        "What is the model architecture and data flow?",
        "What experiments, datasets, baselines and metrics were used?",
        "What are the main results and limitations?"
]
        retrieved_docs = []

        for query in queries:
            docs = retriever.invoke(query)
            retrieved_docs.extend(docs)

    # Remove duplicate chunks
        unique_docs = {}
        for doc in retrieved_docs:
            key = (
            doc.metadata.get("source", ""),
            doc.metadata.get("page", ""),
            doc.page_content
        )
        unique_docs[key] = doc

        context_text = "\n\n".join(
        doc.page_content for doc in unique_docs.values()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

    # 4. Generate Visualization
    try:
        llm = ChatGroq(model='openai/gpt-oss-120b')
        template = load_prompt('prompt.json')
        prompt = template.invoke({
            'title': title,
            'context': context_text
        })

        result = llm.invoke(prompt)

        return {"title": title, "explanation": result.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Error: {str(e)}")


class RecommendRequest(BaseModel):
    query: str

@app.post('/recommend')
def recommend_papers(request: RecommendRequest):
    """
    Given a free-text search query, encode it with the same
    SentenceTransformer used during training, and query the
    Kaggle-built ChromaDB collection to return similar papers.
    The arxiv_id comes from the ChromaDB metadata automatically.
    """
    query_text = request.query.strip().lower()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # 1. Load the recommendation collection
    try:
        collection, encoder = get_rec_collection()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load recommendation DB: {str(e)}")

    # 2. Encode the query and search ChromaDB
    try:
        query_vector = encoder.encode([query_text])
        results = collection.query(
            query_embeddings=query_vector.tolist(),
            n_results=5
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation query failed: {str(e)}")

    # 3. Format results — arxiv_id comes from metadata
    recommendations = []
    if results and results.get("metadatas") and results["metadatas"][0]:
        for meta in results["metadatas"][0]:
            if not meta:
                continue
            rec_id = meta.get("arvix_id", "")   # typo from notebook is intentional
            recommendations.append({
                "title":         meta.get("title", ""),
                "arxiv_id":      rec_id,
                "category_code": meta.get("category_code", ""),
                "url":           f"https://arxiv.org/abs/{rec_id}"
            })

    return {"recommendations": recommendations}


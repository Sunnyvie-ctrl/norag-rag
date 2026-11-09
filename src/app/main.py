import os
import shutil
import re
from collections import defaultdict
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .models import RAGQueryParams, RAGResponse, DocumentInfo
from .rag_engine import RAGEngine

app = FastAPI(title="EPITA BCS S5 - V2 RAG Application", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_engine = RAGEngine()

class WordStats(BaseModel):
    word: str
    frequency: int
    chunk_count: int
    chunks: List[str]

class QuestionWordAnalysis(BaseModel):
    word: str
    exists_in_corpus: bool
    frequency: int
    chunks: List[str]

def tokenize(text: str) -> List[str]:
    """Extract lowercase alphanumeric words, ignoring common stop words."""
    stop_words = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "in", "on", "at", "to", "for", "of", "with", "it", "this", "that"}
    words = re.findall(r'\b[a-z0-9]+\b', text.lower())
    return [w for w in words if w not in stop_words]

@app.get("/api/analytics/corpus", response_model=List[WordStats])
async def get_corpus_analytics():
    collection = rag_engine.ensure_index(chunk_size=800, overlap_percent=10)
    data = collection.get()
    
    docs = data.get("documents") or []
    ids = data.get("ids") or []
    
    word_freq = defaultdict(int)
    word_chunks = defaultdict(set)
    
    for doc_text, chunk_id in zip(docs, ids):
        words = tokenize(str(doc_text))
        for w in words:
            word_freq[w] += 1
            word_chunks[w].add(str(chunk_id))
            
    stats = [
        WordStats(word=w, frequency=freq, chunk_count=len(word_chunks[w]), chunks=list(word_chunks[w]))
        for w, freq in word_freq.items()
    ]
    stats.sort(key=lambda x: x.frequency, reverse=True)
    return stats

@app.get("/api/analytics/question", response_model=List[QuestionWordAnalysis])
async def analyze_question_words(q: str):
    collection = rag_engine.ensure_index(chunk_size=800, overlap_percent=10)
    data = collection.get()
    docs = data.get("documents") or []
    ids = data.get("ids") or []
    
    word_chunks = defaultdict(set)
    for doc_text, chunk_id in zip(docs, ids):
        for w in tokenize(str(doc_text)):
            word_chunks[w].add(str(chunk_id))
            
    q_words = tokenize(q)
    seen = set()
    unique_q_words = [x for x in q_words if not (x in seen or seen.add(x))]
    
    analysis = []
    for w in unique_q_words:
        chunks = list(word_chunks.get(w, set()))
        analysis.append(QuestionWordAnalysis(
            word=w, exists_in_corpus=len(chunks) > 0, frequency=len(chunks), chunks=chunks
        ))
    return analysis

@app.post("/api/query", response_model=RAGResponse)
async def query_rag(params: RAGQueryParams):
    try:
        return rag_engine.query(params)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents", response_model=List[DocumentInfo])
async def list_documents():
    files = [f for f in os.listdir(rag_engine.data_dir) if f.endswith((".txt", ".md"))]
    result = []
    for f in files:
        path = os.path.join(rag_engine.data_dir, f)
        with open(path, "r", encoding="utf-8") as file:
            content = file.read()
        result.append(DocumentInfo(filename=f, char_count=len(content), chunk_count=len(content) // 720))
    return result

@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename or ""
    if not filename.endswith((".txt", ".md")):
        raise HTTPException(status_code=400, detail="Only .txt and .md files are supported.")
    
    file_path = os.path.join(rag_engine.data_dir, filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"message": f"File '{filename}' uploaded successfully."}

@app.delete("/api/documents/{filename}")
async def delete_document(filename: str):
    file_path = os.path.join(rag_engine.data_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")
    os.remove(file_path)
    return {"message": f"File '{filename}' deleted successfully."}

@app.get("/", include_in_schema=False)
async def serve_ui():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(current_dir, "..", "static", "index.html")
    if not os.path.exists(html_path):
        return {"error": f"Could not find index.html at {html_path}."}
    return FileResponse(html_path)
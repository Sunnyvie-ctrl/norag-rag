import os
import shutil
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse 

from .models import RAGQueryParams, RAGResponse, DocumentInfo
from .rag_engine import RAGEngine

app = FastAPI(title="EPITA BCS S5 - V1 RAG Application", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_engine = RAGEngine()

@app.get("/", include_in_schema=False)
async def serve_ui():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(current_dir, "..", "static", "index.html")
    
    if not os.path.exists(html_path):
        return {"error": f"Could not find index.html at {html_path}. Make sure it is in src/static/index.html"}
        
    return FileResponse(html_path)

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
        result.append(DocumentInfo(
            filename=f,
            char_count=len(content),
            chunk_count=len(content) // 720 
        ))
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
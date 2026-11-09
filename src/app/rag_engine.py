import os
import time
from typing import List, Dict, Any
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from openai import OpenAI
from dotenv import load_dotenv

from .models import RAGQueryParams, RAGResponse, GroundedLLMResponse, CitedChunk
from .text_splitter import chunk_text

load_dotenv()

class RAGEngine:
    def __init__(self, data_dir: str = "data/docs", chroma_dir: str = "data/chroma_db"):
        self.data_dir = os.path.abspath(data_dir)
        self.chroma_dir = os.path.abspath(chroma_dir)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.chroma_dir, exist_ok=True)

        self.embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_dir)
        
        api_key = os.getenv("GOOGLE_API_KEY") or ""
        self.openai_client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        self.model_name = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")

    def _get_collection_name(self, chunk_size: int, overlap_percent: int) -> str:
        return f"rag_cs{chunk_size}_ov{overlap_percent}"

    def ensure_index(self, chunk_size: int, overlap_percent: int) -> chromadb.Collection:
        col_name = self._get_collection_name(chunk_size, overlap_percent)
        collection = self.chroma_client.get_or_create_collection(name=col_name, embedding_function=self.embedding_fn) # type: ignore
        if collection.count() == 0:
            collection = self.reindex_all(chunk_size, overlap_percent)
        return collection

    def reindex_all(self, chunk_size: int, overlap_percent: int) -> chromadb.Collection:
        col_name = self._get_collection_name(chunk_size, overlap_percent)
        try:
            self.chroma_client.delete_collection(col_name)
        except Exception:
            pass

        collection = self.chroma_client.create_collection(name=col_name, embedding_function=self.embedding_fn) # type: ignore
        all_files = [f for f in os.listdir(self.data_dir) if f.endswith((".txt", ".md"))]
        
        docs_to_add: List[str] = []
        ids_to_add: List[str] = []
        metadatas_to_add: List[Dict[str, Any]] = []

        for filename in all_files:
            file_path = os.path.join(self.data_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            chunks = chunk_text(content, filename, chunk_size, overlap_percent)
            for c in chunks:
                docs_to_add.append(c["text"])
                ids_to_add.append(c["chunk_id"])
                metadatas_to_add.append(c["metadata"])

        if ids_to_add:
            batch_size = 2000
            for i in range(0, len(ids_to_add), batch_size):
                collection.add(
                    documents=docs_to_add[i:i+batch_size],
                    ids=ids_to_add[i:i+batch_size],
                    metadatas=metadatas_to_add[i:i+batch_size] # type: ignore
                )
        return collection

    def query(self, params: RAGQueryParams) -> RAGResponse:
        start_time = time.time()
        collection = self.ensure_index(params.chunk_size, params.overlap_percent)

        where_clause: Any = None
        if params.selected_docs and len(params.selected_docs) > 0:
            where_clause = {"document_name": params.selected_docs[0]} if len(params.selected_docs) == 1 else {"$or": [{"document_name": d} for d in params.selected_docs]}

        results = collection.query(query_texts=[params.question], n_results=params.top_k, where=where_clause) # type: ignore

        raw_docs, raw_ids, raw_metas, raw_dists = results.get("documents"), results.get("ids"), results.get("metadatas"), results.get("distances")
        retrieved_documents = raw_docs[0] if raw_docs and len(raw_docs) > 0 else []
        retrieved_ids = raw_ids[0] if raw_ids and len(raw_ids) > 0 else []
        retrieved_metas = raw_metas[0] if raw_metas and len(raw_metas) > 0 else []
        retrieved_distances = raw_dists[0] if raw_dists and len(raw_dists) > 0 else []

        if not retrieved_documents:
            return RAGResponse(answer="No relevant content found.", 
                               is_answerable=False, 
                               cited_chunks=[], 
                               reasoning="Zero chunks retrieved.",
                                execution_time_ms=(time.time() - start_time) * 1000)

        context_blocks: List[str] = []
        retrieved_chunk_map: Dict[str, CitedChunk] = {}

        for doc_text, cid, meta, dist in zip(retrieved_documents, retrieved_ids, retrieved_metas, retrieved_distances):
            doc_name = str(meta.get("document_name", "unknown")) if isinstance(meta, dict) else "unknown"
            dist_val = float(dist) if dist is not None else 0.0
            sim_score = round(1.0 - (dist_val / 2.0), 4)
            context_blocks.append(f"[chunk_id: {cid} | doc: {doc_name}]\n{doc_text}")
            retrieved_chunk_map[str(cid)] = CitedChunk(chunk_id=str(cid), 
                                                       document_name=doc_name, 
                                                       excerpt=str(doc_text), 
                                                       similarity_score=sim_score)

        formatted_context = "\n\n---\n\n".join(context_blocks)
        system_prompt = (
            "You are a strict grounded AI assistant. Answer using ONLY the provided EXCERPTS.\n"
            "If information is absent, set is_answerable to false and explain why.\n"
            "Include chunk_ids of the excerpts referenced.\n"
            "Respond in raw JSON matching this structure: {\"answer\": \"...\", \"is_answerable\": true, \"cited_chunk_ids\": [\"chunk_1\"], \"reasoning\": \"...\"}"
        )

        response = self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"EXCERPTS:\n{formatted_context}\n\nQUESTION:\n{params.question}"}],
            response_format={"type": "json_object"},
            temperature=0.1
        )

        raw_json = response.choices[0].message.content or "{}"
        raw_json = raw_json.strip()
        if raw_json.startswith("```"):
            raw_json = raw_json.strip("`").removeprefix("json").strip()

        try:
            parsed_llm = GroundedLLMResponse.model_validate_json(raw_json)
        except Exception as e:
            raise ValueError(f"JSON Parsing failed: {str(e)}. Raw Output: {raw_json}")

        final_citations = [retrieved_chunk_map[cid] for cid in parsed_llm.cited_chunk_ids if cid in retrieved_chunk_map]
        if parsed_llm.is_answerable and not final_citations and retrieved_ids:
            first_id = str(retrieved_ids[0])
            if first_id in retrieved_chunk_map:
                final_citations = [retrieved_chunk_map[first_id]]

        return RAGResponse(answer=parsed_llm.answer, 
                           is_answerable=parsed_llm.is_answerable, 
                           cited_chunks=final_citations, 
                           reasoning=parsed_llm.reasoning, 
                           execution_time_ms=(time.time() - start_time) * 1000)
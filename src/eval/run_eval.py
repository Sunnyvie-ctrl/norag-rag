import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.app.rag_engine import RAGEngine
from src.app.models import RAGQueryParams

def run_evaluation():
    eval_dataset_path = os.path.join(SCRIPT_DIR, "eval_dataset.json")
    
    
    data_dir = os.path.join(PROJECT_ROOT, "data", "docs")
    chroma_dir = os.path.join(PROJECT_ROOT, "data", "chroma_db")
    
    rag_engine = RAGEngine(data_dir=data_dir, chroma_dir=chroma_dir)
    
    with open(eval_dataset_path, "r", encoding="utf-8") as f:
        eval_cases = json.load(f)

    print(f"\n--- Running RAG Baseline Retrieval Evaluation ({len(eval_cases)} Queries) ---\n")
    
    total_precision = 0.0
    total_recall = 0.0
    unanswerable_handled = 0
    total_unanswerable = 0

    for idx, case in enumerate(eval_cases, start=1):
        params = RAGQueryParams(
            question=case["question"],
            chunk_size=case.get("chunk_size", 800),
            overlap_percent=case.get("overlap_percent", 10),
            top_k=case.get("top_k", 5)
        )
        
        response = rag_engine.query(params)
        retrieved_ids = {c.chunk_id for c in response.cited_chunks}
        expected_ids = set(case.get("expected_chunk_ids", []))

        if not expected_ids: 
            total_unanswerable += 1
            if not response.is_answerable:
                unanswerable_handled += 1
            print(f"[{idx}/{len(eval_cases)}] Out-of-Domain Query: '{case['question'][:40]}...' -> Handled Correctly: {not response.is_answerable}")
            continue

        intersection = retrieved_ids.intersection(expected_ids)
        precision = len(intersection) / params.top_k if params.top_k > 0 else 0
        recall = len(intersection) / len(expected_ids) if expected_ids else 0

        total_precision += precision
        total_recall += recall

        print(f"[{idx}/{len(eval_cases)}] Query: '{case['question'][:35]}...' | P@{params.top_k}: {precision:.2f} | R@{params.top_k}: {recall:.2f}")

    evaluable_count = len(eval_cases) - total_unanswerable
    avg_p = (total_precision / evaluable_count) if evaluable_count > 0 else 0
    avg_r = (total_recall / evaluable_count) if evaluable_count > 0 else 0
    unanswerable_acc = (unanswerable_handled / total_unanswerable * 100) if total_unanswerable > 0 else 0

    print("\n================ EVALUATION SUMMARY ================")
    print(f"Average Precision@K:         {avg_p:.4f}")
    print(f"Average Recall@K:            {avg_r:.4f}")
    print(f"Unanswerable Detection Rate: {unanswerable_acc:.1f}% ({unanswerable_handled}/{total_unanswerable})")
    print("====================================================\n")

if __name__ == "__main__":
    run_evaluation()
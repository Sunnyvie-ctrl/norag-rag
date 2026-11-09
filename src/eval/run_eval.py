import json
import os
import sys
import argparse
from collections import defaultdict
from typing import List, Dict, Any

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.app.rag_engine import RAGEngine
from src.app.models import RAGQueryParams

def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def run_evaluation(dataset_path: str, generate: bool, no_context: bool):
    rag = RAGEngine(
        data_dir=os.path.join(PROJECT_ROOT, "docs"),
        chroma_dir=os.path.join(PROJECT_ROOT, "chroma_db")
    )
    
    cases = load_jsonl(dataset_path)
    print(f"\n--- Running V2 Evaluation on {len(cases)} queries ---")
    print(f"Dataset: {os.path.basename(dataset_path)}")
    print(f"Mode: {'NO-CONTEXT (LLM Knowledge Baseline)' if no_context else ('GENERATE ANSWERS' if generate else 'FAST RETRIEVAL ONLY')}\n")

    metrics = defaultdict(lambda: {k: {"hit": 0, "recall": 0.0, "count": 0} for k in [1, 3, 5, 10]})
    results_to_save = []

    for idx, case in enumerate(cases, start=1):
        q_text = case["question"]
        q_type = case.get("type", "general")
        expected_ids = set(case.get("expected_chunk_ids", []))
        
        case_result = {
            "question": q_text,
            "type": q_type,
            "expected_chunk_ids": list(expected_ids)
        }

        if no_context:
            prompt = f"Answer the following question. If you don't know, say 'I don't know'.\n\nQuestion: {q_text}"
            try:
                response = rag.openai_client.chat.completions.create(
                    model=rag.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                answer = response.choices[0].message.content
            except Exception as e:
                answer = f"Error: {str(e)}"
            
            print(f"[{idx}/{len(cases)}] (No-Context) Q: {q_text[:40]}...\nAnswer: {answer}\n")
            case_result["no_context_answer"] = answer
            results_to_save.append(case_result)
            continue

        collection = rag.ensure_index(chunk_size=800, overlap_percent=10)
        retrieval = collection.query(query_texts=[q_text], n_results=10)
        
        raw_ids = retrieval.get("ids", [[]])[0]
        retrieved_ids = [str(cid) for cid in raw_ids]
        case_result["retrieved_chunk_ids"] = retrieved_ids

        if generate:
            params = RAGQueryParams(question=q_text, chunk_size=800, overlap_percent=10, top_k=5)
            response = rag.query(params)
            case_result["generated_answer"] = response.answer
            case_result["is_answerable"] = response.is_answerable

        if expected_ids:
            for k in [1, 3, 5, 10]:
                top_k_retrieved = set(retrieved_ids[:k])
                intersection = top_k_retrieved.intersection(expected_ids)
                
                hit = 1 if len(intersection) > 0 else 0
                recall = len(intersection) / len(expected_ids)
                
                metrics[q_type][k]["hit"] += hit
                metrics[q_type][k]["recall"] += recall
                metrics[q_type][k]["count"] += 1
                
                metrics["OVERALL"][k]["hit"] += hit
                metrics["OVERALL"][k]["recall"] += recall
                metrics["OVERALL"][k]["count"] += 1

        results_to_save.append(case_result)
        print(f"[{idx}/{len(cases)}] Processed: {q_text[:40]}...")

    out_file = dataset_path.replace(".jsonl", "_results.jsonl")
    with open(out_file, "w", encoding="utf-8") as f:
        for r in results_to_save:
            f.write(json.dumps(r) + "\n")

    if not no_context:
        print("\n" + "="*50)
        print(" RETRIEVAL EVALUATION REPORT (HitRate@K / Recall@K)")
        print("="*50)
        
        for q_type, k_data in metrics.items():
            count = k_data[1]["count"]
            if count == 0: continue
            
            print(f"\nType: {q_type.upper()} (n={count})")
            print("-" * 40)
            for k in [1, 3, 5, 10]:
                hr = (k_data[k]["hit"] / count) * 100
                rec = (k_data[k]["recall"] / count) * 100
                print(f"  @K={k:<2} | HitRate: {hr:>5.1f}% | Recall: {rec:>5.1f}%")
                
        print(f"\nResults saved to: {out_file}")
        print("Add manual grades ('correct', 'partial', 'wrong', 'unsupported') to this file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V2 RAG Evaluation Engine")
    parser.add_argument("dataset", help="Path to the JSONL dataset (e.g., src/eval/kestrel.jsonl)")
    parser.add_argument("--generate", action="store_true", help="Generate full LLM answers (slow, costs tokens)")
    parser.add_argument("--no-context", action="store_true", help="Ask LLM without retrieval to test pre-training")
    
    args = parser.parse_args()
    run_evaluation(args.dataset, args.generate, args.no_context)
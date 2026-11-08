from typing import List, Dict, Any

def chunk_text(
    text: str,
    doc_name: str,
    chunk_size: int = 800,
    overlap_percent: int = 10,
    min_paragraph_length: int = 20
) -> List[Dict[str, Any]]:
    """
    Filters short noise paragraphs and splits clean text into sliding character windows.
    Returns dictionaries containing raw chunk text and structural metadata.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) >= min_paragraph_length]
    cleaned_text = "\n\n".join(paragraphs)

    overlap = (chunk_size * overlap_percent) // 100
    step = max(1, chunk_size - overlap)

    chunks = []
    chunk_idx = 0

    for start in range(0, len(cleaned_text), step):
        end = min(start + chunk_size, len(cleaned_text))
        chunk_str = cleaned_text[start:end]

        if chunk_str:
            chunks.append({
                "chunk_id": f"{doc_name}_c{chunk_idx}",
                "text": chunk_str,
                "metadata": {
                    "document_name": doc_name,
                    "chunk_id": f"{doc_name}_c{chunk_idx}",
                    "start_char": start,
                    "end_char": end,
                    "chunk_size": chunk_size,
                    "overlap_percent": overlap_percent
                }
            })
            chunk_idx += 1

        if end == len(cleaned_text):
            break

    return chunks
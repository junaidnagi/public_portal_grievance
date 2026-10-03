"""Semantic retrieval over saved FAISS vectors; no pickle or Groq embedding API."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import faiss
import numpy as np
ROOT = Path(__file__).parent
INDEX_DIR = ROOT / 'faiss_index'
DEFAULT_EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

@lru_cache(maxsize=2)
def get_embedder(model_name: str = DEFAULT_EMBEDDING_MODEL):
    from fastembed import TextEmbedding
    return TextEmbedding(model_name=model_name, threads=2, cache_dir=str(ROOT / '.embedding_cache'))

def embed_texts(texts: list[str], model_name: str = DEFAULT_EMBEDDING_MODEL) -> np.ndarray:
    values = np.asarray(list(get_embedder(model_name).embed(texts, batch_size=16)), dtype='float32')
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('Embedding model returned invalid vectors.')
    values = np.ascontiguousarray(values)
    faiss.normalize_L2(values)
    return values

@lru_cache(maxsize=4)
def _load_index(directory: str, manifest_contents: str):
    folder = Path(directory)
    manifest = json.loads(manifest_contents)
    for name in ['index.faiss', 'chunks.json']:
        if hashlib.sha256((folder/name).read_bytes()).hexdigest() != manifest['sha256'][name]:
            raise ValueError('Index files are incomplete or changed. Run ingest.py again.')
    index = faiss.read_index(str(folder/'index.faiss'))
    chunks = json.loads((folder/'chunks.json').read_text(encoding='utf-8'))
    if index.ntotal != len(chunks) or index.d != manifest['dimensions']:
        raise ValueError('Index and metadata do not match. Run ingest.py again.')
    return index, chunks, manifest

def search_index(query: str, authority: str | list[str] | None = None, top_k: int = 5, directory: Path = INDEX_DIR) -> list[dict]:
    if not query.strip():
        return []
    if not (directory/'manifest.json').exists():
        raise FileNotFoundError('FAISS index missing. Run python ingest.py --input policies first.')
    index, chunks, manifest = _load_index(str(directory.resolve()), (directory/'manifest.json').read_text(encoding='utf-8'))
    vector = embed_texts([query], manifest['embedding_model'])
    if vector.shape[1] != index.d:
        raise ValueError('Query embedding dimension differs from the saved index.')
    authorities = [authority] if isinstance(authority,str) else authority
    allowed = {i for i,c in enumerate(chunks) if authorities is None or c['authority'] in authorities + ['General']}
    if not allowed:
        return []
    scores, ids = index.search(vector, index.ntotal)
    candidates = []
    for score, chunk_id in zip(scores[0], ids[0]):
        if int(chunk_id) not in allowed or float(score) < 0.25:
            continue
        candidates.append({**chunks[int(chunk_id)], 'score': round(float(score),4)})
    # Cap repeated summary hits and reserve room for primary PDF evidence.
    primary = [c for c in candidates if c['source_kind']=='policy_pdf']
    summaries, seen = [], set()
    for item in candidates:
        if item['source_kind']!='policy_pdf' and item['source_file'] not in seen:
            summaries.append(item)
            seen.add(item['source_file'])
    if primary and summaries:
        take_summary = min(2,max(0,top_k-1))
        results = primary[:top_k-take_summary] + summaries[:take_summary]
        if len(results)<top_k:
            included = {item['chunk_id'] for item in results}
            results += [item for item in primary+summaries if item['chunk_id'] not in included][:top_k-len(results)]
        return sorted(results, key=lambda c:c['score'], reverse=True)
    return (primary or summaries)[:top_k]

# Lexical preview for temporary uploads only; saved-corpus search uses semantic embeddings.
def retrieve(query: str, documents: list[dict], limit: int = 4) -> list[dict]:
    import re
    words = set(re.findall(r'\w+',query.lower()))
    scored = [(len(words & set(re.findall(r'\w+',d['text'].lower()))),d) for d in documents]
    return [d for score,d in sorted(scored,key=lambda x:x[0],reverse=True) if score][:limit]

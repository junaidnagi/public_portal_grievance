"""Semantic retrieval over saved FAISS vectors; no pickle or Groq embedding API."""
import hashlib
import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
import faiss
import numpy as np
ROOT = Path(__file__).parent
INDEX_DIR = ROOT / 'faiss_index'
DEFAULT_EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

# The citizen's description often matches a conduct rule without matching the
# separate rules explaining who accepts complaints. Search both questions.
GUIDANCE_QUERIES = {
    'PEMRA': [
        'Filing complaint any person aggrieved by a programme or advertisement may lodge a complaint before Council of Complaints or authorized officer where programme is viewed.',
        'Council shall receive and review complaints about programmes broadcast or distributed by a station through a licence issued by the Authority.'
    ],
    'NEPRA': ['Electricity consumer complaint eligibility filing with NEPRA after complaint to distribution company licensee complaint handling procedure.'],
    'IESCO': ['IESCO electricity billing meter supply complaint registration first contact distribution company customer service.'],
    'PTA': ['PTA consumer complaint registration service provider operator complaint number Complaint Management System procedure.']
}
STOP_WORDS = set('a an and are as at be before by for from has have in is it of on or shall that the their this to under with'.split())


def _words(text: str) -> list[str]:
    return [word for word in re.findall(r'\w+', text.lower())
            if len(word) > 1 and word not in STOP_WORDS]


def _keyword_scores(query: str, chunks: list[dict], allowed: set[int]) -> dict[int, float]:
    """Small BM25-style fallback, using only the validated local corpus."""
    terms = set(_words(query))
    counts = {i: Counter(_words(chunks[i]['text'])) for i in allowed}
    if not terms or not counts:
        return {}
    average = max(1, sum(sum(c.values()) for c in counts.values()) / len(counts))
    frequencies = {term: sum(term in c for c in counts.values()) for term in terms}
    scores = {}
    for i, count in counts.items():
        score = 0.0
        length = sum(count.values())
        for term in terms:
            frequency = count[term]
            if frequency:
                idf = math.log(1 + (len(counts) - frequencies[term] + 0.5) / (frequencies[term] + 0.5))
                score += idf * frequency * 2.2 / (frequency + 1.2 * (0.25 + 0.75 * length / average))
        if score:
            scores[i] = score
    largest = max(scores.values(), default=1)
    return {i: score / largest for i, score in scores.items()}


def _with_context(hit: dict, chunks: list[dict]) -> dict:
    """Join adjacent chunks on the same page so a rule is not cut mid-sentence."""
    page_chunks = sorted((c for c in chunks if
        c['source_file'] == hit['source_file'] and c.get('page') == hit.get('page')),
        key=lambda c: c.get('char_start', 0))
    position = next(i for i, c in enumerate(page_chunks) if c['chunk_id'] == hit['chunk_id'])
    neighbours = page_chunks[max(0, position-1):position+2]
    text = neighbours[0]['text']
    end = neighbours[0].get('char_end', len(text))
    for item in neighbours[1:]:
        start = item.get('char_start', end)
        overlap = max(0, end - start)
        text += ('\n' if start > end else '') + item['text'][overlap:]
        end = item.get('char_end', start + len(item['text']))
    return {**hit, 'text': text, 'char_start': neighbours[0].get('char_start', 0),
            'char_end': end, 'context_chunk_ids': [c['chunk_id'] for c in neighbours]}

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
    if not query.strip() or top_k <= 0:
        return []
    if not (directory/'manifest.json').exists():
        raise FileNotFoundError('FAISS index missing. Run python ingest.py --input policies first.')
    index, chunks, manifest = _load_index(str(directory.resolve()), (directory/'manifest.json').read_text(encoding='utf-8'))
    authorities = [authority] if isinstance(authority,str) else authority
    allowed = {i for i,c in enumerate(chunks) if authorities is None or c['authority'] in authorities + ['General']}
    if not allowed:
        return []
    intents = [(query, 'complaint', allowed)]
    # A general values objection is not automatically a religious-hatred case.
    # Retrieve the broad programme principles rather than letting the word
    # 'religious' select only news, advertisements or sectarianism passages.
    words = set(_words(query))
    if 'PEMRA' in (authorities or []) and (words & {
        'islam', 'islamic', 'religion', 'religious', 'culture', 'cultural',
        'اسلام', 'اسلامی', 'مذہب', 'مذہبی', 'ثقافت', 'ثقافتی'}):
        intents.append(('Fundamental principles: no content is aired which is against '
            'Islamic values, ideology of Pakistan; general programme content standards.',
            'complaint', allowed))
        intents.append(('Preservation of national cultural social and religious values; '
            'programme standards of decency, obscenity or vulgarity.',
            'complaint', allowed))
    for name in authorities or []:
        scoped = {i for i in allowed if chunks[i]['authority'] in (name, 'General')}
        for guidance in GUIDANCE_QUERIES.get(name, []):
            if scoped:
                intents.append((guidance, 'procedure', scoped))
    try:
        vectors = embed_texts([intent[0] for intent in intents], manifest['embedding_model'])
    except Exception:
        # A model download/cache failure must not discard readable source text.
        # Hash/index corruption is checked above and is never bypassed here.
        vectors = None
    if vectors is not None and vectors.shape[1] != index.d:
        raise ValueError('Query embedding dimension differs from the saved index.')
    rankings = []
    for position, (text, purpose, scoped) in enumerate(intents):
        keywords = _keyword_scores(text, chunks, scoped)
        candidates = []
        if vectors is None:
            scores_ids = sorted(((score, i) for i, score in keywords.items()), reverse=True)
            method = 'keyword_fallback'
        else:
            scores, ids = index.search(vectors[position:position+1], index.ntotal)
            scores_ids = zip(scores[0], ids[0])
            method = 'semantic'
        for score, chunk_id in scores_ids:
            chunk_id = int(chunk_id)
            if chunk_id not in scoped or float(score) < (0.15 if vectors is None else 0.25):
                continue
            candidates.append({**chunks[chunk_id], 'score': round(float(score), 4),
                'ranking_score': float(score) * 0.7 + keywords.get(chunk_id, 0) * 0.3,
                'retrieval_purpose': purpose, 'retrieval_method': method})
        rankings.append(sorted(candidates, key=lambda c: c['ranking_score'], reverse=True))
    results, pages, summaries = [], set(), set()

    def add(item):
        key = (item['source_file'], item.get('page'))
        is_summary = item.get('source_kind') != 'policy_pdf'
        if len(results) >= top_k or key in pages or (is_summary and summaries):
            return False
        results.append(item)
        pages.add(key)
        if is_summary:
            summaries.add(item['source_file'])
        return True

    # Reserve room for the route, one summary and the actual complaint content.
    procedure_slots = min(2, max(0, top_k-2))
    for intent, ranking in zip(intents, rankings):
        if intent[1] != 'procedure':
            continue
        if len(results) >= procedure_slots:
            break
        primary = [c for c in ranking if c.get('source_kind') == 'policy_pdf']
        for item in primary or ranking:
            if add(item):
                break
    all_candidates = sorted([c for ranking in rankings for c in ranking],
                            key=lambda c: c['ranking_score'], reverse=True)
    for item in all_candidates:
        if item.get('source_kind') != 'policy_pdf' and add(item):
            break
    content_rankings = [ranking for intent, ranking in zip(intents, rankings)
                        if intent[1] == 'complaint']
    # Where a content-specific expansion was needed, prefer its general
    # provisions before falling back to the original wording's candidates.
    ordered_content = content_rankings[1:] + content_rankings[:1]
    for ranking in ordered_content:
        for item in [c for c in ranking if c.get('source_kind') == 'policy_pdf']:
            if add(item):
                break
        if len(results) >= top_k:
            break
    content = [item for ranking in ordered_content for item in ranking]
    for item in content + all_candidates:
        add(item)
        if len(results) >= top_k:
            break
    return [_with_context(item, chunks) for item in results]

# Lexical preview for temporary uploads; corpus search normally uses semantic embeddings.
def retrieve(query: str, documents: list[dict], limit: int = 4) -> list[dict]:
    import re
    words = set(re.findall(r'\w+',query.lower()))
    scored = [(len(words & set(re.findall(r'\w+',d['text'].lower()))),d) for d in documents]
    return [d for score,d in sorted(scored,key=lambda x:x[0],reverse=True) if score][:limit]

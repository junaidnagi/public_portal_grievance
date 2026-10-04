"""Ingestion/index integrity tests, no embedding-model downloads required."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
import ingest
import rag

class IngestChecks(unittest.TestCase):
    def tokenizer(self):
        tokenizer = Tokenizer(WordLevel({'[UNK]':0, 'electricity':1,'billing':2,'internet':3},unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        return tokenizer
    def test_chunking(self):
        text = ' '.join(['electricity']*240)
        parts = ingest.split_text(text,self.tokenizer())
        self.assertEqual(len(parts),3)
        for first,last,value in parts:
            self.assertEqual(value, text[first:last])
            self.assertLessEqual(len(self.tokenizer().encode(value).ids),96)
        with self.assertRaises(ValueError):
            ingest.split_text(text,self.tokenizer(),10,10)
    def test_corrupt_and_utf8_documents(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder/'NEPRA_summary.txt').write_text('Electricity complaint guidance. '*5, encoding='utf-8')
            (folder/'damaged.pdf').write_bytes(b'invalid')
            pages,report=ingest.read_documents(folder)
            self.assertEqual(pages[0]['source_file'],'NEPRA_summary.txt')
            self.assertEqual(pages[0]['authority'],'NEPRA')
            self.assertTrue(report['warnings'])
    def test_roundtrip_and_integrity(self):
        tokenizer=self.tokenizer()
        fake=SimpleNamespace(model=SimpleNamespace(tokenizer=tokenizer))
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/'policies';folder.mkdir()
            output=Path(temp)/'faiss_index'
            (folder/'NEPRA_policy.txt').write_text('electricity billing complaint guidance. '*5)
            def encode(texts,model):
                return np.tile(np.array([1,0,0],dtype='float32'),(len(texts),1))
            with patch('ingest.get_embedder',return_value=fake), patch('ingest.embed_texts',side_effect=encode):
                report=ingest.build_index(folder,output)
            self.assertGreater(report['chunk_count'],0)
            with patch('rag.embed_texts',side_effect=encode):
                result=rag.search_index('electricity', authority='NEPRA',directory=output)
                self.assertEqual(result[0]['source_file'],'NEPRA_policy.txt')
                self.assertEqual(rag.search_index('electricity',authority='PEMRA',directory=output),[])
            metadata=json.loads((output/'chunks.json').read_text())
            self.assertIn('chunk_id',metadata[0])
            self.assertFalse(metadata[0]['verified'])
            (output/'chunks.json').write_text('[]')
            rag._load_index.cache_clear()
            with patch('rag.embed_texts',side_effect=encode), self.assertRaises(ValueError):
                rag.search_index('electricity',directory=output)
    def test_missing_index(self):
        with tempfile.TemporaryDirectory() as temp, self.assertRaises(FileNotFoundError):
            rag.search_index('electricity',directory=Path(temp))

    def test_content_search_also_returns_distinct_primary_routing_rules(self):
        import faiss
        import hashlib
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            first = 'Filing complaints: any person aggrieved by a programme may lodge '
            second = 'a complaint before the Council of Complaints or authorized officer where it is viewed.'
            rows = [
                ('PEMRA_Code.pdf', 8, 'Religious tolerance: programmes must not incite religious hatred.', [0.98, 0.1, 0.1], 'policy_pdf', 'PEMRA', 0),
                ('PEMRA_COC.pdf', 3, first, [0.1, 0.98, 0.1], 'policy_pdf', 'PEMRA', 0),
                ('PEMRA_COC.pdf', 3, second, [0.1, 0.95, 0.1], 'policy_pdf', 'PEMRA', len(first)),
                ('PEMRA_Ordinance.pdf', 23, 'Each Council shall receive and review complaints against programmes broadcast through a licence issued by the Authority.', [0.1, 0.1, 0.98], 'policy_pdf', 'PEMRA', 0),
                ('PEMRA_summary.txt', None, 'PEMRA receives complaints about television programmes.', [0.4, 0.4, 0.4], 'secondary_summary', 'PEMRA', 0),
                ('NEPRA_policy.pdf', 1, 'Electricity consumer billing complaint.', [1, 0, 0], 'policy_pdf', 'NEPRA', 0)
            ]
            chunks = [dict(source_file=name, page=page, text=text, source_kind=kind,
                authority=authority, chunk_id=str(i), char_start=start, char_end=start+len(text))
                for i, (name, page, text, vector, kind, authority, start) in enumerate(rows)]
            vectors = np.array([row[3] for row in rows], dtype='float32')
            faiss.normalize_L2(vectors)
            index = faiss.IndexFlatIP(3)
            index.add(vectors)
            faiss.write_index(index, str(folder/'index.faiss'))
            (folder/'chunks.json').write_text(json.dumps(chunks))
            manifest = {'dimensions': 3, 'embedding_model': 'synthetic', 'sha256': {
                name: hashlib.sha256((folder/name).read_bytes()).hexdigest()
                for name in ('index.faiss', 'chunks.json')}}
            (folder/'manifest.json').write_text(json.dumps(manifest))
            with patch('rag.embed_texts', return_value=np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0],
                        [0, 1, 0], [0, 0, 1]], dtype='float32')) as embed:
                hits = rag.search_index('Television drama religious content', authority=['PEMRA'], directory=folder)
            self.assertEqual(len(embed.call_args.args[0]), 5)

            routes = [hit for hit in hits if hit['retrieval_purpose'] == 'procedure']
            self.assertEqual({hit['source_file'] for hit in routes}, {'PEMRA_COC.pdf', 'PEMRA_Ordinance.pdf'})
            self.assertTrue(any(hit['source_file'] == 'PEMRA_Code.pdf' for hit in hits))
            self.assertTrue(all(hit['authority'] == 'PEMRA' for hit in hits))
            self.assertEqual(len({(hit['source_file'], hit['page']) for hit in hits}), len(hits))
            coc = next(hit for hit in hits if hit['source_file'] == 'PEMRA_COC.pdf')
            self.assertIn('may lodge a complaint before the Council', coc['text'])
            self.assertEqual(coc['context_chunk_ids'], ['1', '2'])
            self.assertLessEqual(len(hits), 5)
            with patch('rag.embed_texts', side_effect=RuntimeError('Synthetic model download failure')):
                fallback = rag.search_index('Television drama religious content', authority=['PEMRA'], directory=folder)
            self.assertTrue(fallback)
            self.assertTrue(all(hit['retrieval_method'] == 'keyword_fallback' for hit in fallback))
            self.assertTrue(any('may lodge a complaint' in hit['text'] for hit in fallback))

    def test_values_objection_retrieves_general_principles_in_keyword_fallback(self):
        with patch('rag.embed_texts', side_effect=RuntimeError('Synthetic unavailable model')):
            hits = rag.search_index('A television drama conflicts with Islamic and cultural values; please review the content.', authority=['PEMRA'])
        primary = [hit for hit in hits if hit['source_kind'] == 'policy_pdf']
        self.assertTrue(any('may lodge a complaint' in hit['text'].lower() for hit in primary))
        self.assertTrue(any('is against the Islamic values' in hit['text'] for hit in primary))
        self.assertTrue(any('cultural' in hit['text'].lower() for hit in primary))

if __name__=='__main__': unittest.main()

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

if __name__=='__main__': unittest.main()

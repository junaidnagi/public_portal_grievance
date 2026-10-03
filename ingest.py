"""Run once: python ingest.py --input policies --output faiss_index."""
import argparse
import hashlib
import json
import shutil
import sys
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import faiss
from pypdf import PdfReader
from tokenizers import Tokenizer
from rag import DEFAULT_EMBEDDING_MODEL, embed_texts, get_embedder

def authority_for(path: str) -> str:
    for authority in ['IESCO','NEPRA','PEMRA','PTA']:
        if authority in path.upper():
            return authority
    return 'General'

def ocr_page(path: Path, page_number: int) -> str:
    if not shutil.which('tesseract') or not shutil.which('pdftoppm'):
        raise RuntimeError('OCR requires Tesseract and Poppler (pdftoppm) on PATH.')
    # File-hash cache prevents repeating OCR for unchanged documents.
    cache = Path(__file__).parent / '.ocr_cache'
    cache.mkdir(exist_ok=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    cached = cache / f'{digest}-{page_number}.txt'
    if cached.exists():
        return cached.read_text(encoding='utf-8')
    with tempfile.TemporaryDirectory() as directory:
        image = Path(directory)/'page'
        subprocess.run(['pdftoppm','-f',str(page_number),'-l',str(page_number),'-singlefile',
                        '-scale-to','2200','-png',str(path),str(image)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
        result = subprocess.run(['tesseract',str(image)+'.png','stdout','-l','eng'],
                                check=True,capture_output=True,text=True,timeout=60)
        cached.write_text(result.stdout,encoding='utf-8')
        return result.stdout


def read_documents(folder: Path, ocr: bool=False) -> tuple[list[dict],dict]:
    pages, report = [], {'files': [], 'warnings': []}
    if not folder.is_dir():
        raise ValueError('Input folder does not exist: '+str(folder))
    provenance_file = folder/'source_manifest.json'
    provenance = json.loads(provenance_file.read_text()) if provenance_file.exists() else {}
    files = sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in ['.pdf','.txt'])
    for path in files:
        name = path.relative_to(folder).as_posix()
        record = {'source_file':name,'pages_read':0,'characters':0,'ocr_pages':0}
        if path.stat().st_size > 30*1024*1024:
            report['warnings'].append(name+': over 30 MB; skipped.')
            report['files'].append(record)
            continue
        base = {'source_file':name,'source':name,'authority':authority_for(name),
                'source_kind':'secondary_summary' if path.suffix.lower()=='.txt' else 'policy_pdf',
                'verified':False,'file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'source_url':provenance.get(name,{}).get('source_url','')}
        extracted = []
        try:
            if path.suffix.lower()=='.txt':
                try:
                    text = path.read_text(encoding='utf-8-sig')
                except UnicodeDecodeError:
                    report['warnings'].append(name+': not UTF-8; convert encoding and re-run.')
                    text = ''
                extracted.append((None,text,'text_file'))
            else:
                reader = PdfReader(path)
                if reader.is_encrypted and not reader.decrypt(''):
                    report['warnings'].append(name+': password protected; skipped.')
                elif len(reader.pages)>500:
                    report['warnings'].append(name+': over 500 pages; split PDF first.')
                else:
                    for number,page in enumerate(reader.pages,1):
                        try:
                            text = page.extract_text() or ''
                            method = 'pdf_text'
                            if len(text.strip()) < 30 and ocr:
                                try:
                                    text = ocr_page(path, number)
                                    method = 'ocr'
                                    record['ocr_pages'] += 1
                                except Exception:
                                    report['warnings'].append(f'{name}: OCR failed on page {number}; check Tesseract/Poppler and PDF.')
                            extracted.append((number,text,method))
                        except Exception:
                            report['warnings'].append(f'{name}: extraction failed on page {number}; skipped page.')
        except Exception:
            report['warnings'].append(name+': unreadable/corrupt PDF; re-export it.')
        for number,text,method in extracted:
            text = text.replace('\x00','').strip()
            if len(text)<30:
                report['warnings'].append(f'{name}: page {number or "TXT"} has insufficient text; scans require OCR.')
                continue
            pages.append({**base,'page':number,'text':text,'extraction_method':method})
            record['pages_read'] += 1
            record['characters'] += len(text)
        report['files'].append(record)
    if not files:
        report['warnings'].append('No PDF or TXT documents found.')
    return pages,report

def split_text(text: str, tokenizer, chunk_tokens: int=96, overlap_tokens: int=16) -> list[tuple[int,int,str]]:
    if not 0 <= overlap_tokens < chunk_tokens:
        raise ValueError('Overlap must be smaller than chunk size.')
    offsets = tokenizer.encode(text,add_special_tokens=False).offsets
    chunks = []
    for start in range(0,len(offsets),chunk_tokens-overlap_tokens):
        end = min(start+chunk_tokens,len(offsets))
        first,last = offsets[start][0],offsets[end-1][1]
        value = text[first:last].strip()
        if value:
            chunks.append((first,last,value))
        if end==len(offsets):
            break
    return chunks

def build_index(input_folder: Path, output_folder: Path, model_name: str=DEFAULT_EMBEDDING_MODEL, ocr: bool=False) -> dict:
    pages,report = read_documents(input_folder, ocr=ocr)
    if not pages:
        raise ValueError('No usable text found. Check PDF text and TXT encoding. Image-only PDFs require OCR.')
    embedder = get_embedder(model_name)
    tokenizer = Tokenizer.from_str(embedder.model.tokenizer.to_str())
    tokenizer.no_truncation()
    tokenizer.no_padding()
    chunks = []
    for page in pages:
        for first,last,text in split_text(page['text'],tokenizer):
            identifier = hashlib.sha256(f"{page['source_file']}:{page['page']}:{first}:{text}".encode()).hexdigest()[:24]
            chunks.append({**page,'chunk_id':identifier,'text':text,'char_start':first,'char_end':last})
    vectors = embed_texts([c['text'] for c in chunks],model_name)
    if len(vectors)!=len(chunks):
        raise ValueError('Embedding count differs from chunk count.')
    index = faiss.IndexFlatIP(vectors.shape[1]); index.add(vectors)
    output_folder = output_folder.resolve()
    output_folder.parent.mkdir(parents=True,exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.faiss-build-',dir=output_folder.parent))
    backup = output_folder.with_name(output_folder.name+'.previous')
    try:
        faiss.write_index(index,str(stage/'index.faiss'))
        (stage/'chunks.json').write_text(json.dumps(chunks,ensure_ascii=False,indent=2),encoding='utf-8')
        report.update(chunk_count=len(chunks),readable_page_count=len(pages))
        (stage/'ingest_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        manifest = {'schema_version':1,'embedding_model':model_name,'dimensions':int(vectors.shape[1]),
                    'chunk_tokens':96,'overlap_tokens':16,'chunk_count':len(chunks),
                    'created_utc':datetime.now(timezone.utc).isoformat(),
                    'sha256':{name:hashlib.sha256((stage/name).read_bytes()).hexdigest() for name in ['index.faiss','chunks.json']}}
        (stage/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        if backup.exists(): shutil.rmtree(backup)
        if output_folder.exists(): output_folder.rename(backup)
        try:
            stage.rename(output_folder)
        except Exception:
            if backup.exists() and not output_folder.exists(): backup.rename(output_folder)
            raise
        if backup.exists(): shutil.rmtree(backup)
    finally:
        if stage.exists(): shutil.rmtree(stage)
    return report

def main() -> int:
    parser = argparse.ArgumentParser(description='Embed policy PDFs/TXT and save a cited FAISS index.')
    parser.add_argument('--input',type=Path,default=Path(__file__).parent/'policies')
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'faiss_index')
    parser.add_argument('--model',default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument('--ocr', action='store_true', help='OCR scanned pages using installed Tesseract/Poppler')
    args = parser.parse_args()
    try:
        report = build_index(args.input,args.output,args.model,ocr=args.ocr)
        print(f"Created {report['chunk_count']} chunks from {report['readable_page_count']} readable pages/sections in {args.output}")
        for warning in report['warnings']: print('Warning:',warning)
        print('Review ingest_report.json for skipped pages. Filenames and pages are preserved in chunks.json.')
        return 0
    except Exception as error:
        print('Ingestion failed:',str(error),file=sys.stderr)
        print('Check input folder, internet for the first model download, PDF text and dependencies. Existing index is preserved.',file=sys.stderr)
        return 1

if __name__=='__main__': raise SystemExit(main())

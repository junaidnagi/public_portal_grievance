# Public Grievance Portal — CrewAI + Groq + persisted semantic FAISS

This complete beginner project uses your Google Drive policy PDFs and TXT summaries. It includes a Streamlit UI you can change, six small CrewAI agent modules, `ingest.py`, real multilingual embeddings, saved FAISS retrieval, optional voice transcription, checklist scoring and SQLite tracking.

## Exact GitHub structure

Keep all files at the repository root, with these folders:

```text
public_grievance_portal/
    app.py
    ingest.py
    rag.py
    storage.py
    requirements.txt
    .gitignore
    README.md
    BUILD_PROMPT.md
    test_app.py
    test_ingest.py
    agents/
        __init__.py
        intake.py
        jurisdiction.py
        audit.py
        petition.py
        routing.py
        tracker.py
    policies/
        [your seven PDFs and four TXT summaries]
        source_manifest.json
    faiss_index/
        index.faiss
        chunks.json
        manifest.json
        ingest_report.json
```

If your GitHub repository already represents `public_grievance_portal`, upload its CONTENTS at the root. Do not add an extra nesting level. `app.py` should appear directly when opening the repository. Your downloaded policy files are included, unchanged. Original Google Drive files and sharing were not modified.

## What you receive

- `ingest.py`: recursively reads PDF/TXT files, extracts page text, optionally OCRs scans, chunks into 96 model tokens with 16-token overlap, embeds, normalizes and saves FAISS plus JSON metadata. Failed rebuilds preserve the old index.
- `rag.py`: loads the saved index, checks file hashes/metadata, embeds queries with the same model recorded in the manifest, applies authority filters and returns cited excerpts. It reserves results for primary PDF passages as well as summaries.
- `app.py`: Streamlit pages, Groq SDK connection, CrewAI orchestration, bounded error handling, privacy notice, demo fallback and optional voice input.
- `agents/`: six separate agent factories with role-specific goals.
- `storage.py`: SQLite case save/load/delete, isolated by a hashed private recovery key.
- `BUILD_PROMPT.md`: the updated complete prompt, ready to reuse.
- Tests: ingestion, source metadata, index consistency, classification, scoring, storage isolation, UI, mocked Groq and mocked speech processing.

## 1. Fastest deployment: use the included index

The package includes a FAISS index built from your documents. You can deploy it without running ingestion first.

1. Extract the ZIP.
2. Create/open your GitHub repository and upload every file and folder above. Include **faiss_index/** and **policies/**. The model cache, private recovery keys and SQLite database must NOT be uploaded.
3. Open https://share.streamlit.io/ and connect your GitHub account.
4. Create an app, select the repository/branch and choose `app.py` as the main file.
5. Open Advanced settings and select **Python 3.11**.
6. In Secrets, paste:

```toml
GROQ_API_KEY = "your-real-groq-api-key"
GROQ_MODEL = "openai/gpt-oss-20b"
```

7. Save and deploy.

Get your Groq key at https://console.groq.com/keys. Never put it in GitHub. `openai/gpt-oss-20b` is listed as a production model in Groq's model page checked for this delivery. You can change GROQ_MODEL without editing code. No OpenAI key or Google Drive credentials are required by the deployed app.

**First retrieval downloads the embedding model from Hugging Face.** This requires internet and can take a few minutes. The model is cached on that deployment afterward; a cloud restart may require downloading again. The index itself is bundled. If model/index loading fails, the app warns and continues without legal evidence. A missing Groq key produces a clearly labeled local template demo.

## 2. Rebuild after adding/changing documents

Install Python 3.11. Open a terminal INSIDE the project folder:

```bash
python -m venv .venv
```

Activate on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Activate on macOS/Linux:

```bash
source .venv/bin/activate
```

Then install and ingest:

```bash
python -m pip install -r requirements.txt
python ingest.py --input policies --output faiss_index
```

This command needs no Groq key. It reads local copies, not Google Drive links. To use another existing source directory, replace `policies` with its actual path; quote a path containing spaces. Put only regulatory PDF/TXT material in that input directory. Subfolders are read recursively. Re-ingest only when source files, embedding model or ingestion settings change; do not re-ingest for each complaint.

The model is **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2**, running locally through FastEmbed/ONNX. It produces real semantic vectors; this replaces the old hashed lexical vectors. It supports multilingual retrieval, but Urdu/Roman Urdu results still require testing and review. Query and document vectors use the same recorded model and normalization. Direct dependency versions are pinned. Do not mix index files from different builds.

## 3. Scanned PDFs / OCR

Three of your supplied PDFs needed OCR: NEPRA complaint rules, PEMRA Councils of Complaints rules and PEMRA Code of Conduct. The included index was built with OCR for those scans.

For your own rebuild, install **Tesseract OCR** and **Poppler** and ensure the `tesseract` and `pdftoppm` commands are on PATH. Then run:

```bash
python ingest.py --input policies --output faiss_index --ocr
```

On Ubuntu/Debian, install the system tools with:

```bash
sudo apt-get install tesseract-ocr poppler-utils
```

On Windows, install Windows builds of Tesseract and Poppler, add their executable directories to PATH, and reopen the terminal. OCR runs only for pages with insufficient extracted text, uses English recognition, has per-page timeouts and caches unchanged pages locally. It is a best-effort preprocessing step; manually verify OCR of legal sections, dates and numerals. A failed page is reported, never silently claimed as indexed.

Community Cloud does not need OCR system tools to SEARCH the prebuilt index. Run ingestion/OCR locally and upload the generated index. One page in PEMRA Ordinance remained without sufficient text; check `faiss_index/ingest_report.json` for details. No missing legal text was invented.

## 4. What is inside faiss_index?

| File | Meaning |
|---|---|
| index.faiss | Normalized embedding vectors used for similarity search |
| chunks.json | Chunk text, source_file, page/section, authority, source_kind, extraction method and unique chunk ID |
| manifest.json | Embedding model, dimensions, chunk settings, count and integrity hashes |
| ingest_report.json | Files/pages processed, OCR counts and extraction warnings |

Example metadata (illustrative):

```json
{
  "source_file": "NEPRA_Consumer_Service_Manual_2025.pdf",
  "page": 25,
  "authority": "NEPRA",
  "source_kind": "policy_pdf",
  "extraction_method": "pdf_text",
  "chunk_id": "unique identifier",
  "text": "Retrieved passage...",
  "verified": false
}
```

`verified: false` means we have not independently authenticated the copy, checked all amendments or established applicability to a particular complaint. TXT summaries are marked secondary. Their contents are not treated as statutes. The Drive provenance is recorded in `policies/source_manifest.json`, and its source links are copied into chunk metadata. No pickle is used. Only load indexes supplied by a trusted project maintainer; integrity hashes catch accidental mixing/corruption, not malicious replacement of all files.

## 5. Run the app locally

For live mode, create `.streamlit/secrets.toml` locally containing the same secrets above. This file is ignored by Git. Then:

```bash
python -m streamlit run app.py
```

The app has Home, New Complaint, Document preparation, My Cases, Regulations, Analytics and About pages.

## 6. Beginner demo walkthrough

1. Open New Complaint and enter Name, City, Complaint description and Complaint category.
2. Try: **My IESCO electricity bill is Rs 45,000 although my normal bill is approximately Rs 8,000. I contacted IESCO but the issue is unresolved.**
3. Click **Analyze Complaint**. Review six tabs: summary, authority, document preparation, complaint letter, submission/escalation and tracking. Relevant policy excerpts are shown with source filenames/pages.
4. Open Document preparation and check the items you actually have. Four of five gives 80%. This is a generic self-reported checklist, not official filing eligibility.
5. Return to New Complaint. Click Generate Complaint to regenerate a local fact-based template with updated attachments, or edit the AI letter yourself. Download TXT and click Save complaint.
6. In My Cases, download the private recovery key. After filing yourself, enter the actual complaint reference and update status. Click Complaint Not Resolved for a tentative escalation suggestion requiring source and eligibility review.
7. Opt in to your aggregate analytics and view category, status and resolved/unresolved charts.

Live mode runs six real CrewAI agents on Groq. Demo mode and API-error fallback generate deterministic local output and clearly say no CrewAI/LLM call completed. Nothing is submitted automatically. IESCO is an initial provider only where its service territory applies; NEPRA is the relevant regulator example. Telecom/PTA and Media/Broadcasting/PEMRA are distinct categories, not one generic escalation chain.

## 7. Voice input

Open **Optional voice input** on New Complaint. Record English or Urdu, then click Transcribe recording. This sends the recording to Groq's multilingual `whisper-large-v3-turbo`; a real Groq key is required even if complaint generation uses demo mode. The transcript fills Complaint description for you to review before Analyze Complaint. Microphone permission is controlled by your browser. Audio is limited to 20 MB, and empty recordings, rate limits, timeouts and service errors are handled. Live microphone and transcription accuracy require your account and were not tested with real audio.

## Error handling

| Issue | Behavior |
|---|---|
| Missing name/city/complaint | Clear validation message |
| Missing key | Local template demo with explicit label |
| Groq 429 | Bounded Retry-After waits, then fallback |
| Invalid key/model, timeout or server error | Sanitized message and local template; no raw key or response body printed |
| Missing/corrupt FAISS files | Rebuild instruction; app continues without legal evidence |
| Embedding download unavailable | Warning; check internet/cache or retry |
| Corrupt/encrypted PDF or scanned page | Per-file/page warning in ingestion report; optional OCR |
| Save/load failure | Recovery message and JSON backup download |
| Empty voice recording | Recording/type-input guidance |

## Important limits

SQLite lives on the deployment's local disk; Community Cloud restarts/redeployments can erase it. Download JSON case backups and the private recovery key. The key isolates cases while the SQLite file exists; it is not a full login system. Anyone with it can access the saved cases. Authentication and an external durable database are future work.

AI text is a draft requiring fact and legal review. CNIC-like numbers are masked before LLM transmission; avoid entering sensitive identifiers in the first place. Groq keys come only from secrets; public deployment needs access/quota protection. Voice audio is sent only when you request transcription. Analytics contains only opted-in aggregate counts for your workspace.

Not implemented: automatic government submission, binding statutory deadlines, calendar/WhatsApp notifications, public heatmap, account authentication or PDF complaint-letter export. Letters export as TXT, case backups as JSON. Regulatory PDF ingestion and optional offline OCR are implemented. Policy PDFs are copied from your folder and include versioned originals; ingestion does not prove they remain current.

## Tests and deployment status

Run:

```bash
python -m unittest -v test_app.py test_ingest.py
```

All 15 tests passed during creation. Tests use mocked Groq calls, so no key is needed. Ingestion and query retrieval were also run with the real local embedding model on your corpus. Python 3.11 is the deployment target; the available execution runtime was Python 3.12. Package compatibility declarations support 3.11, but a live Community Cloud deployment and real Groq calls still need your account. Do not claim these have already been validated.

## Official technical references

- https://docs.crewai.com/en/learn/custom-llm
- https://console.groq.com/docs/models
- https://console.groq.com/docs/speech-to-text
- https://qdrant.github.io/fastembed/examples/Supported_Models/
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

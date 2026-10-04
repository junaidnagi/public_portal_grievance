# Public Grievance Assistant — updated pitch features

Replace the files in your existing project with this package. Keep your existing deployment secrets. Upload the CONTENTS of public_grievance_portal at your repository root; app.py should sit beside requirements.txt, packages.txt, rag.py and storage.py.

## Implemented in this release

- Bill-photo OCR with local Tesseract, optional explicitly authorized Groq vision, extraction review and account/date/amount form filling.
- PDF complaint letters and complaint/evidence ZIP packets.
- Source-backed, user-confirmed statutory timeline plans, personal follow-up dates and downloadable calendar reminders (.ics).
- Reviewed escalation packets to the relevant regulator, or Wafaqi Mohtasib where federal-agency maladministration jurisdiction applies.
- Explicit authorization of the exact future email, recipient and selected files; signed jobs, cancellation, changed-case checks and per-recipient duplicate protection.
- A scheduler for due actions without an open browser on an always-on host.
- An opt-in public city/sector heatmap; demo cases excluded and groups below three distinct recovery keys suppressed.
- Legal ingestion labels and separate index builds for FIA, POLICE, NCCIA, MUNICIPAL, MOHTASIB and RTS, including provincial metadata.

The guided details → review → evidence → submission → tracking/disposal flow, company/regulator selection, directories, Urdu voice transcription, six CrewAI agents and existing sector RAG remain available. The separate AI-check button remains removed.

## Deployment

1. Upload the full project contents, including requirements.txt and packages.txt. Replacing app.py alone does not install the new PDF/OCR dependencies.
2. Retain your GROQ_API_KEY and GROQ_MODEL in Streamlit Secrets. Do not put secrets, SQLite databases, recovery keys or model caches in GitHub.
3. Install locally with `python -m pip install -r requirements.txt`; run `python -m streamlit run app.py`.
4. Streamlit Community Cloud installs packages.txt. Local bill OCR needs Tesseract; the Urdu option also needs its Urdu language data. Optional cloud OCR requires Groq and explicit photo-sharing consent. Review every extracted value before applying it.
5. Actual email delivery needs your authorized SMTP account or Resend configuration and a verified recipient route. Preserve the existing sender settings. Nothing is submitted merely by opening a portal.

Example secret names (supply your own real values privately): GROQ_API_KEY, SMTP_HOST, SMTP_PORT, SMTP_SECURITY, SMTP_USERNAME, SMTP_PASSWORD, SMTP_FROM; alternatively RESEND_API_KEY and SUBMISSION_FROM. GROQ_VISION_MODEL is optional. The app accepts additional verified COMPLAINT_ROUTES and COMPLAINT_PORTALS using its documented dictionaries.

## Timeline and escalation

Open My Cases and save the official reference. In Deadlines, reminders and automatic escalation, choose a reviewed rule, record the actual starting date and confirm its applicability and exceptions. NEPRA's general prior-company response period can be replaced by a complaint-specific applicable rule. The ombudsman's three-calendar-month ordinary limitation is a filing period, not a universal response deadline; jurisdiction exclusions and possible late-filing condonation need review. Personal follow-up dates assert no statutory deadline.

Save the unresolved-case and jurisdiction declarations, review the exact future message and selected files, then explicitly enable preparation or email escalation. Closing the case, changing the reviewed details or changing the recipient invalidates or cancels pending authorization. Email transmission is recorded separately from official registration and resolution. No portal login or CAPTCHA is bypassed.

The app checks due jobs when opened. For execution while it is closed, run `python scheduler.py --watch` on an always-on host, or run `python scheduler.py` from cron every minute. The worker must use the SAME case database, workflow database, outbox and private secrets as the UI. Community Cloud is not an always-on scheduler. `--prepare-only` suppresses email jobs. Keep all databases on durable private storage and retain backups. CASE_DB_PATH can set the case database through the environment; AUTOMATION_DB_PATH and SUBMISSION_DB_PATH are server secrets. Do not delete duplicate-delivery records to retry uncertain sends.

## Legal source coverage

The supplied faiss_index preserves the existing IESCO/NEPRA/PTA/PEMRA corpus. It has NOT been rebuilt or claimed to contain complete new laws. Add current primary PDFs under policies/FIA/, policies/POLICE/Punjab/ and equivalent collection/province folders. Then run:

`python ingest.py --input policies --all-collections --ocr`

Separate indexes go under legal_indexes/AUTHORITY/. A normal `python ingest.py --input policies --output faiss_index --ocr` rebuilds the combined index. Missing or failed collection builds leave the supplied sector index intact. Regulations shows whether separate indexes exist. Source copies and procedural dates require currency and applicability checks; contact directories are not statutory evidence.

Custom deadline rules may be added by the administrator under DEADLINE_RULES with verified=true, an HTTPS primary source, provision, scope, label, start_label, category, purpose=response/filing, kind=calendar_days/working_days/calendar_months and positive amount. Working-day rules require a maintained holiday list. No universal telecom, police or municipal complaint deadline is assumed.

Timeline sources:
- https://nepra.org.pk/Legislation/2-Rules/2.11%20NEPRA%20Complaint%20Handling%20and%20Dispute%20Resolution%20%28Procedure%29%20Rules%2C%202015/NEPRA%27s%20Compalint%20Handling%20and%20Dispute%20Resolution%20%28Procedure%29%20Rules%202105.PDF
- https://www.mohtasib.gov.pk/SiteImage/Downloads/Compendium%20for%20investigation%2026.12.24-latest.pdf

## Verification

Run `python -m unittest test_pitch_features -v` for OCR parsing, actual local OCR where installed, PDF export/privacy, dates and calendar files, duplicate email dispatch, changed/closed/tampered jobs, ombudsman checks, opt-in aggregation and the scheduler. Email and AI transport are mocked in tests; no real complaint is sent. The guided screens were checked with Streamlit AppTest. Live Groq vision, real email delivery and official portal registration require your configured deployment and have not been exercised in this release.

Cases use a hashed private recovery key, not full account authentication. SQLite files contain private case material and need restricted access, durable hosting and backups. The public view contains only aggregate city/sector counts at approximate city centres. Supporting evidence never enters the public dataset.

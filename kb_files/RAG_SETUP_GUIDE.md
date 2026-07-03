# Building the BVRITH College RAG Chatbot — Setup Guide

## 1. Get the full corpus
Run `crawl_college_site.py` on a machine with normal internet access:
```bash
pip install requests beautifulsoup4 tqdm pypdf
python crawl_college_site.py
```
This walks every internal link from the homepage, strips nav/header/footer
boilerplate, and writes clean text per page to `bvrith_corpus/`. It also logs
every PDF it finds (fee structure, NAAC/NBA reports, syllabi, committee docs)
to `pdf_links.txt` — parse those separately with `pypdf`/`pdfplumber` since
colleges keep a lot of official data (fees, admissions circulars) in PDFs, not
HTML.

## 2. Chunk the text
Split each page's text into ~300–500 token chunks with ~50 token overlap.
Keep the source URL and page title attached to every chunk — you'll want to
cite/verify against them later, and it also lets you filter by category
(e.g. "admissions", "placements", "departments/cse").

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(page_text)
```

## 3. Embed and store
Any vector DB works for a corpus this size (a few hundred pages):
- **Simple/free**: Chroma or FAISS, run locally
- **Managed**: Pinecone, Qdrant Cloud, Weaviate Cloud
- **Embeddings**: OpenAI `text-embedding-3-small`, or Voyage AI's embeddings
  (Anthropic recommends Voyage for use with Claude), or a local model like
  `all-MiniLM-L6-v2` if you want zero API cost.

## 4. Retrieval + generation
Standard RAG loop: embed the user's question → retrieve top-k (5–8) chunks →
pass them to Claude via the API with a system prompt that (a) restricts scope
and (b) forces grounding in retrieved context.

### System prompt (scope-restriction is the important part)
```
You are BVRITH Assistant, the official chatbot for BVRIT HYDERABAD College of
Engineering for Women (bvrithyderabad.edu.in). Answer ONLY using the context
provided below, which was retrieved from the college's official website.

Rules:
- If the answer is not in the provided context, say you don't have that
  information and suggest the person contact the college directly
  (info@bvrithyderabad.edu.in / +91 40 4241 7773) or check the relevant page
  on bvrithyderabad.edu.in.
- If the question is unrelated to BVRITH (general knowledge, other colleges,
  coding help, etc.), politely decline and redirect the user to ask something
  about BVRITH — admissions, departments, placements, facilities, fees,
  faculty, events, etc.
- Never invent facts, fee amounts, dates, or contact details not present in
  the context.
- Keep answers concise and cite which page the info came from when useful.

Context:
{retrieved_chunks}

Question: {user_question}
```

This "answer only from context + explicit refusal instruction for
off-topic/unknown questions" pattern is what keeps a RAG bot on-topic — the
retrieval step naturally limits what's available, and the system prompt
handles the edge cases (off-topic questions, or in-scope questions the corpus
doesn't cover).

### Extra safety net (optional but recommended)
Add a lightweight classifier step before retrieval: if the embedding
similarity of the question to *any* chunk is below a threshold, skip
generation and return the "I don't have that information" message directly
— avoids the model getting creative when nothing relevant was retrieved.

## 5. Keep it fresh
College sites change often (fee updates, new placement stats, event
announcements). Re-run the crawler on a schedule (e.g. weekly cron job) and
re-embed changed pages — hash each page's text and only re-embed chunks whose
hash changed, to keep costs down.

## 6. Suggested category tags for chunk metadata
Based on the site structure, tag chunks with:
`about | admissions | departments/{cse,aiml,ece,eee,it,bsh} | placements |
research | facilities | governance | student-clubs | news | contact`
This lets you do metadata-filtered retrieval (e.g., if the question mentions
"fee" or "admission", boost the `admissions` category) on top of pure vector
similarity.

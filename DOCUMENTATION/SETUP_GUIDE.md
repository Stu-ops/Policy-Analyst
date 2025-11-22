# 🚀 Agentic RAG Policy Analyst - Setup & Deployment Guide

## Quick Start

### 1. Set Up Secrets (Required)
In your Replit workspace, go to **Tools → Secrets** and add these:

**Required:**
```
GEMINI_API_KEY = your_api_key_from_https://aistudio.google.com/apikey
```

**Auto-provided by Replit:**
```
DATABASE_URL
PGHOST
PGPORT
PGUSER
PGPASSWORD
PGDATABASE
SESSION_SECRET
```

### 2. Run the Application
Click **Run** or execute:
```bash
streamlit run app.py --server.port 5000
```

### 3. First Time Using the App
1. Open the preview URL (http://0.0.0.0:5000)
2. Enter your Gemini API key in the **🔑 API Configuration** sidebar
3. Upload insurance policy documents (PDF, DOCX, Excel, TXT)
4. Ask questions about your policies!

---

## Project Structure

```
workspace/
├── app.py                      # Main Streamlit application
├── agentic_orchestrator.py     # Multi-turn ReAct agent logic
├── mcp_server.py              # MCP data server (document management)
├── hybrid_search.py           # FAISS + BM25 hybrid search
├── performance_monitor.py     # Performance metrics & analytics
├── document_processor.py      # Document parsing (PDF/DOCX/Excel)
├── database.py                # PostgreSQL operations
├── actuarial_tools.py         # Insurance calculations
├── contradiction_detector.py  # Policy contradiction detection
├── .env.example               # Environment variables template
├── DEPENDENCIES.md            # Package requirements
└── SETUP_GUIDE.md            # This file
```

---

## Environment Variables

### API & Authentication
- `GEMINI_API_KEY` - Your Google Gemini API key (get free at https://aistudio.google.com/apikey)

### Database (Auto-provided by Replit)
- `DATABASE_URL` - Full PostgreSQL connection string
- `PGHOST` - Database server hostname
- `PGPORT` - Database server port (5432)
- `PGUSER` - Database user
- `PGPASSWORD` - Database password
- `PGDATABASE` - Database name

### Application
- `SESSION_SECRET` - Session encryption key (auto-generated)

---

## Features

### 🤖 Agentic AI
- **Multi-turn ReAct Loop** - Plan → Search → Verify → Refine → Answer
- **Google Gemini 2.5 Flash** with deep thinking capabilities
- **Autonomous reasoning** with transparent steps

### 🔍 Intelligent Search
- **Hybrid Search**: FAISS semantic (70%) + BM25 keyword (30%)
- **Search Caching** - 10-100x faster for repeated queries
- **Advanced Error Handling** - Graceful degradation

### 📊 Performance Monitoring
- Real-time query analytics
- Cache hit/miss tracking
- Error logging and reporting
- API cost tracking

### 📄 Document Management
- Upload PDF, DOCX, Excel, TXT files
- Automatic chunking and indexing
- Contradiction detection across policies

### 💰 Insurance Tools
- Premium calculation with risk adjustments
- Deductible impact analysis
- Policy clause cross-referencing

---

## Usage Examples

### Ask Policy Questions
```
"What are the coverage limits for property damage?"
"What's the deductible policy for claims?"
"Are there exclusions for flood damage?"
"How do I calculate premiums with risk factors?"
```

### Use Advanced Features

**Contradiction Detection** (Tab 4: Contradictions)
- Uploads 2+ policies
- Click "Detect Contradictions"
- See policy conflicts highlighted

**Actuarial Calculator** (Tab 2: Actuarial)
- Select policy type (auto, health, home, life)
- Set risk factor
- Calculate premium instantly

**Performance Dashboard** (Tab 3: Analytics)
- Track query count & response times
- Monitor cache effectiveness
- View recent errors

---

## Performance Tips

1. **Cache Hits** - Ask similar questions to benefit from search caching
2. **Batch Uploads** - Upload all policies at once before asking questions
3. **Concise Queries** - Shorter questions process faster
4. **Monitor Costs** - Check Analytics tab to track API usage

---

## Troubleshooting

### API Key Error
- ✓ Verify API key in **Secrets** tab
- ✓ Get free key at https://aistudio.google.com/apikey
- ✓ Key must be 20+ characters

### Database Connection Error
- ✓ Check PostgreSQL is running (it should auto-start)
- ✓ Verify DATABASE_URL in Secrets
- ✓ Replit manages DB automatically

### Search Not Finding Results
- ✓ Ensure documents are uploaded first
- ✓ Try more general query terms
- ✓ Check document file format (PDF/DOCX supported)

### Slow Performance
- ✓ Check cache hit rate in Analytics tab
- ✓ Monitor response times
- ✓ Try similar queries for cache benefits

---

## Deployment

### Deploy to Production (Publish)
1. Click **Publish** in Replit
2. Get live URL for your app
3. App auto-scales and is always available
4. Your database persists across deployments

### Environment Variables in Production
- All secrets automatically copied to production
- No additional setup needed
- Database connection handled by Replit

---

## API Limits & Costs

### Google Gemini API
- Free tier: 60 requests/minute
- See pricing: https://ai.google.dev/pricing

### FAISS Embeddings
- Unlimited local embeddings (no API cost)
- Fast vector search on your machine

### PostgreSQL
- Included with Replit
- No additional costs

---

## Architecture Overview

```
User Query
    ↓
Streamlit UI (app.py)
    ↓
MCP Data Server (mcp_server.py)
    ↓
Hybrid Search Engine (hybrid_search.py)
├── FAISS Semantic Search (70% weight)
└── BM25 Keyword Search (30% weight)
    ↓
Agentic Orchestrator (agentic_orchestrator.py)
├── Plan Node - Decide search strategy
├── Search Node - Execute with caching
├── Verify Node - Check if sufficient info
├── Answer Node - Synthesize response
    ↓
Google Gemini 2.0 Flash
    ↓
Performance Monitor (performance_monitor.py)
├── Query metrics
├── Error tracking
├── Cache statistics
    ↓
PostgreSQL Database (database.py)
    ↓
Response with Citations
```

---

## Next Steps

1. **Set your API key** in Secrets
2. **Upload test policies**
3. **Ask questions** and watch the agent think
4. **Monitor performance** in Analytics tab
5. **Deploy when ready** using Publish

---

**Questions?** Check the in-app "About This System" section for more details!

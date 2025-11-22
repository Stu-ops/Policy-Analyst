# 🤖 Agentic RAG Policy Analyst Platform

A next-generation Retrieval-Augmented Generation (RAG) system designed to interpret complex insurance policies using autonomous AI agents.

## 🎯 Overview

Unlike traditional RAG chatbots that follow a simple "retrieve-then-answer" loop, this platform uses an **Agentic Workflow** powered by LangGraph. The agent autonomously plans its reasoning steps, querying a standardized data layer to retrieve specific policy clauses only when needed.

## 🏗️ Architecture

### The Brain (Agentic Orchestrator)
- **Core**: LangGraph with Stateful ReAct Agent pattern
- **Model**: Google Gemini 2.0 Flash with "Deep Thinking" capabilities
- **Function**: Autonomous decision-making and complex query decomposition

### The Body (MCP Data Server)
- **Protocol**: Model Context Protocol (MCP) inspired architecture
- **RAG Layer**: Hybrid Search Engine (FAISS + BM25)
- **Data Handling**: Decoupled document parsing (PDF/DOCX/Excel)

## ✨ Key Features

### 1. Agentic RAG with Deep Thinking
- Leverages Gemini 2.0's reasoning blocks for ambiguous queries
- Cross-references contradictory policy clauses
- Autonomous planning and tool selection

### 2. Hybrid Search Engine
- **Dense Retrieval**: Gemini text-embedding-004 (768-dim) stored in FAISS
- **Sparse Retrieval**: BM25 for exact keyword matching (policy IDs, specific terms)
- **Optimized Parameters**: k=14 dense, k=9 sparse, final k=5
- **Weighting**: 70% Semantic / 30% Keyword

### 3. Standardized Data Layer
- MCP-inspired architecture isolates document processing
- Easily swappable components
- Support for PDF, DOCX, Excel, and TXT files

### 4. Interactive Web Interface
- Document upload and management
- Real-time chat with policy analysis
- View retrieved policy sections with relevance scores
- Citation tracking with source and page numbers

## 🚀 Getting Started

> **📌 Note**: This is a Streamlit web application. GitHub Pages (https://stu-ops.github.io/Policy-Analyst/) shows a static landing page with instructions. To run the actual application, see the deployment options below.

### Prerequisites
- Python 3.11+
- Google Gemini API Key (get one at https://aistudio.google.com/apikey)

### Quick Start (Local)

```bash
# Clone the repository
git clone https://github.com/Stu-ops/Policy-Analyst.git
cd Policy-Analyst

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

Then:
1. Open http://localhost:8501 in your browser
2. Enter your `GEMINI_API_KEY` in the sidebar when prompted
3. Upload policy documents and start chatting!

### Deployment Options

For production deployment, see **[DEPLOYMENT.md](DEPLOYMENT.md)** for detailed instructions on:
- 🌐 **Streamlit Cloud** (Recommended - Free & Easy)
- 🤗 **Hugging Face Spaces**
- 🚂 **Railway**
- 🎨 **Render**
- And more...

### Using the Platform

1. **Upload Documents**: Use the file uploader to add insurance policy documents (PDF, DOCX, Excel, or TXT)
2. **Wait for Processing**: The system will chunk and index your documents
3. **Ask Questions**: Use natural language to query your policies
4. **View Results**: See AI-generated answers with supporting evidence from policy documents

## 📋 Example Queries

- "What are the coverage limits for property damage?"
- "Explain the deductible policy for medical claims"
- "What exclusions apply to flood damage?"
- "Compare the premium calculation methods across policies"
- "Are pre-existing conditions covered under this health policy?"
- "What is the claims process timeline for auto insurance?"

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **AI Orchestration**: LangGraph
- **LLM**: Google Gemini 2.5 Flash (Thinking)
- **Vector Search**: FAISS
- **Keyword Search**: BM25 (rank-bm25)
- **Document Processing**: PyPDF2, python-docx, openpyxl
- **Embeddings**: Google text-embedding-004

## 📁 Project Structure

```
.
├── app.py                      # Main Streamlit application
├── actuarial_tools.py          # Actuarial calculation tools
├── agentic_orchestrator.py     # LangGraph ReAct agent
├── contradiction_detection.py  # Contradiction detection logic
├── database.py                 # Database interactions
├── document_processor.py       # Multi-format document parser
├── hybrid_search.py            # FAISS + BM25 search engine
├── mcp_server.py               # MCP-style data server
├── performance_monitor.py      # Performance monitoring
├── sample_documents/           # Sample insurance policies
│   ├── auto_insurance_policy.txt
│   ├── health_insurance_policy.txt
│   └── home_insurance_policy.txt
└── README.md
```

## 🎓 How It Works

### 1. Document Processing
Documents are chunked into ~500-word segments with 50-word overlap for context preservation.

### 2. Indexing
Each chunk is:
- Embedded using Gemini text-embedding-004 (768 dimensions)
- Indexed in FAISS for semantic similarity search
- Tokenized and indexed in BM25 for keyword matching

### 3. Query Processing
When you ask a question:
1. LangGraph agent analyzes the query
2. Agent decides if it needs to search documents
3. Hybrid search retrieves top-k relevant chunks
4. Agent synthesizes answer using retrieved context
5. Response includes citations and source references

### 4. Agentic Reasoning
The ReAct pattern allows the agent to:
- Think through complex multi-step queries
- Search multiple times if needed
- Cross-reference different policy sections
- Provide nuanced answers to ambiguous questions

## 🔧 Configuration

Current optimized settings:
- **Semantic Weight**: 0.7 (70%)
- **Keyword Weight**: 0.3 (30%)
- **Dense Retrieval**: k=14
- **Sparse Retrieval**: k=9
- **Final Results**: k=5
- **Chunk Size**: 500 words
- **Chunk Overlap**: 50 words

## 🌟 Advanced Features

### Model Context Protocol (MCP)
The data layer follows MCP principles:
- **Tools**: Search functionality exposed as executable tools
- **Resources**: Documents served as standardized resources
- **Isolation**: Document parsing logic is decoupled from the agent

### ReAct Pattern
The agent follows a Reasoning + Acting loop:
1. **Reason**: Analyze the query and plan next steps
2. **Act**: Execute search tools when needed
3. **Observe**: Review search results
4. **Repeat**: Continue until sufficient information is gathered
5. **Answer**: Synthesize final response

## 📊 Performance

- Supports documents up to thousands of pages
- Sub-second search latency
- Handles ambiguous and complex queries
- Cross-references multiple policy documents simultaneously

## 🔐 Security

- API keys stored securely in environment variables
- No data persistence (session-based only)
- Documents processed in memory only

## 🤝 Contributing

This is a demonstration platform showcasing agentic RAG architecture. Feel free to extend it with:
- Additional document formats
- Database persistence
- Multi-user support
- Actuarial calculation tools
- Policy comparison features

## 📄 License

This project is for educational and demonstration purposes.

## 🙏 Acknowledgments

Built with:
- Google Gemini AI
- LangChain & LangGraph
- Meta's FAISS
- The open-source community

---

**Note**: This system is designed for insurance policy analysis but can be adapted for any domain requiring intelligent document retrieval and analysis.

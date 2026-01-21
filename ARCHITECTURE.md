# Policy Analyst - Software Architecture Documentation

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Architecture Layers](#architecture-layers)
4. [Core Components](#core-components)
5. [File-by-File Functionality](#file-by-file-functionality)
6. [Data Flow](#data-flow)
7. [Design Patterns](#design-patterns)
8. [Technology Stack](#technology-stack)

---

## Overview

The Policy Analyst is a next-generation Retrieval-Augmented Generation (RAG) system designed to interpret complex insurance policies using autonomous AI agents. Unlike traditional RAG chatbots that follow a simple "retrieve-then-answer" loop, this platform uses an **Agentic Workflow** powered by LangGraph where the agent autonomously plans its reasoning steps.

### Key Innovation
- **Agentic RAG**: Autonomous decision-making with multi-turn reasoning
- **Hybrid Search**: Combines semantic (FAISS) and keyword (BM25) search
- **MCP-Inspired Architecture**: Decoupled, standardized data layer
- **Deep Thinking**: Leverages Gemini 2.0's advanced reasoning capabilities

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│                    (Streamlit Web App)                          │
│                         app.py                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                     ORCHESTRATION LAYER                         │
│              (Agentic ReAct Pattern - LangGraph)                │
│                  agentic_orchestrator.py                        │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────┐ │
│  │   Plan     │→ │   Search   │→ │   Verify   │→ │  Answer  │ │
│  │    Node    │  │    Node    │  │    Node    │  │   Node   │ │
│  └────────────┘  └────────────┘  └────────────┘  └──────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER (MCP)                         │
│                        mcp_server.py                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Resource Management                          │  │
│  │  • Document Storage  • Metadata Tracking                 │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                ┌────────────┴────────────┐
                ↓                         ↓
┌───────────────────────────┐  ┌──────────────────────────────┐
│  DOCUMENT PROCESSING      │  │     HYBRID SEARCH ENGINE     │
│  document_processor.py    │  │     hybrid_search.py         │
│  ┌────────────────────┐   │  │  ┌────────────────────────┐  │
│  │ PDF Processing     │   │  │  │ FAISS (Semantic)       │  │
│  │ DOCX Processing    │   │  │  │ • Vector Embeddings    │  │
│  │ Excel Processing   │   │  │  │ • L2 Distance Search   │  │
│  │ TXT Processing     │   │  │  └────────────────────────┘  │
│  │ Text Chunking      │   │  │  ┌────────────────────────┐  │
│  └────────────────────┘   │  │  │ BM25 (Keyword)         │  │
└───────────────────────────┘  │  │ • Token-based Search   │  │
                               │  │ • Exact Match          │  │
                               │  └────────────────────────┘  │
                               └──────────────────────────────┘
                                           │
                ┌──────────────────────────┴──────────────┐
                ↓                                         ↓
┌───────────────────────────┐             ┌───────────────────────┐
│  SUPPORT SERVICES         │             │  AI SERVICES          │
│                           │             │                       │
│  • Database (PostgreSQL)  │             │  • Google Gemini 2.0  │
│    database.py            │             │    (Deep Thinking)    │
│                           │             │  • Text Embeddings    │
│  • Performance Monitor    │             │    (768-dim vectors)  │
│    performance_monitor.py │             │                       │
│                           │             │                       │
│  • Actuarial Calculator   │             │                       │
│    actuarial_tools.py     │             └───────────────────────┘
│                           │
│  • Contradiction Detector │
│    contradiction_detector.py
│                           │
└───────────────────────────┘
```

---

## Architecture Layers

### 1. **Presentation Layer**
- **Component**: `app.py`
- **Purpose**: User interface and interaction management
- **Technology**: Streamlit web framework
- **Responsibilities**:
  - Document upload handling
  - Chat interface management
  - Results visualization
  - Session state management
  - API key validation

### 2. **Orchestration Layer**
- **Component**: `agentic_orchestrator.py`
- **Purpose**: Autonomous agent workflow management
- **Technology**: LangGraph + Google Gemini
- **Responsibilities**:
  - Multi-turn reasoning (ReAct pattern)
  - Query planning and decomposition
  - Search execution and verification
  - Answer synthesis
  - Iterative refinement

### 3. **Data Access Layer**
- **Component**: `mcp_server.py`
- **Purpose**: Standardized data abstraction
- **Technology**: MCP-inspired architecture
- **Responsibilities**:
  - Resource management
  - Document indexing
  - Search tool execution
  - Statistics tracking

### 4. **Processing Layer**
- **Components**: 
  - `document_processor.py` - Document parsing
  - `hybrid_search.py` - Search engine
- **Purpose**: Document processing and retrieval
- **Responsibilities**:
  - Multi-format document parsing
  - Text chunking with overlap
  - Semantic embedding generation
  - Hybrid search execution
  - Result ranking and fusion

### 5. **Support Layer**
- **Components**: 
  - `database.py` - Persistence
  - `performance_monitor.py` - Metrics
  - `actuarial_tools.py` - Domain tools
  - `contradiction_detector.py` - Analysis
- **Purpose**: Supporting services and utilities
- **Responsibilities**:
  - Data persistence
  - Performance tracking
  - Domain-specific calculations
  - Quality assurance

---

## Core Components

### 1. Agentic Orchestrator (Brain)
**File**: `agentic_orchestrator.py`

The orchestrator implements a sophisticated **ReAct (Reasoning + Acting)** pattern using LangGraph:

#### State Graph Nodes:
- **Plan Node**: Determines what information is needed
- **Search Node**: Executes searches based on the plan
- **Verify Node**: Checks if information is sufficient
- **Answer Node**: Synthesizes final response

#### Key Features:
- Multi-turn reasoning with up to 5 iterations
- Automatic query refinement
- Context accumulation across searches
- Error handling and fallback mechanisms
- Performance metrics tracking

### 2. Hybrid Search Engine (Body)
**File**: `hybrid_search.py`

Combines two complementary search approaches:

#### Semantic Search (FAISS):
- **Embeddings**: Google text-embedding-004 (768 dimensions)
- **Index Type**: FAISS IndexFlatL2
- **Distance Metric**: L2 (Euclidean distance)
- **Default k**: 14 results

#### Keyword Search (BM25):
- **Algorithm**: BM25 (Best Match 25)
- **Tokenization**: Lowercase, whitespace split
- **Default k**: 9 results

#### Fusion Strategy:
- **Semantic Weight**: 70%
- **Keyword Weight**: 30%
- **Final Results**: Top 5 combined
- **Caching**: Query-level result caching

### 3. MCP Data Server (Nervous System)
**File**: `mcp_server.py`

Inspired by Model Context Protocol (MCP), provides standardized data access:

#### Responsibilities:
- **Resource Management**: Add, retrieve, list, clear documents
- **Tool Execution**: Standardized search interface
- **Statistics**: Track usage and performance
- **Abstraction**: Decouples document processing from agents

### 4. Document Processor (Digestive System)
**File**: `document_processor.py`

Multi-format document ingestion and processing:

#### Supported Formats:
- **PDF**: PyPDF2-based extraction with page tracking
- **DOCX**: python-docx for Word documents
- **Excel**: openpyxl for spreadsheets (multi-sheet support)
- **TXT**: Plain text files

#### Processing Pipeline:
1. Format detection
2. Text extraction
3. Chunking (500 words, 50-word overlap)
4. Metadata enrichment (source, page, chunk index)

---

## File-by-File Functionality

### Core Application Files

#### `app.py` - Main Application Entry Point
**Purpose**: Streamlit web application and UI orchestration

**Key Functions**:
- `initialize_minimal_state()`: Sets up session state without API key
- `initialize_with_api_key(api_key)`: Initializes MCP server and orchestrator
- `setup_api_key_sidebar()`: Manages API key validation UI
- `process_uploaded_file(uploaded_file)`: Handles document upload
- `main()`: Main application loop

**Features**:
- Document upload and management
- Real-time chat interface
- Reasoning step visualization
- Search results display with citations
- Performance analytics dashboard
- Actuarial calculation tools
- Contradiction detection
- Session state management

**UI Sections**:
1. **Document Management Panel**: Upload, view stats, clear documents
2. **Chat Interface**: Query input and response display
3. **Advanced Tools Sidebar**:
   - Configuration display
   - Actuarial calculator
   - Performance analytics
   - Contradiction detection

---

#### `agentic_orchestrator.py` - AI Agent Orchestration
**Purpose**: Multi-turn ReAct agent using LangGraph

**Key Classes**:
- `AgentState(TypedDict)`: State container for agent workflow
- `AgenticOrchestrator`: Main orchestration class

**Key Methods**:
- `_create_graph()`: Builds LangGraph state machine
- `_plan_node(state)`: Plans search strategy
- `_search_node(state)`: Executes searches
- `_verify_node(state)`: Validates information sufficiency
- `_answer_node(state)`: Synthesizes final answer
- `_decide_next_action(state)`: Routing logic
- `run(query)`: Main execution method
- `stream_run(query)`: Streaming execution

**Workflow**:
```
START → Plan → Search → Verify
                  ↑        ↓
                  └───┬────┘
                      ↓
                   Answer → END
```

**Features**:
- Up to 5 reasoning iterations
- Context accumulation across turns
- Performance monitoring
- Error recovery
- Citation tracking
- Reasoning step logging

---

#### `mcp_server.py` - Model Context Protocol Server
**Purpose**: Standardized data layer following MCP principles

**Key Class**: `MCPDataServer`

**Key Methods**:
- `add_resource(file_path, resource_id)`: Add document to system
- `add_documents_from_uploaded(documents, metadata)`: Bulk document addition
- `execute_search_tool(query, k)`: Execute search operation
- `get_resource(resource_id)`: Retrieve specific resource
- `list_resources()`: List all loaded resources
- `clear_all()`: Clear all documents and indexes
- `get_statistics()`: Get system statistics

**Architecture Benefits**:
- **Decoupling**: Separates data logic from agent logic
- **Standardization**: Consistent interface for data operations
- **Testability**: Easy to mock for testing
- **Extensibility**: Simple to add new data sources

---

#### `document_processor.py` - Document Processing Pipeline
**Purpose**: Multi-format document parsing and chunking

**Key Class**: `DocumentProcessor`

**Supported Formats**:
1. **PDF** (`_process_pdf`):
   - Uses PyPDF2.PdfReader
   - Extracts text page-by-page
   - Tracks page numbers in metadata
   - Handles extraction errors gracefully

2. **DOCX** (`_process_docx`):
   - Uses python-docx Document
   - Extracts paragraphs
   - Preserves document structure

3. **Excel** (`_process_excel`):
   - Uses pandas read_excel
   - Processes all sheets
   - Converts tables to text
   - Tracks sheet names

4. **TXT** (`_process_txt`):
   - Direct UTF-8 file reading
   - Simple text processing

**Text Chunking** (`_chunk_text`):
- **Chunk Size**: 500 words (configurable)
- **Overlap**: 50 words (configurable)
- **Purpose**: Maintain context while enabling efficient retrieval
- **Strategy**: Sliding window with word-level splitting

**Output Format**:
```python
{
    'chunks': List[str],           # Text chunks
    'metadata': List[Dict],        # Metadata per chunk
    'source': str,                 # Original filename
    'num_pages': int               # Page count (PDF)
}
```

---

#### `hybrid_search.py` - Hybrid Search Engine
**Purpose**: Combine semantic and keyword search for optimal retrieval

**Key Class**: `HybridSearchEngine`

**Components**:

1. **Semantic Search** (FAISS):
   - **Model**: Google text-embedding-004
   - **Dimensions**: 768
   - **Index**: FAISS IndexFlatL2 (exact L2 distance)
   - **Normalization**: Distance → Similarity score (0-1)

2. **Keyword Search** (BM25):
   - **Algorithm**: BM25Okapi (rank-bm25 library)
   - **Tokenization**: Lowercase, whitespace split
   - **Scoring**: TF-IDF variant with tuned parameters

3. **Result Fusion**:
   - Weighted score combination
   - De-duplication by document index
   - Top-k re-ranking

**Key Methods**:
- `add_documents(documents, metadata)`: Index new documents
- `_get_embeddings(texts)`: Generate embeddings via Gemini
- `_build_faiss_index()`: Create FAISS index
- `_build_bm25_index()`: Create BM25 index
- `search(query, k_dense, k_sparse, final_k)`: Execute hybrid search
- `_semantic_search(query, k)`: FAISS search
- `_keyword_search(query, k)`: BM25 search
- `_combine_results(dense, sparse, final_k)`: Fusion logic

**Caching**:
- Query-level caching with MD5 hashing
- Cache hit/miss tracking
- Statistics reporting

**Optimization**:
- Batch embedding generation
- Efficient numpy operations
- In-memory indexes for speed

---

### Support Files

#### `database.py` - PostgreSQL Database Manager
**Purpose**: Persistent storage for documents, chat history, and analytics

**Key Class**: `DatabaseManager`

**Database Schema**:

1. **documents**: Stores uploaded documents
   - Fields: id, filename, file_type, chunks_count, created_at, updated_at, version

2. **document_chunks**: Stores individual document chunks
   - Fields: id, document_id, chunk_index, content, page_number, embedding_vector, created_at

3. **chat_history**: Stores user conversations
   - Fields: id, session_id, query_text, response_text, search_results, created_at, tokens_used

4. **analytics**: Tracks query performance
   - Fields: id, session_id, query_type, num_documents_searched, response_time_ms, relevance_score, created_at

5. **contradictions**: Flags policy contradictions
   - Fields: id, document1_id, document2_id, clause1_text, clause2_text, severity, description, flagged_at

**Key Methods**:
- `initialize_schema()`: Create all tables
- `add_document(filename, file_type)`: Add document record
- `add_chunks(document_id, chunks)`: Store chunks
- `add_chat_message(session_id, query, response, search_results)`: Log chat
- `get_chat_history(session_id, limit)`: Retrieve chat history
- `add_analytics(...)`: Log performance metrics
- `flag_contradiction(...)`: Store detected contradiction
- `get_contradictions()`: Retrieve flagged issues

**Connection Management**:
- Uses psycopg2 with connection pooling
- Environment variable configuration (DATABASE_URL)
- Automatic connection cleanup

---

#### `actuarial_tools.py` - Actuarial Calculation Tools
**Purpose**: Domain-specific insurance calculations and analysis

**Key Classes**:
- `PremiumCalculation`: Data class for premium results
- `RiskLevel(Enum)`: Risk level enumeration (LOW, MEDIUM, HIGH, VERY_HIGH)
- `ActuarialCalculator`: Main calculator class

**Key Methods**:

1. **Premium Calculation** (`calculate_premium`):
   - Base rates by policy type (auto, health, home, life)
   - Risk factor multipliers
   - Annual and monthly premium calculation
   - Detailed cost breakdown

2. **Deductible Analysis** (`calculate_deductible_impact`):
   - Effective coverage calculation
   - Coverage percentage
   - Out-of-pocket maximum

3. **Claim Payout** (`calculate_claim_payout`):
   - Apply deductible
   - Calculate coinsurance
   - Determine insurer vs. member cost split

4. **Loss Frequency** (`estimate_loss_frequency`):
   - Probability of claims over time
   - Expected number of claims
   - Risk assessment

5. **NPV Analysis** (`calculate_net_present_value`):
   - Present value of premiums
   - Present value of claims
   - Loss ratio calculation
   - Combined ratio with expenses

6. **Policy Text Extraction** (`extract_policy_numbers`):
   - Regex-based extraction of:
     - Coverage limits
     - Deductibles
     - Premiums
     - Percentages

**Base Rates**:
```python
{
    'auto': $1,500/year,
    'health': $450/month,
    'home': $1,200/year,
    'life': $50/month
}
```

---

#### `contradiction_detector.py` - Policy Contradiction Detection
**Purpose**: Identify conflicting clauses across policy documents

**Key Classes**:
- `ContradictionSeverity(Enum)`: LOW, MEDIUM, HIGH, CRITICAL
- `ContradictionDetector`: Main detection class

**Detection Strategy**:

1. **Keyword-Based Extraction**:
   - Coverage-related terms
   - Limit/cap language
   - Exclusion phrases
   - Condition statements
   - Frequency terms
   - Numeric amounts (regex)

2. **Clause Comparison**:
   - Negation detection (not, no, never, exclude)
   - Word overlap analysis
   - Sentence similarity scoring

3. **Severity Assessment**:
   - **CRITICAL**: Conflicts in exclusions/limits with numbers
   - **HIGH**: Coverage or condition conflicts
   - **MEDIUM**: Frequency or amount discrepancies
   - **LOW**: Other conflicts

**Key Methods**:
- `detect_contradictions(documents)`: Main detection pipeline
- `_compare_documents(doc1, doc2)`: Pairwise comparison
- `_extract_clauses(text, keywords)`: Clause extraction
- `_is_contradictory(clause1, clause2)`: Contradiction test
- `_assess_severity(...)`: Severity calculation
- `cross_reference_clauses(documents, query)`: Find related clauses

**Use Cases**:
- Policy audit and compliance
- Multi-document comparison
- Quality assurance
- Risk identification

---

#### `performance_monitor.py` - Performance Metrics Tracking
**Purpose**: Monitor system performance and resource usage

**Key Class**: `PerformanceMonitor`

**Tracked Metrics**:

1. **Query Metrics**:
   - Total query count
   - Response times (avg, min, max)
   - Iteration counts
   - Success/failure rates

2. **Search Metrics**:
   - Search execution times
   - Number of results returned
   - Cache hit/miss rates

3. **Error Metrics**:
   - Error count and types
   - Error messages and context
   - Timestamps

4. **System Metrics**:
   - Session uptime
   - Total processing time
   - Cache efficiency

**Key Methods**:
- `record_query(query, response_time, iterations, num_results, success)`: Log query
- `record_search(query, num_results, search_time, cached)`: Log search
- `record_error(error_type, error_msg, context)`: Log error
- `record_embedding(num_documents, embedding_time)`: Log embedding
- `get_statistics()`: Get comprehensive stats
- `get_recent_errors(limit)`: Get recent errors
- `get_recent_queries(limit)`: Get recent queries

**Statistics Output**:
```python
{
    "query_count": int,
    "error_count": int,
    "error_rate": float,
    "avg_response_time": float,
    "cache_hit_rate": float,
    "uptime_seconds": float,
    ...
}
```

---

### Configuration Files

#### `requirements.txt` - Python Dependencies
**Purpose**: Define all project dependencies

**Key Dependencies**:
- **Web Framework**: streamlit
- **AI/ML**: google-genai, langchain, langgraph, langchain-google-genai
- **Vector Search**: faiss-cpu, numpy
- **Keyword Search**: rank-bm25
- **Document Processing**: pypdf2, python-docx, openpyxl
- **Data**: pandas
- **Database**: psycopg2-binary
- **Utilities**: sift-stack-py

#### `pyproject.toml` - Project Configuration
**Purpose**: Python project metadata and build configuration

---

## Data Flow

### 1. Document Upload Flow
```
User Uploads Document (PDF/DOCX/Excel/TXT)
              ↓
    app.py: process_uploaded_file()
              ↓
    DocumentProcessor.process_file()
              ↓
    Extract text + Chunk (500 words, 50 overlap)
              ↓
    MCPDataServer.add_resource()
              ↓
    HybridSearchEngine.add_documents()
              ↓
    [Parallel Processing]
    ├─→ Generate embeddings (Gemini API)
    │   └─→ Build FAISS index
    └─→ Tokenize documents
        └─→ Build BM25 index
              ↓
    Document Ready for Search
```

### 2. Query Processing Flow
```
User Enters Question
        ↓
app.py validates API key & documents
        ↓
AgenticOrchestrator.run(query)
        ↓
┌───────────────────────────────┐
│  ReAct Loop (up to 5 turns)  │
│                               │
│  1. PLAN NODE                 │
│     ├─→ Analyze query         │
│     └─→ Determine strategy    │
│           ↓                   │
│  2. SEARCH NODE               │
│     ├─→ MCPDataServer.search()│
│     │   ├─→ Hybrid Search     │
│     │   │   ├─→ FAISS (k=14) │
│     │   │   └─→ BM25 (k=9)   │
│     │   └─→ Combine (k=5)    │
│     └─→ Return results        │
│           ↓                   │
│  3. VERIFY NODE               │
│     ├─→ Check sufficiency     │
│     └─→ Decide: more searches?│
│           ↓                   │
│     [If yes: loop to SEARCH]  │
│     [If no: proceed]          │
│           ↓                   │
│  4. ANSWER NODE               │
│     ├─→ Compile context       │
│     ├─→ Call Gemini 2.0      │
│     └─→ Synthesize answer    │
└───────────────────────────────┘
        ↓
Return response with:
- Final answer
- Search results (with citations)
- Reasoning steps
- Performance metrics
        ↓
app.py displays to user
```

### 3. Search Execution Flow
```
Search Query
     ↓
Check Cache (MD5 hash)
     ├─→ [Hit] Return cached results
     └─→ [Miss] Execute search
            ↓
Parallel Search Execution:
     ├─────────────────┬─────────────────┐
     ↓                 ↓                 ↓
SEMANTIC (70%)   KEYWORD (30%)    COMBINE
     ↓                 ↓                 ↓
Embed query      Tokenize query   Merge scores
     ↓                 ↓                 ↓
FAISS search     BM25 search      Deduplicate
k=14 results     k=9 results      Top k=5
     ↓                 ↓                 ↓
Score → [0,1]    Score → [0,1]    Final ranking
     │                 │                 │
     └─────────────────┴─────────────────┘
                       ↓
              Cache results
                       ↓
              Return to caller
```

---

## Design Patterns

### 1. **ReAct Pattern (Orchestration)**
**Location**: `agentic_orchestrator.py`

The ReAct (Reasoning + Acting) pattern enables autonomous decision-making:

```
Reason → Act → Observe → Repeat
```

**Benefits**:
- Autonomous multi-step problem solving
- Self-correcting behavior
- Transparency through reasoning traces
- Flexible iteration control

### 2. **Model Context Protocol (MCP)**
**Location**: `mcp_server.py`

Inspired by Anthropic's MCP, provides standardized data access:

**Principles**:
- **Tools**: Expose operations as callable tools
- **Resources**: Standardize resource representation
- **Prompts**: Reusable prompt templates
- **Separation**: Decouple data logic from AI logic

**Benefits**:
- Easy testing and mocking
- Swappable implementations
- Clear contracts between layers

### 3. **Strategy Pattern (Search)**
**Location**: `hybrid_search.py`

Multiple search strategies combined dynamically:

```python
class HybridSearch:
    semantic_search()   # Strategy 1
    keyword_search()    # Strategy 2
    combine_results()   # Fusion strategy
```

**Benefits**:
- Flexible search algorithm composition
- Easy to add new search methods
- Configurable weighting

### 4. **Template Method Pattern (Document Processing)**
**Location**: `document_processor.py`

Abstract processing pipeline with format-specific implementations:

```python
process_file(path):
    detect_format()
    extract_text()      # Format-specific
    chunk_text()        # Common
    enrich_metadata()   # Common
```

**Benefits**:
- Code reuse across formats
- Easy to add new formats
- Consistent processing pipeline

### 5. **Observer Pattern (Performance Monitoring)**
**Location**: `performance_monitor.py`

Event-driven metrics collection:

```python
orchestrator.run()
    ├─→ monitor.record_query()
    ├─→ monitor.record_search()
    └─→ monitor.record_error()
```

**Benefits**:
- Decoupled monitoring from business logic
- Centralized metrics
- Easy analytics

### 6. **Factory Pattern (Document Processing)**
**Location**: `document_processor.py`

Dynamic processor selection based on file type:

```python
if extension == '.pdf':
    return _process_pdf()
elif extension == '.docx':
    return _process_docx()
```

### 7. **State Pattern (LangGraph)**
**Location**: `agentic_orchestrator.py`

State machine for agent workflow:

```python
StateGraph(AgentState)
    .add_node("plan", plan_node)
    .add_node("search", search_node)
    .add_edge("plan", "search")
```

---

## Technology Stack

### Frontend
- **Streamlit**: Web application framework
  - Real-time updates
  - Session state management
  - Component-based UI

### AI/ML Stack
- **Google Gemini 2.0 Flash (Thinking)**: LLM with deep reasoning
  - Model: `gemini-2.0-flash-thinking-exp-01-21`
  - Features: Extended reasoning blocks, function calling
  
- **Google text-embedding-004**: Text embeddings
  - Dimensions: 768
  - Task: retrieval_document

- **LangGraph**: Agent orchestration framework
  - State machines for workflows
  - Multi-turn conversations
  - Tool integration

- **LangChain**: LLM application framework
  - Message abstractions
  - Tool/function calling
  - Integration layer

### Search & Retrieval
- **FAISS**: Vector similarity search
  - Library: faiss-cpu
  - Index: IndexFlatL2 (exact search)
  - Optimized for CPU

- **BM25**: Keyword search
  - Library: rank-bm25
  - Algorithm: BM25Okapi
  - TF-IDF based ranking

### Document Processing
- **PyPDF2**: PDF text extraction
- **python-docx**: Word document parsing
- **openpyxl**: Excel file processing
- **pandas**: Data manipulation

### Database
- **PostgreSQL**: Relational database
  - Connection: psycopg2-binary
  - Features: JSONB support for search results

### Utilities
- **NumPy**: Numerical operations
- **sift-stack-py**: Stack management utilities

---

## Key Optimizations

### 1. **Hybrid Search Tuning**
Optimized through experimentation:
- **Semantic Weight**: 70% (captures meaning)
- **Keyword Weight**: 30% (captures exact terms)
- **Dense k**: 14 (broader semantic search)
- **Sparse k**: 9 (focused keyword search)
- **Final k**: 5 (top results to LLM)

### 2. **Text Chunking Strategy**
- **Chunk Size**: 500 words (optimal for context)
- **Overlap**: 50 words (10% - preserves continuity)
- **Benefits**: 
  - Prevents context fragmentation
  - Improves answer quality
  - Enables precise citations

### 3. **Caching**
- **Query Caching**: MD5-hashed queries
- **Embedding Reuse**: Documents embedded once
- **Index Persistence**: In-memory for speed

### 4. **Error Recovery**
- Graceful degradation in all components
- Try-except blocks with fallbacks
- Error logging for debugging

---

## Security Considerations

### 1. **API Key Management**
- Stored only in session memory
- Never persisted to disk
- User-provided per session
- Validation before use

### 2. **Database Security**
- Environment variable configuration
- Connection string protection
- Parameterized queries (SQL injection prevention)

### 3. **File Processing**
- Temporary file cleanup
- Size limits (implicit in Streamlit)
- Format validation before processing

---

## Future Enhancement Opportunities

### 1. **Scalability**
- Move to async/await for I/O operations
- Distributed search with Elasticsearch
- Redis for query caching
- Background job processing

### 2. **Features**
- Multi-user authentication
- Document versioning
- Collaborative annotations
- Real-time collaboration
- Custom actuarial models

### 3. **Performance**
- GPU-accelerated FAISS (faiss-gpu)
- Approximate nearest neighbors (FAISS IVF)
- Batch embedding generation
- Connection pooling for database

### 4. **Observability**
- Structured logging (JSON logs)
- Distributed tracing (OpenTelemetry)
- Real-time dashboards (Grafana)
- Alerting (PagerDuty/Slack)

---

## Conclusion

This architecture demonstrates a modern approach to building intelligent document analysis systems. Key strengths include:

1. **Autonomous Intelligence**: ReAct pattern enables self-directed reasoning
2. **Hybrid Retrieval**: Combines semantic understanding with keyword precision
3. **Modularity**: Clean separation of concerns across layers
4. **Extensibility**: Easy to add new document formats, search methods, or tools
5. **Observability**: Built-in monitoring and analytics
6. **Domain-Specific**: Tailored for insurance policy analysis

The system is production-ready for insurance policy analysis but designed to be adaptable to other document-intensive domains.

---

## Quick Reference

### Key Parameters
```python
# Search Configuration
SEMANTIC_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3
DENSE_K = 14
SPARSE_K = 9
FINAL_K = 5

# Processing Configuration
CHUNK_SIZE = 500  # words
CHUNK_OVERLAP = 50  # words

# Agent Configuration
MAX_ITERATIONS = 5

# Embedding Configuration
EMBEDDING_MODEL = "text-embedding-004"
EMBEDDING_DIM = 768

# LLM Configuration
LLM_MODEL = "gemini-2.0-flash-thinking-exp-01-21"
```

### Main Entry Points
```python
# Application start
app.py → main()

# Document upload
app.py → process_uploaded_file()

# Query processing
agentic_orchestrator.py → AgenticOrchestrator.run()

# Search execution
mcp_server.py → MCPDataServer.execute_search_tool()
```

---

**Document Version**: 1.0  
**Last Updated**: 2026-01-21  
**Architecture Status**: Production-Ready  
**Maintained By**: Development Team

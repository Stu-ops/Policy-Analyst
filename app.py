import streamlit as st
import os
from pathlib import Path
import tempfile
import uuid
from mcp_server import MCPDataServer
from agentic_orchestrator import AgenticOrchestrator
from database import DatabaseManager
from actuarial_tools import ActuarialCalculator
from contradiction_detector import ContradictionDetector

# Try to load .env file for local development (optional)
try:
    from dotenv import load_dotenv
    env_path = Path('.env')
    if env_path.exists():
        load_dotenv()
except ImportError:
    pass


st.set_page_config(
    page_title="Agentic RAG Policy Analyst",
    page_icon="🤖",
    layout="wide"
)


def validate_api_key(api_key: str) -> bool:
    """Validate Gemini API key format"""
    if not api_key or not isinstance(api_key, str):
        return False
    api_key = api_key.strip()
    if len(api_key) < 20:
        return False
    return True


def initialize_minimal_state():
    """Initialize basic session state without API key"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if 'user_api_key' not in st.session_state:
        st.session_state.user_api_key = None
    if 'api_key_validated' not in st.session_state:
        st.session_state.api_key_validated = False
    
    if 'mcp_server' not in st.session_state:
        st.session_state.mcp_server = None
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = None
    
    if 'db' not in st.session_state:
        try:
            st.session_state.db = DatabaseManager()
            st.session_state.db.initialize_schema()
        except Exception as e:
            st.session_state.db = None
    
    if 'calculator' not in st.session_state:
        st.session_state.calculator = ActuarialCalculator()
    
    if 'detector' not in st.session_state:
        st.session_state.detector = ContradictionDetector()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    
    if 'performance_monitor' not in st.session_state:
        st.session_state.performance_monitor = None


def initialize_with_api_key(api_key: str):
    """Initialize MCP server and orchestrator with API key"""
    try:
        st.session_state.mcp_server = MCPDataServer(api_key=api_key)
        st.session_state.orchestrator = AgenticOrchestrator(
            st.session_state.mcp_server,
            api_key=api_key
        )
        st.session_state.performance_monitor = st.session_state.orchestrator.monitor
        return True
    except Exception as e:
        st.error(f"Failed to initialize with API key: {str(e)}")
        st.session_state.performance_monitor = None
        return False


def setup_api_key_sidebar():
    """Setup API key input in sidebar without blocking main content"""
    with st.sidebar:
        st.header("🔑 API Configuration")
        
        if not st.session_state.api_key_validated:
            st.warning("⚠️ **API Key Required to Ask Questions**")
            
            api_key_input = st.text_input(
                "Enter Gemini API Key:",
                type="password",
                placeholder="Paste your API key...",
                key="api_key_sidebar_input",
                help="Get free at: https://aistudio.google.com/apikey"
            )
            
            if st.button("Validate & Activate", use_container_width=True, type="primary"):
                if not api_key_input:
                    st.error("Please enter an API key")
                elif not validate_api_key(api_key_input):
                    st.error("Invalid API key format")
                else:
                    st.session_state.user_api_key = api_key_input
                    if initialize_with_api_key(api_key_input):
                        st.session_state.api_key_validated = True
                        st.success("✅ API Key validated and activated!")
                        st.rerun()
            
            with st.expander("🔒 Security Information"):
                st.caption("""
                ✓ Stored only in browser session
                ✓ Never saved to server
                ✓ Never logged or recorded
                ✓ Deleted when tab closes
                """)
        else:
            st.success("✅ API Key Active", icon="✅")
            st.caption(f"Session: {st.session_state.session_id[:12]}...")
            
            if st.button("🔄 Change API Key", use_container_width=True, type="secondary"):
                st.session_state.user_api_key = None
                st.session_state.api_key_validated = False
                st.session_state.mcp_server = None
                st.session_state.orchestrator = None
                st.rerun()
            
            st.divider()


def process_uploaded_file(uploaded_file):
    if st.session_state.mcp_server is None:
        st.warning("⚠️ Please validate API key first to process documents")
        return None
    
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        result = st.session_state.mcp_server.add_resource(tmp_path, resource_id=uploaded_file.name)
        return result
    except Exception as e:
        raise Exception(f"Error processing file {uploaded_file.name}: {str(e)}") from e
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def main():
    # Initialize basic state
    initialize_minimal_state()
    
    st.title("🤖 Agentic RAG Policy Analyst Platform")
    st.markdown("### Next-Generation Insurance Policy Intelligence")
    
    # Setup API key in sidebar (doesn't block main content)
    setup_api_key_sidebar()
    
    with st.expander("ℹ️ About This System", expanded=False):
        st.markdown("""
        This is an **Agentic RAG** system powered by:
        - **Brain**: LangGraph with ReAct pattern + Google Gemini 2.0 Flash (Deep Thinking)
        - **Knowledge Base**: Hybrid Search (FAISS Semantic + BM25 Keyword)
        - **Data Layer**: MCP (Model Context Protocol) Server
        
        **Features**:
        - Autonomous reasoning and planning
        - Cross-references policy clauses
        - Handles ambiguous queries
        - Optimized hybrid search (70% semantic / 30% keyword)
        """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📄 Document Management")
        
        uploaded_files = st.file_uploader(
            "Upload Insurance Policy Documents",
            type=['pdf', 'docx', 'xlsx', 'xls', 'txt'],
            accept_multiple_files=True,
            help="Upload PDF, DOCX, Excel, or TXT files"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.uploaded_files:
                    if st.session_state.mcp_server is None:
                        st.warning("⚠️ Validate API key in sidebar first")
                    else:
                        with st.spinner(f"Processing {uploaded_file.name}..."):
                            try:
                                result = process_uploaded_file(uploaded_file)
                                if result:
                                    st.session_state.uploaded_files.append(uploaded_file.name)
                                    st.success(f"✅ Processed: {uploaded_file.name} ({result['num_chunks']} chunks)")
                            except Exception as e:
                                st.error(f"❌ Failed to process {uploaded_file.name}: {str(e)}")
        
        # Show stats even if MCP not initialized
        if st.session_state.mcp_server:
            stats = st.session_state.mcp_server.get_statistics()
        else:
            stats = {'num_resources': 0, 'total_chunks': 0, 'resource_list': []}
        
        st.metric("Total Documents", stats['num_resources'])
        st.metric("Total Chunks Indexed", stats['total_chunks'])
        
        if stats['resource_list']:
            st.write("**Loaded Documents:**")
            for doc in stats['resource_list']:
                st.text(f"• {doc}")
        
        if st.button("🗑️ Clear All Documents", type="secondary"):
            if st.session_state.mcp_server:
                st.session_state.mcp_server.clear_all()
            st.session_state.uploaded_files = []
            st.session_state.chat_history = []
            st.rerun()
    
    with col2:
        st.subheader("💬 Policy Analysis Chat")
        
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
                    if message["role"] == "assistant":
                        # Show reasoning steps if available
                        if message.get("reasoning_steps"):
                            with st.expander(f"🧠 Agent Reasoning ({message.get('iterations', 1)} iterations)"):
                                for step in message.get("reasoning_steps", []):
                                    st.caption(f"→ {step}")
                        
                        # Show retrieved results
                        if message.get("search_results"):
                            with st.expander("🔍 View Retrieved Policy Sections"):
                                for i, result in enumerate(message["search_results"], 1):
                                    metadata = result.get("metadata", {})
                                    st.markdown(f"**Result {i}** (Score: {result.get('score', 0):.2f})")
                                    st.markdown(f"*Source: {metadata.get('source', 'Unknown')}, Page: {metadata.get('page', 'N/A')}*")
                                    st.text(result.get("content", "")[:300] + "...")
                                    st.divider()
        
        user_query = st.chat_input("Ask a question about your insurance policies...")
        
        if user_query:
            # Get latest stats from MCP server (avoid stale data)
            if st.session_state.mcp_server:
                current_stats = st.session_state.mcp_server.get_statistics()
            else:
                current_stats = {'num_resources': 0, 'total_chunks': 0, 'resource_list': []}
            
            # Check API key before asking question
            if not st.session_state.api_key_validated:
                st.error("⚠️ **Please validate your API key in the sidebar first!**")
                st.info("👈 Scroll to the sidebar and enter your Gemini API key to ask questions")
            elif st.session_state.mcp_server is None or st.session_state.orchestrator is None:
                st.error("⚠️ System not initialized. Please refresh the page.")
            elif current_stats['total_chunks'] == 0:
                st.warning("⚠️ Please upload policy documents before asking questions.")
            else:
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_query
                })
                
                with st.chat_message("user"):
                    st.markdown(user_query)
                
                with st.chat_message("assistant"):
                    with st.spinner("🧠 Agent thinking and searching..."):
                        try:
                            response = st.session_state.orchestrator.run(user_query)
                            
                            answer = response.get("answer", "I couldn't generate an answer.")
                            search_results = response.get("search_results", [])
                            reasoning_steps = response.get("reasoning_steps", [])
                            iterations = response.get("iterations", 1)
                            
                            # Show reasoning process
                            if reasoning_steps:
                                with st.expander(f"🧠 Agent Reasoning ({iterations} iterations)"):
                                    for step in reasoning_steps:
                                        st.caption(f"→ {step}")
                            
                            st.markdown(answer)
                            
                            if search_results:
                                with st.expander("🔍 View Retrieved Policy Sections"):
                                    for i, result in enumerate(search_results, 1):
                                        metadata = result.get("metadata", {})
                                        st.markdown(f"**Result {i}** (Score: {result.get('score', 0):.2f})")
                                        st.markdown(f"*Source: {metadata.get('source', 'Unknown')}, Page: {metadata.get('page', 'N/A')}*")
                                        st.text(result.get("content", "")[:300] + "...")
                                        st.divider()
                            
                            if st.session_state.db:
                                try:
                                    st.session_state.db.add_chat_message(
                                        st.session_state.session_id,
                                        user_query,
                                        answer,
                                        search_results
                                    )
                                except Exception as e:
                                    st.warning(f"Could not save chat history: {str(e)}")
                            
                            response_time = response.get("response_time", 0)
                            
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": answer,
                                "search_results": search_results,
                                "reasoning_steps": reasoning_steps,
                                "iterations": iterations,
                                "response_time": response_time
                            })
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                            # Only pop if we actually appended (safe guard against empty list)
                            if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
                                st.session_state.chat_history.pop()
    
    st.divider()
    
    # Sidebar Tools - Always visible
    with st.sidebar:
        st.header("⚙️ Advanced Tools & Features")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Config", "Actuarial", "Analytics", "Contradictions"])
        
        with tab1:
            st.markdown("**Model**: Gemini 2.0 Flash")
            st.markdown("**Search**: Hybrid (70/30)")
            st.markdown("**Dense**: k=14 FAISS")
            st.markdown("**Sparse**: k=9 BM25")
            st.markdown("**Results**: k=5")
            
            st.divider()
            
            st.markdown("### 🎯 Example Queries")
            st.markdown("""
            - Coverage limits for property damage?
            - Deductible policy for claims?
            - Exclusions for flood damage?
            - Premium calculations?
            - Pre-existing conditions coverage?
            """)
        
        with tab2:
            st.subheader("💰 Actuarial Calculator")
            
            calc_tab1, calc_tab2, calc_tab3 = st.tabs(["Premiums", "Claim Payout", "Risk Analysis"])
            
            with calc_tab1:
                policy_type = st.selectbox("Policy Type", ["auto", "health", "home", "life"], key="policy_type")
                risk_factor = st.slider("Risk Factor", 0.5, 2.0, 1.0, key="risk_factor")
                coverage_amount = st.number_input("Coverage Amount ($)", 1000, 1000000, 100000, key="cov_amt")
                deductible = st.number_input("Deductible ($)", 0, 100000, 500, key="ded")
                
                if st.button("Calculate Premium", key="calc_prem_btn"):
                    try:
                        calc = st.session_state.calculator.calculate_premium(
                            policy_type,
                            {"overall_risk": risk_factor}
                        )
                        
                        st.success("Premium Calculated!")
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Annual Premium", f"${calc.annual_premium:,.2f}")
                            st.metric("Monthly Premium", f"${calc.monthly_premium:,.2f}")
                        with col_b:
                            st.metric("Risk Factor", f"{calc.risk_factor:.2f}x")
                            
                            deduct_info = st.session_state.calculator.calculate_deductible_impact(
                                coverage_amount, deductible
                            )
                            st.metric("Effective Coverage", f"${deduct_info['effective_coverage']:,.0f}")
                    except Exception as e:
                        st.error(f"Calculation error: {e}")
            
            with calc_tab2:
                st.markdown("### Claim Payout Estimator")
                claim_amount = st.number_input("Claim Amount ($)", 0, 500000, 10000, key="claim_amt")
                claim_coverage = st.number_input("Coverage Limit ($)", 0, 1000000, 100000, key="claim_cov")
                claim_deductible = st.number_input("Deductible ($)", 0, 50000, 1000, key="claim_ded")
                coinsurance = st.slider("Coinsurance (%)", 50, 100, 80, key="coinsure")
                
                if st.button("Calculate Payout", key="calc_payout_btn"):
                    try:
                        payout = st.session_state.calculator.calculate_claim_payout(
                            claim_amount, claim_coverage, claim_deductible, coinsurance
                        )
                        st.success("Payout Estimated!")
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Insurer Payout", f"${payout['insurer_payout']:,.2f}")
                            st.metric("After Deductible", f"${payout['after_deductible']:,.2f}")
                        with col_b:
                            st.metric("Your Out-of-Pocket", f"${payout['member_out_of_pocket']:,.2f}")
                            st.metric("Coinsurance Applied", f"{payout['coverage_applied']}%")
                        with col_c:
                            savings = payout['claim_submitted'] - payout['insurer_payout']
                            st.metric("You Save", f"${savings:,.2f}")
                            st.metric("Submitted Amount", f"${payout['claim_submitted']:,.2f}")
                    except Exception as e:
                        st.error(f"Payout error: {e}")
            
            with calc_tab3:
                st.markdown("### Loss Frequency Estimator")
                policy_years = st.number_input("Policy Period (Years)", 1, 30, 5, key="pol_years")
                claim_prob = st.slider("Annual Claim Probability", 0.01, 0.99, 0.1, 0.01, format="%.2f", key="claim_prob")
                
                if st.button("Estimate Risk", key="risk_btn"):
                    try:
                        freq = st.session_state.calculator.estimate_loss_frequency(policy_years, claim_prob)
                        st.success("Risk Estimated!")
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Prob. No Claims", f"{freq['probability_no_claims']:.1%}")
                            st.metric("Expected Claims", f"{freq['expected_number_of_claims']:.2f}")
                        with col_b:
                            st.metric("Prob. ≥1 Claim", f"{freq['probability_at_least_one_claim']:.1%}")
                            st.metric("Policy Period", f"{freq['policy_period_years']} years")
                    except Exception as e:
                        st.error(f"Risk estimation error: {e}")
        
        with tab3:
            st.subheader("📊 Query Analytics & Performance")
            
            if st.session_state.performance_monitor:
                stats = st.session_state.performance_monitor.get_statistics()
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Total Queries", stats['query_count'])
                    st.metric("Error Count", stats['error_count'])
                    st.metric("Cache Hit Rate", f"{stats['cache_hit_rate']*100:.1f}%")
                
                with col_b:
                    st.metric("Avg Response Time", f"{stats['avg_response_time']:.2f}s")
                    st.metric("Cache Hits", stats['cache_hits'])
                    st.metric("Uptime", f"{stats['uptime_seconds']:.0f}s")
                
                if stats['error_count'] > 0:
                    st.warning(f"⚠️ {stats['error_count']} errors recorded")
                    recent_errors = st.session_state.performance_monitor.get_recent_errors(3)
                    for err in recent_errors:
                        st.caption(f"❌ {err['error_type']}: {err['error_msg']}")
            else:
                st.info("Performance metrics will display after first query")
            
            st.divider()
            
            if st.session_state.db and st.session_state.chat_history:
                try:
                    history = st.session_state.db.get_chat_history(st.session_state.session_id, limit=10)
                    
                    if history:
                        st.write(f"**Total DB Queries**: {len(history)}")
                        
                        recent = history[:3]
                        st.write("**Recent Queries**:")
                        for msg in recent:
                            with st.expander(f"Q: {msg['query_text'][:50]}..."):
                                st.text(msg['response_text'][:200])
                except Exception as e:
                    st.info("Query history will accumulate with usage")
            else:
                st.info("Query history will be tracked here")
        
        with tab4:
            st.subheader("🔍 Contradiction Detection")
            
            if st.button("Detect Contradictions"):
                if st.session_state.mcp_server is None:
                    st.warning("⚠️ Validate API key first")
                else:
                    stats = st.session_state.mcp_server.get_statistics()
                    if stats['total_chunks'] < 2:
                        st.warning("Upload at least 2 documents to detect contradictions")
                    else:
                        with st.spinner("Analyzing for contradictions..."):
                            try:
                                docs = st.session_state.mcp_server.get_all_documents()
                                
                                # NOTE: Limiting to first 5 documents for performance.
                                # If more than 5 documents are loaded, contradictions between
                                # documents beyond the 5th will not be detected.
                                max_docs_for_analysis = min(len(docs), 5)
                                contradictions = st.session_state.detector.detect_contradictions(docs[:max_docs_for_analysis])
                                
                                if contradictions:
                                    st.warning(f"Found {len(contradictions)} potential contradictions!")
                                    for i, contra in enumerate(contradictions[:3], 1):
                                        with st.expander(f"⚠️ Issue {i}: {contra['type'].title()}"):
                                            st.markdown(f"**Severity**: {contra['severity'].upper()}")
                                            st.markdown(f"**Between**: {contra['doc1']} ↔ {contra['doc2']}")
                                            st.markdown(f"**Issue**: {contra['description']}")
                                else:
                                    st.success("No contradictions detected!")
                            except Exception as e:
                                st.error(f"Detection error: {e}")
        
        st.divider()
        
        if st.button("📊 System Architecture"):
            st.info("""
            **Agentic RAG Architecture**
            
            1. **User Query** → Streamlit UI
            2. **MCP Server** → Retrieves chunks
            3. **Hybrid Search** → FAISS(70%) + BM25(30%)
            4. **Gemini 2.0(Deep Thinking)** → Synthesizes answer
            5. **Database** → Persists history
            6. **Monitoring** → Evaluating metrics
            7. **Response** → User with citations
            """)


if __name__ == "__main__":
    main()

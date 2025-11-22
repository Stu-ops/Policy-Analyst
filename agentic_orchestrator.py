import os
from typing import Dict, Any, List, TypedDict, Optional, Generator
from google import genai
from google.genai import types
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from mcp_server import MCPDataServer
import json
import time
from performance_monitor import PerformanceMonitor


class AgentState(TypedDict):
    messages: List[Any]
    search_results: List[Dict[str, Any]]
    iteration: int
    final_answer: str
    user_query: str
    reasoning_steps: List[str]
    all_search_results: List[Dict[str, Any]]
    need_more_info: bool
    plan: str


class AgenticOrchestrator:
    def __init__(self, mcp_server: MCPDataServer, api_key: str):
        self.mcp_server = mcp_server
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.0-flash-thinking-exp-01-21"
        self.max_iterations = 5
        self.graph = self._create_graph()
        self.monitor = PerformanceMonitor()
        self.last_error = None
        
    def _create_graph(self):
        """Create enhanced multi-turn ReAct graph"""
        workflow = StateGraph(AgentState)
        
        # Nodes for multi-turn reasoning
        workflow.add_node("plan", self._plan_node)
        workflow.add_node("search", self._search_node)
        workflow.add_node("verify", self._verify_node)
        workflow.add_node("answer", self._answer_node)
        
        # Start with planning
        workflow.add_edge(START, "plan")
        
        # Planning leads to search
        workflow.add_edge("plan", "search")
        
        # Search leads to verification
        workflow.add_edge("search", "verify")
        
        # Conditional: verify decides if we need more searches or go to answer
        workflow.add_conditional_edges(
            "verify",
            self._decide_next_action,
            {
                "search": "search",
                "answer": "answer"
            }
        )
        
        # Answer is terminal
        workflow.add_edge("answer", END)
        
        return workflow.compile()
    
    def _plan_node(self, state: AgentState) -> AgentState:
        """Plan what information we need to find"""
        user_query = state.get("user_query", "")
        iteration = state.get("iteration", 0)
        
        try:
            # Skip planning on first iteration
            if iteration == 0:
                plan_prompt = f"""As an insurance policy analyst, what information do you need to answer this question?
            
Question: {user_query}

Provide a brief plan (1-2 sentences) of what specific information to search for."""
            else:
                # Subsequent iterations - refine the plan
                previous_results = state.get("all_search_results", [])
                plan_prompt = f"""Based on previous search results, what additional information is needed?

Question: {user_query}

Previous findings summary: {len(previous_results)} results found

What additional search terms should we try?"""
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=[{"role": "user", "parts": [{"text": plan_prompt}]}],
            )
            plan = response.text if response.text else "Search for policy details"
            
            reasoning_steps = state.get("reasoning_steps", [])
            reasoning_steps.append(f"Step {iteration+1} Plan: {plan[:100]}...")
            
            return {
                **state,
                "plan": plan,
                "reasoning_steps": reasoning_steps,
                "iteration": iteration + 1
            }
        except Exception as e:
            error_msg = f"Planning error: {str(e)[:100]}"
            self.monitor.record_error("planning", str(e), "plan_node")
            self.last_error = error_msg
            reasoning_steps = state.get("reasoning_steps", [])
            reasoning_steps.append(f"⚠️ {error_msg}")
            return {
                **state,
                "plan": "Search for relevant policy information",
                "reasoning_steps": reasoning_steps,
                "iteration": iteration + 1
            }
    
    def _search_node(self, state: AgentState) -> AgentState:
        """Execute search based on plan"""
        user_query = state.get("user_query", "")
        plan = state.get("plan", user_query)
        
        try:
            # Use plan to refine search query if needed
            search_query = f"{user_query}. Focus on: {plan[:50]}" if len(plan) > 10 else user_query
            
            start_time = time.time()
            search_results = self.mcp_server.execute_search_tool(search_query, k=5)
            search_time = time.time() - start_time
            
            results = search_results.get("results", [])
            
            # Record search metrics
            self.monitor.record_search(search_query, len(results), search_time)
            
            # Accumulate all results across iterations
            all_results = state.get("all_search_results", [])
            all_results.extend(results)
            
            reasoning_steps = state.get("reasoning_steps", [])
            reasoning_steps.append(f"Search found {len(results)} relevant results in {search_time:.2f}s")
            
            return {
                **state,
                "search_results": results,
                "all_search_results": all_results,
                "reasoning_steps": reasoning_steps
            }
        except Exception as e:
            error_msg = f"Search error: {str(e)[:50]}"
            self.monitor.record_error("search", str(e), "search_node")
            self.last_error = error_msg
            reasoning_steps = state.get("reasoning_steps", [])
            reasoning_steps.append(f"⚠️ {error_msg}")
            return {
                **state,
                "search_results": [],
                "reasoning_steps": reasoning_steps
            }
    
    def _verify_node(self, state: AgentState) -> AgentState:
        """Verify if we have enough information or need more searches"""
        user_query = state.get("user_query", "")
        search_results = state.get("search_results", [])
        all_results = state.get("all_search_results", [])
        iteration = state.get("iteration", 0)
        
        try:
            if not search_results:
                # No results found, need to refine search
                reasoning_steps = state.get("reasoning_steps", [])
                reasoning_steps.append("No results found - will refine search strategy")
                return {
                    **state,
                    "need_more_info": True,
                    "reasoning_steps": reasoning_steps
                }
            
            # Check if results are sufficient
            verify_prompt = f"""Based on these search results, do we have enough information to answer the question?

Question: {user_query}

Results found: {len(search_results)}
Total cumulative results: {len(all_results)}

Be concise - just say YES (sufficient) or NO (need more searches)"""
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=[{"role": "user", "parts": [{"text": verify_prompt}]}],
            )
            
            verification = response.text.strip().upper()
            need_more = "NO" in verification or "NOT ENOUGH" in verification
            
            # Don't exceed max iterations
            if iteration >= self.max_iterations:
                need_more = False
            
            reasoning_steps = state.get("reasoning_steps", [])
            reasoning_steps.append(f"Verification: {'Need more info' if need_more else 'Sufficient info found'}")
            
            return {
                **state,
                "need_more_info": need_more,
                "reasoning_steps": reasoning_steps
            }
        except Exception as e:
            error_msg = f"Verification error: {str(e)[:50]}"
            self.monitor.record_error("verification", str(e), "verify_node")
            self.last_error = error_msg
            reasoning_steps = state.get("reasoning_steps", [])
            reasoning_steps.append(f"⚠️ {error_msg} - proceeding with answer")
            return {
                **state,
                "need_more_info": False,
                "reasoning_steps": reasoning_steps
            }
    
    def _decide_next_action(self, state: AgentState) -> str:
        """Decide whether to search again or synthesize answer"""
        need_more = state.get("need_more_info", False)
        iteration = state.get("iteration", 0)
        
        if need_more and iteration < self.max_iterations:
            return "search"
        else:
            return "answer"
    
    def _answer_node(self, state: AgentState) -> AgentState:
        """Synthesize final answer from all gathered information"""
        user_query = state.get("user_query", "")
        all_search_results = state.get("all_search_results", [])
        reasoning_steps = state.get("reasoning_steps", [])
        iteration = state.get("iteration", 0)
        
        system_prompt = """You are an expert insurance policy analyst with deep knowledge of policy terms, coverage, exclusions, and claims.

Your task is to answer questions about insurance policies using ONLY the information provided in the search results.

Guidelines:
1. Always cite specific sources, page numbers, and policy sections
2. If information is insufficient, clearly state what's missing
3. Provide clear, accurate, comprehensive answers
4. Quote relevant policy text when helpful
5. Cite exact numbers and limits as they appear
6. Cross-reference related clauses when relevant
7. Identify any contradictions or ambiguities

Be thorough, precise, and grounded in the provided documents."""
        
        try:
            if not all_search_results:
                context = "No relevant policy documents were found for this query."
            else:
                context_parts = ["RETRIEVED POLICY INFORMATION:\n"]
                for i, result in enumerate(all_search_results, 1):
                    content = result.get("content", "")
                    metadata = result.get("metadata", {})
                    score = result.get("score", 0)
                    
                    source = metadata.get("source", "Unknown")
                    page = metadata.get("page", "N/A")
                    
                    context_parts.append(f"\n[Source {i}] (Relevance: {score:.2f})")
                    context_parts.append(f"Document: {source}, Page: {page}")
                    context_parts.append(f"Content: {content}")
                    context_parts.append("-" * 80)
                
                context = "\n".join(context_parts)
            
            # Include reasoning steps in context for better answers
            reasoning_context = "\n".join([f"• {step}" for step in reasoning_steps[:5]])
            
            prompt = f"""{context}

AGENT REASONING PROCESS:
{reasoning_context}

USER QUESTION: {user_query}

Based on the above policy information and reasoning, provide a comprehensive answer. Always cite your sources and be precise."""
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt
                )
            )
            
            answer = response.text if response.text else "I apologize, but I couldn't generate an answer."
            
            reasoning_steps.append(f"Final answer synthesized after {iteration} iterations")
            
            return {
                **state,
                "final_answer": answer,
                "reasoning_steps": reasoning_steps,
                "messages": state.get("messages", []) + [
                    HumanMessage(content=user_query),
                    AIMessage(content=answer)
                ]
            }
        except Exception as e:
            error_msg = f"Error generating answer: {str(e)[:100]}"
            self.monitor.record_error("answer_generation", str(e), "answer_node")
            self.last_error = error_msg
            reasoning_steps.append(f"⚠️ Answer generation failed")
            
            return {
                **state,
                "final_answer": f"I encountered an error while generating an answer: {error_msg}",
                "reasoning_steps": reasoning_steps,
                "messages": state.get("messages", []) + [
                    HumanMessage(content=user_query),
                    AIMessage(content=error_msg)
                ]
            }
    
    def run(self, query: str) -> Dict[str, Any]:
        """Run multi-turn ReAct loop"""
        start_time = time.time()
        try:
            initial_state = {
                "messages": [],
                "search_results": [],
                "iteration": 0,
                "final_answer": "",
                "user_query": query,
                "reasoning_steps": [],
                "all_search_results": [],
                "need_more_info": False,
                "plan": ""
            }
            
            final_state = self.graph.invoke(initial_state)
            
            response_time = time.time() - start_time
            search_results = final_state.get("all_search_results", [])
            iterations = final_state.get("iteration", 1)
            
            # Record query metrics
            self.monitor.record_query(query, response_time, iterations, len(search_results), success=True)
            
            return {
                "query": query,
                "answer": final_state.get("final_answer", ""),
                "search_results": search_results,
                "iterations": iterations,
                "reasoning_steps": final_state.get("reasoning_steps", []),
                "full_conversation": final_state.get("messages", []),
                "response_time": response_time
            }
        except Exception as e:
            response_time = time.time() - start_time
            self.monitor.record_query(query, response_time, 0, 0, success=False)
            self.monitor.record_error("run", str(e), "orchestrator.run")
            self.last_error = str(e)
            
            return {
                "query": query,
                "answer": f"System error: {str(e)[:100]}",
                "search_results": [],
                "iterations": 0,
                "reasoning_steps": ["⚠️ Query processing failed"],
                "full_conversation": [],
                "response_time": response_time
            }
    
    def stream_run(self, query: str):
        """Stream multi-turn ReAct execution"""
        initial_state = {
            "messages": [],
            "search_results": [],
            "iteration": 0,
            "final_answer": "",
            "user_query": query,
            "reasoning_steps": [],
            "all_search_results": [],
            "need_more_info": False,
            "plan": ""
        }
        
        for state in self.graph.stream(initial_state):
            yield state

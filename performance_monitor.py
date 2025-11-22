import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from collections import defaultdict
import json


class PerformanceMonitor:
    """Monitor and log system performance metrics"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.session_start = datetime.now()
        self.query_count = 0
        self.error_count = 0
        self.total_processing_time = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
        
    def record_query(self, query: str, response_time: float, iterations: int, num_results: int, success: bool = True):
        """Record query metrics"""
        metric = {
            "timestamp": datetime.now().isoformat(),
            "query_length": len(query),
            "response_time": response_time,
            "iterations": iterations,
            "num_results": num_results,
            "success": success
        }
        self.metrics["queries"].append(metric)
        self.query_count += 1
        self.total_processing_time += response_time
        if not success:
            self.error_count += 1
    
    def record_search(self, query: str, num_results: int, search_time: float, cached: bool = False):
        """Record search metrics"""
        metric = {
            "timestamp": datetime.now().isoformat(),
            "query_hash": hash(query) % 1000000,
            "num_results": num_results,
            "search_time": search_time,
            "cached": cached
        }
        self.metrics["searches"].append(metric)
        if cached:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
    
    def record_error(self, error_type: str, error_msg: str, context: str = ""):
        """Record error metrics"""
        metric = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "error_msg": error_msg[:100],
            "context": context
        }
        self.metrics["errors"].append(metric)
        self.error_count += 1
    
    def record_embedding(self, num_documents: int, embedding_time: float):
        """Record embedding performance"""
        metric = {
            "timestamp": datetime.now().isoformat(),
            "num_documents": num_documents,
            "embedding_time": embedding_time,
            "avg_time_per_doc": embedding_time / num_documents if num_documents > 0 else 0
        }
        self.metrics["embeddings"].append(metric)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        avg_response_time = self.total_processing_time / self.query_count if self.query_count > 0 else 0
        cache_hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        uptime = (datetime.now() - self.session_start).total_seconds()
        
        return {
            "query_count": self.query_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / self.query_count if self.query_count > 0 else 0,
            "avg_response_time": avg_response_time,
            "total_processing_time": self.total_processing_time,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "uptime_seconds": uptime,
            "session_start": self.session_start.isoformat(),
            "last_updated": datetime.now().isoformat()
        }
    
    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent errors"""
        return self.metrics["errors"][-limit:] if "errors" in self.metrics else []
    
    def get_recent_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent queries"""
        return self.metrics["queries"][-limit:] if "queries" in self.metrics else []

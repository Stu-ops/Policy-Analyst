from typing import List, Dict, Any, Optional
from hybrid_search import HybridSearchEngine
from document_processor import DocumentProcessor
import time


class MCPDataServer:
    def __init__(self, api_key: str):
        self.search_engine = HybridSearchEngine(api_key=api_key, semantic_weight=0.7, keyword_weight=0.3)
        self.document_processor = DocumentProcessor()
        self.resources = {}
        
    def add_resource(self, file_path: str, resource_id: Optional[str] = None):
        try:
            if resource_id is None:
                resource_id = file_path
            
            processed_doc = self.document_processor.process_file(file_path)
            
            self.resources[resource_id] = processed_doc
            
            self.search_engine.add_documents(
                processed_doc['chunks'],
                processed_doc['metadata']
            )
            
            return {
                'resource_id': resource_id,
                'num_chunks': len(processed_doc['chunks']),
                'source': processed_doc['source'],
                'error': None
            }
        except Exception as e:
            return {
                'error': str(e),
                'resource_id': None
            }
    
    def add_documents_from_uploaded(self, documents: List[str], metadata: List[Dict[str, Any]]):
        self.search_engine.add_documents(documents, metadata)
    
    def execute_search_tool(self, query: str, k: int = 5) -> Dict[str, Any]:
        try:
            results = self.search_engine.search(query, final_k=k)
            
            return {
                'query': query,
                'results': results,
                'num_results': len(results),
                'error': None
            }
        except Exception as e:
            return {
                'query': query,
                'results': [],
                'num_results': 0,
                'error': str(e)
            }
    
    def get_resource(self, resource_id: str) -> Optional[Dict[str, Any]]:
        return self.resources.get(resource_id)
    
    def list_resources(self) -> List[str]:
        return list(self.resources.keys())
    
    def clear_all(self):
        self.resources = {}
        self.search_engine.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        try:
            cache_stats = self.search_engine.get_cache_stats() if hasattr(self.search_engine, 'get_cache_stats') else {}
            return {
                'num_resources': len(self.resources),
                'total_chunks': len(self.search_engine.documents),
                'resource_list': list(self.resources.keys()),
                'cache_hits': cache_stats.get('hits', 0),
                'cache_misses': cache_stats.get('misses', 0)
            }
        except Exception as e:
            return {
                'num_resources': len(self.resources),
                'total_chunks': 0,
                'resource_list': list(self.resources.keys()),
                'error': str(e)
            }

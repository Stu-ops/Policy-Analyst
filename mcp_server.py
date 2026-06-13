from typing import List, Dict, Any, Optional
from hybrid_search import HybridSearchEngine
from document_processor import DocumentProcessor


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
                'resource_id': None,
                'num_chunks': 0,
                'source': None
            }
    
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
    
    def clear_all(self):
        self.resources = {}
        self.search_engine.clear()
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all indexed documents with their content and metadata safely"""
        docs = self.search_engine.documents
        metas = self.search_engine.metadata
        # Ensure lists are aligned; zip() would silently truncate if out of sync
        if len(docs) != len(metas):
            min_len = min(len(docs), len(metas))
            return [
                {"content": docs[i], "source": metas[i].get('source', 'Unknown')}
                for i in range(min_len)
            ]
        return [
            {"content": chunk, "source": meta.get('source', 'Unknown')}
            for chunk, meta in zip(docs, metas)
        ]
    
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

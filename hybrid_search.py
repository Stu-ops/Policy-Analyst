import numpy as np
from typing import List, Dict, Any, Tuple
import faiss
from rank_bm25 import BM25Okapi
from google import genai
import time
import hashlib


class HybridSearchEngine:
    def __init__(self, api_key: str, semantic_weight: float = 0.7, keyword_weight: float = 0.3):
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.index = None
        self.bm25 = None
        self.documents = []
        self.metadata = []
        self.embeddings = []
        self.client = genai.Client(api_key=api_key)
        self.embedding_dimension = 768
        self.search_cache = {}  # Query hash -> results cache (bounded, see _trim_cache)
        self.cache_stats = {"hits": 0, "misses": 0}
        self._cache_max_size = 100  # Prevents unbounded memory growth
        
    def add_documents(self, documents: List[str], metadata: List[Dict[str, Any]]):
        self.documents.extend(documents)
        self.metadata.extend(metadata)
        
        new_embeddings = self._get_embeddings(documents)
        self.embeddings.extend(new_embeddings)
        
        self._build_faiss_index()
        self._build_bm25_index()
        # Invalidate search cache since document corpus changed
        self.clear_cache()
    
    def _get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for a list of texts using batch API where possible"""
        embeddings = []
        
        # Process in batches of up to 100 texts per API call for efficiency
        batch_size = 100
        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start:batch_start + batch_size]
            batch_embeddings = []
            valid_texts = []
            valid_indices = []
            
            for i, text in enumerate(batch):
                if not text or not text.strip():
                    batch_embeddings.append((batch_start + i, np.zeros(self.embedding_dimension, dtype=np.float32)))
                else:
                    valid_texts.append(text[:10000])
                    valid_indices.append(i)
            
            if valid_texts:
                try:
                    result = self.client.models.embed_content(
                        model='models/text-embedding-004',
                        content=valid_texts,
                        config={
                            'task_type': 'retrieval_document',
                            'output_dimensionality': self.embedding_dimension
                        }
                    )
                    
                    for j, embedding_obj in enumerate(result.embeddings):
                        actual_idx = batch_start + valid_indices[j]
                        embedding = np.array(embedding_obj.values, dtype=np.float32)
                        batch_embeddings.append((actual_idx, embedding))
                except Exception as e:
                    print(f"Error getting embeddings for batch starting at {batch_start}: {e}")
                    for idx in valid_indices:
                        actual_idx = batch_start + idx
                        batch_embeddings.append((actual_idx, np.zeros(self.embedding_dimension, dtype=np.float32)))
            
            # Sort by original index and extract embeddings
            batch_embeddings.sort(key=lambda x: x[0])
            embeddings.extend([emb for _, emb in batch_embeddings])
        
        return embeddings
    
    def _build_faiss_index(self):
        if not self.embeddings:
            return
        
        embedding_matrix = np.array(self.embeddings).astype('float32')
        
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.embedding_dimension)
            self.index.add(embedding_matrix)
        else:
            # Track existing count to only add new embeddings incrementally
            existing_count = len(self.embeddings) - len(embedding_matrix)
            if existing_count < len(self.embeddings):
                new_count = len(self.embeddings) - existing_count
                new_embeddings = np.array(self.embeddings[-new_count:]).astype('float32')
                self.index.add(new_embeddings)
    
    def _build_bm25_index(self):
        tokenized_docs = [doc.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)
    
    def search(self, query: str, k_dense: int = 14, k_sparse: int = 9, final_k: int = 5) -> List[Dict[str, Any]]:
        if not self.documents:
            return []
        
        # Check cache
        cache_key = self._get_cache_key(query, k_dense, k_sparse, final_k)
        if cache_key in self.search_cache:
            self.cache_stats["hits"] += 1
            return self.search_cache[cache_key]
        
        self.cache_stats["misses"] += 1
        
        dense_results = self._semantic_search(query, k_dense)
        sparse_results = self._keyword_search(query, k_sparse)
        
        combined_results = self._combine_results(dense_results, sparse_results, final_k)
        
        # Cache the results
        self.search_cache[cache_key] = combined_results
        self._trim_cache()
        
        return combined_results
    
    def _get_cache_key(self, query: str, k_dense: int, k_sparse: int, final_k: int) -> str:
        """Generate cache key for search query"""
        key_str = f"{query}_{k_dense}_{k_sparse}_{final_k}"
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def _trim_cache(self):
        """Prevent unbounded cache growth by evicting oldest entries when over limit"""
        if len(self.search_cache) > self._cache_max_size:
            # Remove the oldest 25% of entries
            remove_count = len(self.search_cache) - self._cache_max_size
            # dict preserves insertion order in Python 3.7+
            for key in list(self.search_cache.keys())[:remove_count]:
                del self.search_cache[key]
    
    def _semantic_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        if self.index is None:
            return []
        
        query_embedding = self._get_embeddings([query])[0]
        query_embedding = query_embedding.reshape(1, -1)
        
        k = min(k, len(self.documents))
        distances, indices = self.index.search(query_embedding, k)
        
        # Convert L2 distances to similarity scores using inverse distance with epsilon
        # This is more robust than linear normalization when distances cluster tightly
        distances = distances[0]
        if len(distances) == 0:
            return []
        
        eps = 1e-8
        # Use softmax-like normalization: score = 1 / (1 + distance)
        scores = [(idx, float(1.0 / (1.0 + dist + eps))) 
                  for dist, idx in zip(distances, indices[0])]
        
        return scores
    
    def _keyword_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        if self.bm25 is None:
            return []
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        top_k_indices = np.argsort(scores)[-k:][::-1]
        max_score = np.max(scores) if len(scores) > 0 else 1.0
        
        results = [(idx, scores[idx] / max_score if max_score > 0 else 0) 
                   for idx in top_k_indices]
        
        return results
    
    def _combine_results(self, dense_results: List[Tuple[int, float]], 
                        sparse_results: List[Tuple[int, float]], 
                        final_k: int) -> List[Dict[str, Any]]:
        score_dict = {}
        
        for idx, score in dense_results:
            score_dict[idx] = score_dict.get(idx, 0) + self.semantic_weight * score
        
        for idx, score in sparse_results:
            score_dict[idx] = score_dict.get(idx, 0) + self.keyword_weight * score
        
        sorted_results = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)[:final_k]
        
        results = []
        for idx, score in sorted_results:
            results.append({
                'content': self.documents[idx],
                'metadata': self.metadata[idx],
                'score': float(score),
                'index': int(idx)
            })
        
        return results
    
    def clear(self):
        self.documents = []
        self.metadata = []
        self.embeddings = []
        self.index = None
        self.bm25 = None
        self.search_cache.clear()
        self.cache_stats = {"hits": 0, "misses": 0}
    
    def clear_cache(self):
        """Clear search results cache (preserve cache stats for monitoring)"""
        self.search_cache.clear()
        # Reset cache stats since the corpus changed and prior stats are no longer meaningful
        self.cache_stats = {"hits": 0, "misses": 0}
    
    def get_cache_stats(self):
        """Get cache statistics"""
        return self.cache_stats

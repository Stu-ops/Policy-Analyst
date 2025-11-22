import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import psycopg2
from psycopg2.extras import RealDictCursor

class DatabaseManager:
    def __init__(self):
        self.conn_string = os.environ.get("DATABASE_URL")
        if not self.conn_string:
            raise ValueError("DATABASE_URL not found in environment variables")
    
    def _get_connection(self):
        return psycopg2.connect(self.conn_string)
    
    def initialize_schema(self):
        """Create all required tables"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    filename VARCHAR(255) NOT NULL UNIQUE,
                    file_type VARCHAR(50) NOT NULL,
                    chunks_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    version INTEGER DEFAULT 1
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    page_number INTEGER,
                    embedding_vector BYTEA,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
                    UNIQUE(document_id, chunk_index)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_history (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255) NOT NULL,
                    query_text TEXT NOT NULL,
                    response_text TEXT NOT NULL,
                    search_results JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    tokens_used INTEGER DEFAULT 0
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analytics (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255) NOT NULL,
                    query_type VARCHAR(100),
                    num_documents_searched INTEGER,
                    response_time_ms FLOAT,
                    relevance_score FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS contradictions (
                    id SERIAL PRIMARY KEY,
                    document1_id INTEGER NOT NULL,
                    document2_id INTEGER NOT NULL,
                    clause1_text TEXT NOT NULL,
                    clause2_text TEXT NOT NULL,
                    severity VARCHAR(20),
                    description TEXT,
                    flagged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document1_id) REFERENCES documents(id),
                    FOREIGN KEY (document2_id) REFERENCES documents(id)
                )
            ''')
            
            conn.commit()
            print("Database schema initialized successfully")
        finally:
            cursor.close()
            conn.close()
    
    def add_document(self, filename: str, file_type: str) -> int:
        """Add a document to the database and return its ID"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO documents (filename, file_type) 
                VALUES (%s, %s)
                RETURNING id
            ''', (filename, file_type))
            
            doc_id = cursor.fetchone()[0]
            conn.commit()
            return doc_id
        finally:
            cursor.close()
            conn.close()
    
    def add_chunks(self, document_id: int, chunks: List[Dict[str, Any]]):
        """Add chunks for a document"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            for i, chunk in enumerate(chunks):
                cursor.execute('''
                    INSERT INTO document_chunks (document_id, chunk_index, content, page_number)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (document_id, chunk_index) DO NOTHING
                ''', (
                    document_id,
                    i,
                    chunk.get('content', ''),
                    chunk.get('page', None)
                ))
            
            cursor.execute('UPDATE documents SET chunks_count = %s WHERE id = %s',
                          (len(chunks), document_id))
            
            conn.commit()
        finally:
            cursor.close()
            conn.close()
    
    def add_chat_message(self, session_id: str, query: str, response: str, 
                        search_results: List[Dict] = None, tokens: int = 0):
        """Add a chat message to history"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO chat_history (session_id, query_text, response_text, search_results, tokens_used)
                VALUES (%s, %s, %s, %s, %s)
            ''', (
                session_id,
                query,
                response,
                json.dumps(search_results) if search_results else None,
                tokens
            ))
            
            conn.commit()
        finally:
            cursor.close()
            conn.close()
    
    def get_chat_history(self, session_id: str, limit: int = 50) -> List[Dict]:
        """Get chat history for a session"""
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            cursor.execute('''
                SELECT id, session_id, query_text, response_text, search_results, created_at
                FROM chat_history
                WHERE session_id = %s
                ORDER BY created_at DESC
                LIMIT %s
            ''', (session_id, limit))
            
            return cursor.fetchall()
        finally:
            cursor.close()
            conn.close()
    
    def add_analytics(self, session_id: str, query_type: str, num_docs: int,
                     response_time_ms: float, relevance_score: float):
        """Log analytics data"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO analytics (session_id, query_type, num_documents_searched, 
                                      response_time_ms, relevance_score)
                VALUES (%s, %s, %s, %s, %s)
            ''', (session_id, query_type, num_docs, response_time_ms, relevance_score))
            
            conn.commit()
        finally:
            cursor.close()
            conn.close()
    
    def get_documents(self) -> List[Dict]:
        """Get all documents"""
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            cursor.execute('SELECT id, filename, file_type, chunks_count, created_at, version FROM documents')
            return cursor.fetchall()
        finally:
            cursor.close()
            conn.close()
    
    def flag_contradiction(self, doc1_id: int, doc2_id: int, clause1: str, 
                          clause2: str, severity: str, description: str):
        """Flag a contradiction between two documents"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO contradictions (document1_id, document2_id, clause1_text, 
                                          clause2_text, severity, description)
                VALUES (%s, %s, %s, %s, %s, %s)
            ''', (doc1_id, doc2_id, clause1, clause2, severity, description))
            
            conn.commit()
        finally:
            cursor.close()
            conn.close()
    
    def get_contradictions(self) -> List[Dict]:
        """Get all flagged contradictions"""
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            cursor.execute('''
                SELECT id, d1.filename as doc1, d2.filename as doc2, severity, description, flagged_at
                FROM contradictions c
                JOIN documents d1 ON c.document1_id = d1.id
                JOIN documents d2 ON c.document2_id = d2.id
                ORDER BY flagged_at DESC
            ''')
            
            return cursor.fetchall()
        finally:
            cursor.close()
            conn.close()

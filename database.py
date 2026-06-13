import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import pool

class DatabaseManager:
    def __init__(self):
        self.conn_string = os.environ.get("DATABASE_URL")
        if not self.conn_string:
            raise ValueError("DATABASE_URL not found in environment variables")
        # Thread-safe connection pool (min 1, max 10 connections)
        self._pool = pool.ThreadedConnectionPool(1, 10, self.conn_string)
    
    def _get_connection(self):
        return self._pool.getconn()
    
    def _put_connection(self, conn):
        """Return connection to pool gracefully"""
        try:
            self._pool.putconn(conn)
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
    
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
            
            # DEAD CODE: 'analytics' table is created here but never written to or read from
            # anywhere in the codebase. Consider removing this table creation.
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
            
            # DEAD CODE: 'contradictions' table is created here but never written to or read from.
            # Contradictions are detected in-memory by contradiction_detector.py and never persisted.
            # Consider removing this table creation.
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
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()
            self._put_connection(conn)
    
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
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()
            self._put_connection(conn)
    
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
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()
            self._put_connection(conn)
    

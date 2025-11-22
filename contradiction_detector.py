from typing import List, Dict, Any, Tuple
from enum import Enum
import re


class ContradictionSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ContradictionDetector:
    def __init__(self):
        self.contradiction_keywords = {
            'coverage': ['covers', 'covered', 'included', 'not included', 'excluded', 'excludes'],
            'limits': ['limit', 'maximum', 'cap', 'ceiling', 'unlimited'],
            'exclusions': ['exclude', 'excluded', 'not covered', 'not eligible'],
            'conditions': ['require', 'prerequisite', 'condition', 'must', 'shall'],
            'frequency': ['annual', 'monthly', 'per year', 'per month', 'lifetime'],
            'amounts': [r'\$\d+', r'\d+\s*%', r'\d+\s*days?', r'\d+\s*months?', r'\d+\s*years?']
        }
    
    def detect_contradictions(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect contradictions between multiple documents"""
        contradictions = []
        
        for i, doc1 in enumerate(documents):
            for doc2 in documents[i+1:]:
                doc_contradictions = self._compare_documents(doc1, doc2)
                contradictions.extend(doc_contradictions)
        
        return sorted(contradictions, key=lambda x: self._severity_to_int(x['severity']), reverse=True)
    
    def _compare_documents(self, doc1: Dict[str, Any], doc2: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Compare two documents for contradictions"""
        contradictions = []
        
        text1 = doc1.get('content', '').lower()
        text2 = doc2.get('content', '').lower()
        
        for clause_type, keywords in self.contradiction_keywords.items():
            clauses1 = self._extract_clauses(text1, keywords)
            clauses2 = self._extract_clauses(text2, keywords)
            
            for clause1 in clauses1:
                for clause2 in clauses2:
                    if self._is_contradictory(clause1, clause2):
                        severity = self._assess_severity(clause_type, clause1, clause2)
                        
                        contradictions.append({
                            'type': clause_type,
                            'clause1': clause1[:100],
                            'clause2': clause2[:100],
                            'doc1': doc1.get('source', 'Unknown'),
                            'doc2': doc2.get('source', 'Unknown'),
                            'severity': severity,
                            'description': self._generate_description(clause_type, clause1, clause2)
                        })
        
        return contradictions
    
    def _extract_clauses(self, text: str, keywords: List[str]) -> List[str]:
        """Extract relevant clauses from text"""
        clauses = []
        sentences = re.split(r'[.!?]\s+', text)
        
        for sentence in sentences:
            for keyword in keywords:
                if keyword in sentence:
                    clauses.append(sentence.strip())
                    break
        
        return clauses
    
    def _is_contradictory(self, clause1: str, clause2: str) -> bool:
        """Check if two clauses are contradictory"""
        negation_words = ['not', 'no', 'never', 'exclude', 'excluded', 'cannot', "can't"]
        
        has_negation1 = any(word in clause1 for word in negation_words)
        has_negation2 = any(word in clause2 for word in negation_words)
        
        if has_negation1 == has_negation2:
            return False
        
        words1 = set(clause1.split())
        words2 = set(clause2.split())
        
        overlap = len(words1.intersection(words2)) / max(len(words1), len(words2))
        
        return overlap > 0.3
    
    def _assess_severity(self, clause_type: str, clause1: str, clause2: str) -> str:
        """Assess contradiction severity"""
        critical_types = ['exclusions', 'limits', 'coverage']
        
        has_numbers1 = bool(re.search(r'\$\d+|\d+\s*%', clause1))
        has_numbers2 = bool(re.search(r'\$\d+|\d+\s*%', clause2))
        
        if clause_type in critical_types and (has_numbers1 or has_numbers2):
            return ContradictionSeverity.CRITICAL.value
        elif clause_type in ['coverage', 'conditions']:
            return ContradictionSeverity.HIGH.value
        elif clause_type in ['frequency', 'amounts']:
            return ContradictionSeverity.MEDIUM.value
        else:
            return ContradictionSeverity.LOW.value
    
    def _generate_description(self, clause_type: str, clause1: str, clause2: str) -> str:
        """Generate description of contradiction"""
        negation_words = ['not', 'no', 'never', 'exclude', 'excluded', 'cannot']
        
        is_negation1 = any(word in clause1 for word in negation_words)
        is_negation2 = any(word in clause2 for word in negation_words)
        
        if is_negation1 and not is_negation2:
            return f"Document 1 excludes {clause_type} while Document 2 includes it"
        elif is_negation2 and not is_negation1:
            return f"Document 2 excludes {clause_type} while Document 1 includes it"
        else:
            return f"Conflicting statements about {clause_type} between documents"
    
    def _severity_to_int(self, severity: str) -> int:
        """Convert severity to int for sorting"""
        severity_map = {
            'critical': 4,
            'high': 3,
            'medium': 2,
            'low': 1
        }
        return severity_map.get(severity, 0)
    
    def cross_reference_clauses(self, documents: List[Dict[str, Any]], 
                               search_query: str) -> List[Dict[str, Any]]:
        """Find related clauses across documents matching a query"""
        matching_clauses = []
        
        for doc in documents:
            text = doc.get('content', '').lower()
            query_lower = search_query.lower()
            
            if query_lower in text:
                sentences = re.split(r'[.!?]\s+', text)
                
                for sentence in sentences:
                    if query_lower in sentence:
                        matching_clauses.append({
                            'document': doc.get('source', 'Unknown'),
                            'page': doc.get('page', 'N/A'),
                            'clause': sentence.strip(),
                            'relevance': self._calculate_relevance(sentence, search_query)
                        })
        
        return sorted(matching_clauses, key=lambda x: x['relevance'], reverse=True)
    
    def _calculate_relevance(self, text: str, query: str) -> float:
        """Calculate relevance score"""
        text_lower = text.lower()
        query_lower = query.lower()
        
        query_words = query_lower.split()
        matches = sum(1 for word in query_words if word in text_lower)
        
        return matches / len(query_words) if query_words else 0

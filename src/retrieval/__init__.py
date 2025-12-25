"""Retrieval components for legal document processing"""
from .chunking import DocumentChunker, RecursiveCharacterTextSplitter
from .embedding import EmbeddingModel
from .retriever import LegalRetriever

__all__ = ['DocumentChunker', 'RecursiveCharacterTextSplitter', 'EmbeddingModel', 'LegalRetriever']

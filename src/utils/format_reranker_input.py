"""
Format Reranker Input Module
Adapter to convert ChromaDB output (column-based) to Reranker input (row-based)
"""

from typing import Any, Dict, List, Union, Optional


def format_chroma_to_reranker(chroma_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert ChromaDB query results (column-based) to Reranker input format (row-based).
    """
    if not chroma_results:
        return []
    
    # ChromaDB returns nested lists: [["doc1", "doc2"]] for single query
    documents = chroma_results.get("documents", [[]])
    metadatas = chroma_results.get("metadatas", [[]])
    ids = chroma_results.get("ids", [[]])
    distances = chroma_results.get("distances", [[]])
    
    # If multiple queries, only process the first one
    doc_list = documents[0] if documents and len(documents) > 0 else []
    meta_list = metadatas[0] if metadatas and len(metadatas) > 0 else []
    id_list = ids[0] if ids and len(ids) > 0 else []
    dist_list = distances[0] if distances and len(distances) > 0 else []
    
    # Convert to row-based format
    formatted_docs = []
    for idx in range(len(doc_list)):
        doc_dict = {
            "content": doc_list[idx],
            "text": doc_list[idx]
        }
        
        # Add metadata (aid, law_id, etc.)
        if idx < len(meta_list) and meta_list[idx]:
            meta = meta_list[idx]
            if isinstance(meta, dict):
                doc_dict.update(meta)
        
        # Add ChromaDB ID if available
        if idx < len(id_list):
            doc_dict["chroma_id"] = id_list[idx]
        
        # Add distance if available
        if idx < len(dist_list):
            doc_dict["distance"] = dist_list[idx]
        
        formatted_docs.append(doc_dict)
    
    return formatted_docs


def format_documents_for_reranker(
    documents: Union[List[str], List[Dict[str, Any]]],
    metadatas: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Format documents for reranker input.
    Handles both string lists and dict lists, with optional metadata.
    
    Args:
        documents: List of document strings or dicts
        metadatas: Optional list of metadata dicts (if documents are strings)
        
    Returns:
        List of document dictionaries with standardized format
    """
    if not documents:
        return []
    
    formatted_docs = []
    
    # If documents are already dicts
    if isinstance(documents[0], dict):
        for doc in documents:
            # Ensure standard format
            formatted_doc = {
                "content": doc.get("content", doc.get("text", "")),
                "text": doc.get("text", doc.get("content", ""))
            }
            # Copy all other fields
            for key, value in doc.items():
                if key not in ["content", "text"]:
                    formatted_doc[key] = value
            formatted_docs.append(formatted_doc)
    
    # If documents are strings
    elif isinstance(documents[0], str):
        for idx, doc_text in enumerate(documents):
            formatted_doc = {
                "content": doc_text,
                "text": doc_text
            }
            # Add metadata if provided
            if metadatas and idx < len(metadatas):
                if isinstance(metadatas[idx], dict):
                    formatted_doc.update(metadatas[idx])
            formatted_docs.append(formatted_doc)
    
    else:
        raise ValueError(f"Unsupported document type: {type(documents[0])}")
    
    return formatted_docs


def extract_texts_from_documents(documents: List[Dict[str, Any]]) -> List[str]:
    """
    Extract text content from formatted document dictionaries.
    """
    return [
        doc.get("content", doc.get("text", ""))
        for doc in documents
    ]


def extract_ids_from_documents(documents: List[Dict[str, Any]], id_key: str = "aid") -> List[Any]:
    """
    Extract IDs from formatted document dictionaries.
    """
    return [
        doc.get(id_key, doc.get("id", idx))
        for idx, doc in enumerate(documents)
    ]


def build_documents_dict(documents: List[Dict[str, Any]], id_key: str = "aid") -> Dict[str, Dict[str, Any]]:
    """
    Build documents dictionary from list for efficient lookup.
    
    Example:
        Input: [{"aid": "A1", "content": "..."}, {"aid": "A2", "content": "..."}]
        Output: {"A1": {"aid": "A1", "content": "..."}, "A2": {"aid": "A2", "content": "..."}}
    """
    documents_dict = {}
    for idx, doc in enumerate(documents):
        # Get ID from document
        doc_id = doc.get(id_key, doc.get("id", f"doc_{idx}"))
        if doc_id:
            documents_dict[doc_id] = doc
    return documents_dict
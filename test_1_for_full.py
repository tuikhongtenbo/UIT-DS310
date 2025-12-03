"""
Test script to debug pipeline with 1 sample from public_test.json
Shows output of each module: Retriever -> Reranker -> Qwen -> Final
"""
import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from config.config import AppConfig
from src.utils.logger import setup_logger
from main import ExperimentPipeline

logger = setup_logger("test")


def print_section(title: str, char: str = "=", width: int = 80):
    """Print a formatted section header"""
    print("\n" + char * width)
    print(f"  {title}")
    print(char * width)


def print_list_items(items: List[Any], max_items: int = 10, prefix: str = "  "):
    """Print list items with limit"""
    for i, item in enumerate(items[:max_items]):
        print(f"{prefix}[{i+1}] {item}")
    if len(items) > max_items:
        print(f"{prefix}... and {len(items) - max_items} more items")


def test_pipeline_with_sample():
    """Test pipeline with 1 sample from public_test.json"""
    
    # Load config
    config_path = os.path.join(current_dir, "config", "exp_7.yaml")
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return
    
    print_section("CONFIGURATION")
    print(f"Config file: {config_path}")
    
    # Load config to get test path
    config = AppConfig(config_path).config
    pipeline_config = config.get('pipeline', {})
    data_config = pipeline_config.get('data', {})
    test_path = data_config.get('test_path', './dataset/public_test.json')
    
    # Resolve path
    if not os.path.isabs(test_path):
        test_path = os.path.join(current_dir, test_path)
    
    print(f"Test data path: {test_path}")
    
    # Load test data
    if not os.path.exists(test_path):
        print(f"❌ Test file not found: {test_path}")
        return
    
    with open(test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    if not test_data:
        print("❌ Test data is empty")
        return
    
    # Get first sample
    sample = test_data[0]
    qid = sample.get('qid')
    question = sample.get('question', '')
    
    print_section("SAMPLE QUERY")
    print(f"QID: {qid}")
    print(f"Question: {question}")
    print(f"Total samples in test file: {len(test_data)}")
    
    # Initialize pipeline
    print_section("INITIALIZING PIPELINE")
    try:
        pipeline = ExperimentPipeline(config_path)
        print("✅ Pipeline initialized successfully")
        
        # Check components
        print(f"\nComponents status:")
        print(f"  - Retriever: {'✅' if pipeline.retriever else '❌'}")
        print(f"  - Reranker (Cross-Encoder): {'✅' if pipeline.reranker else '❌'}")
        print(f"  - Qwen Reranker: {'✅' if pipeline.qwen_reranker else '❌'}")
        print(f"  - Legal Corpus Dict: {len(pipeline.legal_corpus_dict)} articles loaded")
        
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 1: Retrieve
    print_section("STEP 1: RETRIEVER OUTPUT")
    try:
        retrieved_results = pipeline.retriever.retrieve(question, top_k=pipeline.retriever_top_k)
        print(f"✅ Retrieved {len(retrieved_results)} results (top_k={pipeline.retriever_top_k})")
        
        print(f"\nTop 10 retrieved results:")
        for i, (aid, score) in enumerate(retrieved_results[:10]):
            print(f"  [{i+1}] AID: {aid}, Score: {score:.4f}")
        
        retrieved_aids = [str(aid) for aid, _ in retrieved_results]
        print(f"\nRetrieved AIDs (first 20):")
        print_list_items(retrieved_aids[:20])
        
    except Exception as e:
        print(f"❌ Retrieval failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: Get documents for reranking
    print_section("STEP 2: GET DOCUMENTS FOR RERANKING")
    try:
        filtered_docs = pipeline._get_reranker_documents(retrieved_aids)
        print(f"✅ Found {len(filtered_docs)} documents for reranking")
        
        if filtered_docs:
            print(f"\nSample document (first one):")
            first_doc = filtered_docs[0]
            print(f"  AID: {first_doc.get('aid')}")
            content_preview = first_doc.get('content', first_doc.get('text', ''))[:200]
            print(f"  Content preview: {content_preview}...")
        
    except Exception as e:
        print(f"❌ Failed to get documents: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Cross-Encoder Reranker (if enabled)
    if pipeline.reranker:
        print_section("STEP 3: CROSS-ENCODER RERANKER OUTPUT")
        try:
            intermediate_k = min(30, len(filtered_docs))
            reranked_results = pipeline.reranker.rerank(question, filtered_docs, top_k=intermediate_k)
            print(f"✅ Reranked {len(reranked_results)} results (intermediate_k={intermediate_k})")
            
            print(f"\nTop 10 reranked results:")
            for i, (aid, score) in enumerate(reranked_results[:10]):
                print(f"  [{i+1}] AID: {aid}, Score: {score:.4f}")
            
        except Exception as e:
            print(f"❌ Reranking failed: {e}")
            import traceback
            traceback.print_exc()
            reranked_results = [(d['aid'], 1.0) for d in filtered_docs]
    else:
        print_section("STEP 3: CROSS-ENCODER RERANKER")
        print("⚠️  Cross-Encoder reranker not enabled, using retriever results")
        reranked_results = [(d['aid'], 1.0) for d in filtered_docs]
    
    # Step 4: Qwen Reranker (if enabled)
    if pipeline.qwen_reranker:
        print_section("STEP 4: QWEN LLM RERANKER OUTPUT")
        try:
            num_candidates_for_qwen = min(15, len(reranked_results))
            qwen_candidates = reranked_results[:num_candidates_for_qwen]
            
            print(f"✅ Preparing {len(qwen_candidates)} candidates for Qwen")
            print(f"\nCandidates for Qwen:")
            for i, (aid, score) in enumerate(qwen_candidates):
                print(f"  [{i+1}] AID: {aid}, Score: {score:.4f}")
            
            # Get documents for Qwen
            top_aids = [str(aid) for aid, _ in qwen_candidates]
            qwen_docs = []
            for aid in top_aids:
                content = pipeline.legal_corpus_dict.get(aid, '')
                if not content:
                    for doc in filtered_docs:
                        if str(doc.get('aid', '')) == aid:
                            content = doc.get('content', doc.get('text', ''))
                            break
                
                if content:
                    qwen_docs.append({
                        "aid": aid,
                        "content": content,
                        "text": content
                    })
            
            print(f"\n✅ Prepared {len(qwen_docs)} documents for Qwen")
            if qwen_docs:
                print(f"\nSample document for Qwen:")
                sample_doc = qwen_docs[0]
                print(f"  AID: {sample_doc.get('aid')}")
                content_preview = sample_doc.get('content', '')[:300]
                print(f"  Content preview: {content_preview}...")
            
            # Run Qwen reranker
            if qwen_docs:
                documents_dict = pipeline.qwen_reranker.build_documents_dict(qwen_docs, id_key="aid")
                qwen_results = pipeline.qwen_reranker.rerank(
                    question,
                    qwen_candidates,
                    documents_dict
                )
                
                print(f"\n✅ Qwen reranked {len(qwen_results)} results")
                print(f"\nQwen selected articles:")
                for i, (aid, score) in enumerate(qwen_results):
                    print(f"  [{i+1}] AID: {aid}, Score: {score:.4f}")
                
                final_aids = [str(aid) for aid, _ in qwen_results]
            else:
                print("⚠️  No documents prepared for Qwen, using previous results")
                final_aids = [str(aid) for aid, _ in reranked_results[:pipeline.reranker_top_k]]
        except Exception as e:
            print(f"❌ Qwen reranking failed: {e}")
            import traceback
            traceback.print_exc()
            final_aids = [str(aid) for aid, _ in reranked_results[:pipeline.reranker_top_k]]
    else:
        print_section("STEP 4: QWEN LLM RERANKER")
        print("⚠️  Qwen reranker not enabled")
        final_aids = [str(aid) for aid, _ in reranked_results[:pipeline.reranker_top_k]]
    
    # Step 5: Final Output
    print_section("STEP 5: FINAL OUTPUT")
    print(f"✅ Final result: {len(final_aids)} articles")
    print(f"\nFinal AIDs:")
    for i, aid in enumerate(final_aids):
        print(f"  [{i+1}] {aid}")
    
    # Also test the full pipeline
    print_section("FULL PIPELINE TEST")
    try:
        full_result = pipeline.run_query(question)
        print(f"✅ Full pipeline returned {len(full_result)} articles")
        print(f"\nFull pipeline AIDs:")
        for i, aid in enumerate(full_result):
            print(f"  [{i+1}] {aid}")
        
        # Compare
        if full_result == final_aids[:len(full_result)]:
            print("\n✅ Full pipeline result matches step-by-step result")
        else:
            print("\n⚠️  Full pipeline result differs from step-by-step result")
            print(f"  Step-by-step: {final_aids[:len(full_result)]}")
            print(f"  Full pipeline: {full_result}")
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print_section("TEST COMPLETED", char="=")


if __name__ == "__main__":
    test_pipeline_with_sample()
import sys
import os
import json
import shutil
import warnings
from typing import List, Dict, Any

sys.path.append(os.getcwd())
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.embeddings.embedder import VietnameseEmbedder
from src.embeddings.chroma_index import ChromaIndex
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.ensemble import EnsembleRetriever
from src.utils.logger import setup_logger

logger = setup_logger("optimizer")


PATH_CORPUS = 'dataset/legal_corpus.json'

PATH_QA_DATA = 'dataset/qa_train.json'


PATH_TEMP_DB = "./data/chroma_db_temp_opt"


def load_json(path: str) -> Any:
    """Hàm helper để đọc file JSON an toàn."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Không tìm thấy file: {path}")
        return None
    except json.JSONDecodeError:
        logger.error(f"File {path} bị lỗi format JSON.")
        return None

def prepare_chunks(raw_data: List[Dict]) -> List[Dict[str, Any]]:
    """
    Chuyển đổi dữ liệu luật gốc (Hierarchical) thành dạng phẳng (Flat Chunks).
    Format đầu ra: [{'id': ..., 'text': ..., 'metadata': ...}]
    """
    chunks = []
    logger.info("Đang xử lý và làm phẳng dữ liệu (Flattening Corpus)...")

    for law in raw_data:
        law_id = law.get('law_id', 'unknown')
        for article in law.get('content', []):
            text = article.get('content_Article', '').strip()

            if not text:
                continue

            aid = article.get('aid', 'unknown')

            unique_id = f"{law_id}_{aid}"

            chunks.append({
                "id": unique_id,
                "text": text,
                "metadata": {
                    "law_id": law_id,
                    "aid": aid,
                    "doc_id": unique_id
                }
            })
    return chunks

def run_grid_search():
    print("\n" + "="*60)
    logger.info("BẮT ĐẦU TỐI ƯU TRỌNG SỐ (GRID SEARCH) - REAL DATA MODE")
    print("="*60 + "\n")

    if not os.path.exists(PATH_CORPUS) or not os.path.exists(PATH_QA_DATA):
        logger.error("Vui lòng kiểm tra lại đường dẫn file dữ liệu trong phần CẤU HÌNH.")
        logger.error(f"   - Corpus Path: {PATH_CORPUS}")
        logger.error(f"   - QA Data Path: {PATH_QA_DATA}")
        return

    raw_corpus = load_json(PATH_CORPUS)
    qa_data = load_json(PATH_QA_DATA)

    if not raw_corpus or not qa_data:
        return

    chunks = prepare_chunks(raw_corpus)
    logger.info(f"Dữ liệu sẵn sàng: {len(chunks)} chunks văn bản.")
    logger.info(f"Tập Test: {len(qa_data)} câu hỏi.")

    if os.path.exists(PATH_TEMP_DB):
        shutil.rmtree(PATH_TEMP_DB)

    logger.info("Đang khởi tạo các mô hình (Việc này có thể mất thời gian)...")

    embedder = VietnameseEmbedder()

    logger.info("Building BM25 Index...")
    bm25 = BM25Retriever(embedder=None)
    bm25.fit(chunks)

    logger.info("Building Dense Index (ChromaDB)...")
    chroma_index = ChromaIndex(persist_directory=PATH_TEMP_DB, collection_name="opt_temp")
    dense = DenseRetriever(chroma_index=chroma_index, embedder=embedder)

    dense.build_index(chunks, batch_size=32)


    ensemble = EnsembleRetriever(bm25, dense, weights=(0.5, 0.5))


    best_score = -1.0
    best_weights = (0.5, 0.5)

    steps = [round(x * 0.1, 1) for x in range(11)]

    print("\n" + "-"*55)
    print(f"{'w1 (BM25)':<12} | {'w2 (Dense)':<12} | {'Accuracy (Top-1)':<15}")
    print("-" * 55)

    for w1 in steps:
        w2 = round(1.0 - w1, 1)
        ensemble.set_weights((w1, w2))

        correct_count = 0
        total_queries = len(qa_data)

        for item in qa_data:
            query = item.get('query', '')
            correct_id = item.get('correct_id', '')

            if not query or not correct_id:
                continue

            # Retrieve Top 1
            results = ensemble.retrieve(query, top_k=1)

            # Logic tính điểm: Top 1 ID khớp với ID đúng
            if results and results[0][0] == correct_id:
                correct_count += 1

        # Tính Accuracy
        score = correct_count / total_queries if total_queries > 0 else 0.0
        print(f"{w1:<12} | {w2:<12} | {score:.4f}")

        # Cập nhật kết quả tốt nhất
        if score >= best_score:
            best_score = score
            best_weights = (w1, w2)

    # 4. KẾT LUẬN & DỌN DẸP
    print("-" * 55)
    print(f"\nKẾT QUẢ TỐI ƯU NHẤT:")
    print(f"   ► Trọng số: BM25 = {best_weights[0]}, Dense = {best_weights[1]}")
    print(f"   ► Độ chính xác: {best_score:.4f}")
    print("\nHãy dùng cặp số này để cập nhật vào file config của hệ thống Production.")

    if os.path.exists(PATH_TEMP_DB):
        shutil.rmtree(PATH_TEMP_DB)
        logger.info("Đã xóa DB tạm thời.")

if __name__ == "__main__":
    run_grid_search()
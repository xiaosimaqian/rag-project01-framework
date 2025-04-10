import os
import chromadb
from chromadb.config import Settings
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
from utils.config import CHROMA_CONFIG
from langchain_ollama import OllamaEmbeddings

logger = logging.getLogger(__name__)

class ChromaService:
    def __init__(self, config: Optional[Any] = None):
        """
        初始化 ChromaService
        
        参数:
            config: 可选的向量数据库配置对象
        """
        os.makedirs(CHROMA_CONFIG["uri"], exist_ok=True)
        # 使用 Ollama 的嵌入模型
        self.embeddings = OllamaEmbeddings(
            model="bge-m3:latest",
            base_url="http://localhost:11434"
        )
        # 使用新的客户端配置方式
        self.client = chromadb.PersistentClient(
            path=CHROMA_CONFIG["uri"]
        )
        self.config = config

    def search(self, query: str, collection_name: str, n_results: int = 3, where: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        try:
            # 使用 Ollama 创建查询向量
            query_embedding = self.embeddings.embed_query(query)
            
            collection = self.client.get_collection(collection_name)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where
            )
            
            # 将结果转换为标准格式
            processed_results = []
            if results["documents"] and len(results["documents"]) > 0:
                for i, doc in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    processed_results.append({
                        "document": doc,
                        "distance": float(results["distances"][0][i]) if results["distances"] else 0.0,
                        "metadata": {
                            "source": metadata.get("document_name", ""),
                            "page": metadata.get("page_number", ""),
                            "chunk": metadata.get("chunk_id", ""),
                            "total_chunks": metadata.get("total_chunks", ""),
                            "page_range": metadata.get("page_range", ""),
                            "word_count": metadata.get("word_count", 0),
                            "embedding_provider": metadata.get("embedding_provider", ""),
                            "embedding_model": metadata.get("embedding_model", ""),
                            "embedding_timestamp": metadata.get("embedding_timestamp", "")
                        }
                    })
            
            return processed_results
        
        except Exception as e:
            logger.error(f"Error searching in Chroma: {str(e)}")
            raise

    def index_embeddings(self, embeddings_data: Dict[str, Any], index_mode: str) -> Dict[str, Any]:
        try:
            # 准备集合名称
            filename = embeddings_data.get("filename", "")
            base_name = filename.replace('.pdf', '') if filename else "doc"
            embedding_provider = embeddings_data.get("embedding_provider", "unknown")
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            collection_name = f"{base_name}_{embedding_provider}_{timestamp}"
            
            # 创建集合
            collection = self.client.create_collection(
                name=collection_name,
                metadata={
                    **CHROMA_CONFIG["collection_metadata"],
                    "index_mode": index_mode
                }
            )
            
            # 准备数据
            embeddings = []
            documents = []
            metadatas = []
            ids = []
            
            for idx, item in enumerate(embeddings_data["embeddings"]):
                embeddings.append(item["embedding"])
                documents.append(item["metadata"]["content"])
                metadatas.append({
                    "document_name": item["metadata"].get("document_name", ""),
                    "chunk_id": item["metadata"].get("chunk_id", idx),
                    "page_number": item["metadata"].get("page_number", ""),
                    "page_range": item["metadata"].get("page_range", ""),
                    "word_count": item["metadata"].get("word_count", 0)
                })
                ids.append(str(idx))
            
            # 添加数据到集合
            collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            return {
                "collection_name": collection_name,
                "index_size": len(embeddings)
            }
            
        except Exception as e:
            logger.error(f"Error indexing to Chroma: {str(e)}")
            raise

    def list_collections(self) -> List[Dict[str, Any]]:
        try:
            collections = self.client.list_collections()
            return [
                {
                    "id": collection.name,
                    "name": collection.name,
                    "count": collection.count()
                }
                for collection in collections
            ]
        except Exception as e:
            logger.error(f"Error listing Chroma collections: {str(e)}")
            raise

    def delete_collection(self, collection_name: str) -> bool:
        try:
            self.client.delete_collection(collection_name)
            return True
        except Exception as e:
            logger.error(f"Error deleting Chroma collection: {str(e)}")
            return False

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        try:
            collection = self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "num_entities": collection.count(),
                "metadata": collection.metadata
            }
        except Exception as e:
            logger.error(f"Error getting Chroma collection info: {str(e)}")
            raise

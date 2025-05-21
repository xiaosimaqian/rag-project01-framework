from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from pymilvus import connections, Collection, utility
from services.embedding_service import EmbeddingService, EmbeddingConfig
from services.vector_store_service import VectorStoreService, VectorDBConfig
from utils.config import VectorDBProvider, MILVUS_CONFIG, CHROMA_CONFIG
from fastapi import FastAPI, HTTPException
import os
import json
from pathlib import Path
import chromadb
from chromadb.config import Settings
from langchain.embeddings import OllamaEmbeddings

logger = logging.getLogger(__name__)

class SearchService:
    """
    搜索服务类，提供向量搜索功能
    """
    
    def __init__(self):
        """
        初始化搜索服务
        """
        self._chroma_client = None
        self._embeddings = OllamaEmbeddings(
            model="bge-m3:latest",
            base_url="http://localhost:11434"
        )
    
    def _init_chroma_client(self):
        """初始化 Chroma 客户端"""
        if self._chroma_client is None:
            self._chroma_client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=CHROMA_CONFIG.get("uri", "03-vector-store/chroma_db")
            ))
    
    def search(self, query: str, collection_name: str, top_k: int = 5, threshold: float = 0.7) -> Dict:
        """
        搜索相似文档
        
        参数:
            query: 查询文本
            collection_name: 集合名称
            top_k: 返回结果数量
            threshold: 相似度阈值
            
        返回:
            搜索结果
        """
        try:
            # 获取查询向量
            query_embedding = self._embeddings.embed_query(query)
            
            # 初始化 Chroma 客户端
            self._init_chroma_client()
            collection = self._chroma_client.get_collection(collection_name)
            
            # 执行搜索
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            # 处理结果
            processed_results = []
            if results["documents"] and len(results["documents"]) > 0:
                for i, doc in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    distance = float(results["distances"][0][i]) if results["distances"] else 0.0
                    
                    # 应用相似度阈值
                    if distance <= threshold:
                        processed_results.append({
                            "document": doc,
                            "distance": distance,
                            "metadata": metadata
                        })
            
            return {
                "results": processed_results,
                "total": len(processed_results)
            }
            
        except Exception as e:
            logger.error(f"搜索失败: {str(e)}")
            raise
    
    def list_collections(self) -> List[Dict[str, Any]]:
        """
        列出所有可用的集合
        """
        try:
            self._init_chroma_client()
            collections = []
            for collection in self._chroma_client.list_collections():
                collections.append({
                    "id": collection.id,
                    "name": collection.name,
                    "count": collection.count(),
                    "metadata": collection.metadata
                })
            return collections
        except Exception as e:
            logger.error(f"列出集合时出错: {e}")
            raise

    def get_document_embedding_config(self, collection_name: str) -> EmbeddingConfig:
        """获取文档的嵌入配置"""
        try:
            # 从集合名称中提取嵌入模型信息
            # 假设集合名称格式为: {doc_name}_by_{chunking_method}_{timestamp}
            parts = collection_name.split('_by_')
            if len(parts) >= 2:
                # 使用默认配置
                return EmbeddingConfig(
                    provider="ollama",
                    model_name="bge-m3:latest"
                )
            else:
                raise ValueError(f"无法从集合名称 {collection_name} 中提取嵌入配置信息")
        except Exception as e:
            logger.error(f"获取文档嵌入配置失败: {e}")
            # 返回默认配置
            return EmbeddingConfig(
                provider="ollama",
                model_name="bge-m3:latest"
            )

    def get_providers(self) -> List[Dict[str, str]]:
        """
        获取支持的向量数据库列表
        
        Returns:
            List[Dict[str, str]]: 支持的向量数据库提供商列表
        """

        try:
            providers = [
                {"id": "milvus", "name": "Milvus"},
                {"id": "chroma", "name": "Chroma"}
            ]
            return providers

        except Exception as e:
            logger.error(f"Error getting providers: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def save_search_results(self, query: str, collection_id: str, results: List[Dict[str, Any]]) -> str:
        """
        保存搜索结果到JSON文件
        
        Args:
            query (str): 搜索查询文本
            collection_id (str): 集合ID
            results (List[Dict[str, Any]]): 搜索结果列表
            
        Returns:
            str: 保存文件的路径
            
        Raises:
            Exception: 保存文件时发生错误
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            # 使用集合ID的基础名称（去掉路径相关字符）
            collection_base = os.path.basename(collection_id)
            filename = f"search_{collection_base}_{timestamp}.json"
            filepath = os.path.join(self.search_results_dir, filename)
            
            search_data = {
                "query": query,
                "collection_id": collection_id,
                "timestamp": datetime.now().isoformat(),
                "results": results
            }
            
            logger.info(f"Saving search results to: {filepath}")
            
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(search_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Successfully saved search results to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving search results: {str(e)}")
            raise
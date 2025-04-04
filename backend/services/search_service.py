from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from pymilvus import connections, Collection, utility
from services.embedding_service import EmbeddingService
from utils.config import VectorDBProvider, MILVUS_CONFIG, CHROMA_CONFIG
from fastapi import FastAPI, HTTPException
import os
import json
from services.chroma_service import ChromaService
from langchain_ollama import OllamaEmbeddings

logger = logging.getLogger(__name__)

class SearchService:
    """
    搜索服务类，负责向量数据库的连接和向量搜索功能
    提供集合列表查询、向量相似度搜索和搜索结果保存等功能
    """
    def __init__(self):
        """
        初始化搜索服务
        创建嵌入服务实例，设置Milvus连接URI，初始化搜索结果保存目录
        """
        self.embedding_service = EmbeddingService()
        self.milvus_uri = MILVUS_CONFIG["uri"]
        self.chroma_service = None
        self.search_results_dir = "04-search-results"
        os.makedirs(self.search_results_dir, exist_ok=True)
    def _init_chroma_service(self):
        """
        初始化 ChromaService
        """
        if self.chroma_service is None:
            self.chroma_service = ChromaService()

    async def search(self, 
                query: str, 
                collection_id: str, 
                provider: str = VectorDBProvider.MILVUS.value,
                top_k: int = 3, 
                threshold: float = 0.7,
                word_count_threshold: int = 20,
                save_results: bool = False) -> Dict[str, Any]:
        """
        执行向量搜索
        
        Args:
            query (str): 搜索查询文本
            collection_id (str): 要搜索的集合ID
            provider (str): 向量数据库提供商，默认为Milvus
            top_k (int): 返回的最大结果数量，默认为3
            threshold (float): 相似度阈值，低于此值的结果将被过滤，默认为0.7
            word_count_threshold (int): 文本字数阈值，低于此值的结果将被过滤，默认为20
            save_results (bool): 是否保存搜索结果，默认为False
            
        Returns:
            Dict[str, Any]: 包含搜索结果的字典，如果保存结果则包含保存路径
        """
        try:
            # 添加参数日志
            logger.info(f"Search parameters:")
            logger.info(f"- Query: {query}")
            logger.info(f"- Collection ID: {collection_id}")
            logger.info(f"- Provider: {provider}")
            logger.info(f"- Top K: {top_k}")
            logger.info(f"- Threshold: {threshold}")
            logger.info(f"- Word Count Threshold: {word_count_threshold}")
            logger.info(f"- Save Results: {save_results}")

            if provider == VectorDBProvider.CHROMA.value:
                # 使用 Chroma 进行搜索
                self._init_chroma_service()
                results = self.chroma_service.search(
                    query=query,
                    collection_name=collection_id,
                    n_results=top_k,
                    where={"word_count": {"$gte": word_count_threshold}}
                )
                
                # 处理 Chroma 的搜索结果
                processed_results = []
                for doc in results:
                    if doc["distance"] >= threshold:  # Chroma 返回的是相似度，不需要转换
                        processed_results.append({
                            "text": doc["document"],
                            "score": float(doc["distance"]),
                            "metadata": {
                                "source": doc["metadata"].get("source", ""),
                                "page": doc["metadata"].get("page", ""),
                                "chunk": doc["metadata"].get("chunk", ""),
                                "total_chunks": doc["metadata"].get("total_chunks", ""),
                                "page_range": doc["metadata"].get("page_range", ""),
                                "word_count": doc["metadata"].get("word_count", 0),
                                "embedding_provider": doc["metadata"].get("embedding_provider", ""),
                                "embedding_model": doc["metadata"].get("embedding_model", ""),
                                "embedding_timestamp": doc["metadata"].get("embedding_timestamp", "")
                            }
                        })
            else:
                # 使用 Milvus 进行搜索
                connections.connect(
                    alias="default",
                    uri=self.milvus_uri
                )
                
                try:
                    # 检查集合是否存在
                    if not utility.has_collection(collection_id):
                        raise ValueError(f"Collection '{collection_id}' does not exist")
                    
                    collection = Collection(collection_id)
                    collection.load()
                    
                    # 从collection中读取embedding配置
                    sample_entity = collection.query(
                        expr="id >= 0", 
                        output_fields=["embedding_provider", "embedding_model"],
                        limit=1
                    )
                    if not sample_entity:
                        raise ValueError(f"Collection {collection_id} is empty")
                    
                    # 使用collection中存储的配置创建查询向量
                    query_embedding = self.embedding_service.create_single_embedding(
                        query,
                        provider=sample_entity[0]["embedding_provider"],
                        model=sample_entity[0]["embedding_model"]
                    )
                    
                    # 执行搜索
                    search_params = {
                        "metric_type": "COSINE",
                        "params": {"nprobe": 10}
                    }
                    
                    results = collection.search(
                        data=[query_embedding],
                        anns_field="vector",
                        param=search_params,
                        limit=top_k,
                        expr=f"word_count >= {word_count_threshold}",
                        output_fields=[
                            "content",
                            "document_name",
                            "chunk_id",
                            "total_chunks",
                            "word_count",
                            "page_number",
                            "page_range",
                            "embedding_provider",
                            "embedding_model",
                            "embedding_timestamp"
                        ]
                    )
                    
                    # 处理 Milvus 的搜索结果
                    processed_results = []
                    for hits in results:
                        for hit in hits:
                            if hit.score >= threshold:
                                processed_results.append({
                                    "text": hit.entity.content,
                                    "score": float(hit.score),
                                    "metadata": {
                                        "source": hit.entity.document_name,
                                        "page": hit.entity.page_number,
                                        "chunk": hit.entity.chunk_id,
                                        "total_chunks": hit.entity.total_chunks,
                                        "page_range": hit.entity.page_range,
                                        "embedding_provider": hit.entity.embedding_provider,
                                        "embedding_model": hit.entity.embedding_model,
                                        "embedding_timestamp": hit.entity.embedding_timestamp
                                    }
                                })
                finally:
                    connections.disconnect("default")

            response_data = {"results": processed_results}
            
            # 保存搜索结果
            if save_results and processed_results:
                filepath = self.save_search_results(query, collection_id, processed_results)
                response_data["saved_filepath"] = filepath
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error performing search: {e}")
            raise

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

    def list_collections(self, provider: str = VectorDBProvider.MILVUS.value) -> List[Dict[str, Any]]:
        """
        获取指定向量数据库中的所有集合
        
        Args:
            provider (str): 向量数据库提供商，默认为Milvus
            
        Returns:
            List[Dict[str, Any]]: 集合信息列表，包含id、名称和实体数量
            
        Raises:
            Exception: 连接或查询集合时发生错误
        """
        try:
            if provider == VectorDBProvider.CHROMA.value:
                # 使用 Chroma 获取集合列表
                self._init_chroma_service()
                return self.chroma_service.list_collections()
            else:
                # 使用 Milvus 获取集合列表
                connections.connect(
                    alias="default",
                    uri=self.milvus_uri
                )
                
                try:
                    collections = []
                    collection_names = utility.list_collections()
                    
                    for name in collection_names:
                        try:
                            collection = Collection(name)
                            collections.append({
                                "id": name,
                                "name": name,
                                "count": collection.num_entities
                            })
                        except Exception as e:
                            logger.error(f"Error getting info for collection {name}: {str(e)}")
                    
                    return collections
                    
                finally:
                    connections.disconnect("default")
            
        except Exception as e:
            logger.error(f"Error listing collections: {str(e)}")
            raise

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
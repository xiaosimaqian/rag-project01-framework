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
        self.vector_store_service = VectorStoreService()
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

    async def search(
        self,
        query: str,
        collection_name: str,
        provider: str,
        top_k: int = 3,
        threshold: float = 0.7,
        word_count_threshold: int = 100
    ) -> List[Dict[str, Any]]:
        """
        执行相似度搜索
        """
        try:
            logger.info(f"开始搜索，提供商: {provider}, 集合: {collection_name}")
            
            # 获取原始文档使用的嵌入配置
            embedding_config = self.embedding_service.get_document_embedding_config(collection_name)
            
            # 创建向量数据库配置
            vector_db_config = VectorDBConfig(
                provider=provider,
                index_mode="flat"  # 使用默认的 flat 索引模式进行搜索
            )
            
            # 验证配置
            if not vector_db_config.uri:
                raise ValueError(f"向量数据库 URI 配置不正确，提供商: {provider}")
            
            logger.info(f"使用向量数据库配置: {vector_db_config.__dict__}")
            
            # 生成查询向量
            logger.info(f"使用 {embedding_config.provider} - {embedding_config.model_name} 生成查询向量")
            query_embedding = await self.embedding_service.create_single_embedding(
                text=query,
                config=embedding_config
            )
            
            # 执行向量搜索
            logger.info(f"在集合 {collection_name} 中执行向量搜索")
            results = self.vector_store_service.search(
                query_embedding=query_embedding,
                collection_name=collection_name,
                provider=provider,
                config=vector_db_config,
                top_k=top_k,
                threshold=threshold
            )
            
            # 过滤结果
            if word_count_threshold > 0:
                filtered_results = [
                    r for r in results 
                    if r["metadata"].get("word_count", 0) >= word_count_threshold
                ]
                logger.info(f"基于词数阈值过滤结果，从 {len(results)} 减少到 {len(filtered_results)}")
                results = filtered_results
            
            return results
            
        except Exception as e:
            logger.error(f"搜索操作出错: {e}", exc_info=True)
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
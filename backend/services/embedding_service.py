import os
import dotenv
dotenv.load_dotenv()
import json
from datetime import datetime
from enum import Enum
import boto3
import numpy as np  # 添加这个导入
from langchain_community.embeddings import BedrockEmbeddings, OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# 添加 CompactJSONEncoder 类定义
class CompactJSONEncoder(json.JSONEncoder):
    """
    自定义 JSON 编码器，用于处理 NumPy 数组和其他特殊类型的序列化
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class EmbeddingProvider(str, Enum):
    """
    嵌入提供商枚举类，定义支持的嵌入模型提供商
    """
    OPENAI = "openai"
    BEDROCK = "bedrock"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"

class EmbeddingConfig:
    """
    嵌入配置类，用于存储嵌入模型的配置信息
    """
    def __init__(self, provider: str, model_name: str):
        """
        初始化嵌入配置
        
        参数:
            provider: 嵌入提供商名称
            model_name: 嵌入模型名称
        """
        self.provider = provider
        self.model_name = model_name
        self.aws_region = "ap-southeast-1"  # 可配置

class EmbeddingService:
    """
    嵌入服务类，提供创建和管理文本嵌入的功能
    """
    def __init__(self):
        """初始化嵌入服务，创建嵌入工厂实例"""
        self.embedding_factory = EmbeddingFactory()
        # 初始化默认的嵌入配置为 Ollama
        self.embedding_config = EmbeddingConfig(
            provider=EmbeddingProvider.OLLAMA.value,
            model_name="bge-m3:latest"
        )

    def create_embeddings(
            self,
            chunks: List[Dict[str, Any]],
            metadata: Dict[str, Any],
            doc_timestamp: str = None,
            embedding_config: EmbeddingConfig = None
        ) -> Dict[str, Any]:
        """创建文本块的嵌入向量"""
        try:
            if not chunks:
                raise ValueError("没有可处理的文本块")
            
            # 如果没有传入时间戳，则生成新的
            if not doc_timestamp:
                doc_timestamp = f"doc_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # 生成嵌入阶段的时间戳
            emb_timestamp = f"emb_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # 使用传入的配置或默认配置
            config = embedding_config or self.embedding_config
            
            # 准备嵌入结果
            results = []
            for chunk in chunks:
                # 创建嵌入向量
                embedding = self.embedding_factory.create_embedding_function(config).embed_query(chunk["content"])
                
                # 准备元数据
                chunk_metadata = {
                    "content": chunk["content"],
                    "chunk_id": chunk.get("chunk_id"),
                    "page_number": chunk.get("page_number"),
                    "word_count": len(chunk["content"].split()),
                    "embedding_provider": config.provider,
                    "embedding_model": config.model_name,
                    "doc_timestamp": doc_timestamp,
                    "emb_timestamp": emb_timestamp
                }
                
                # 合并文档级别的元数据
                chunk_metadata.update(metadata)
                
                results.append({
                    "embedding": embedding,
                    "metadata": chunk_metadata
                })
            
            # 准备返回的数据结构
            return {
                "embeddings": results,
                "metadata": {
                    "document_name": metadata.get("document_name", ""),
                    "embedding_provider": config.provider,
                    "embedding_model": config.model_name,
                    "created_at": datetime.now().isoformat(),
                    "total_vectors": len(results),
                    "vector_dimension": len(embedding) if results else 0,
                    "doc_timestamp": doc_timestamp,
                    "emb_timestamp": emb_timestamp
                },
                "doc_timestamp": doc_timestamp,
                "emb_timestamp": emb_timestamp
            }
            
        except Exception as e:
            logger.error(f"创建嵌入向量失败: {str(e)}", exc_info=True)
            raise

    def save_embeddings(self, doc_name: str, embeddings: Dict[str, Any]) -> str:
        """
        保存嵌入向量到文件
        
        参数:
            doc_name: 文档名称
            embeddings: 嵌入向量数据
            
        返回:
            保存的文件路径
        """
        try:
            # 构建保存目录
            save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "02-embedded-docs")
            os.makedirs(save_dir, exist_ok=True)
            
            # 生成文件名，确保不会重复添加时间戳
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            # 如果文件名已经包含时间戳，则不再添加
            if '_' in doc_name and doc_name.split('_')[-1].isdigit() and len(doc_name.split('_')[-1]) == 14:
                filename = f"{doc_name}.json"
            else:
                filename = f"{doc_name}_{timestamp}.json"
            file_path = os.path.join(save_dir, filename)
            
            # 确保 embeddings 包含所有必要字段
            if not isinstance(embeddings, dict):
                raise ValueError("embeddings must be a dictionary")
                
            if "embeddings" not in embeddings:
                raise ValueError("embeddings must contain an 'embeddings' key")
                
            # 添加元数据
            embeddings["metadata"] = {
                "filename": filename,
                "timestamp": timestamp,
                "total_vectors": len(embeddings["embeddings"]),
                "document_name": doc_name,
                "embedding_provider": embeddings.get("metadata", {}).get("embedding_provider", "unknown"),
                "embedding_model": embeddings.get("metadata", {}).get("embedding_model", "unknown"),
                "created_at": datetime.now().isoformat(),
                "vector_dimension": embeddings.get("metadata", {}).get("vector_dimension", 0)
            }
            
            # 保存数据
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(embeddings, f, ensure_ascii=False, indent=2, cls=CompactJSONEncoder)
                
            logger.info(f"Successfully saved embeddings to {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving embeddings: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to save embeddings: {str(e)}")

    async def create_single_embedding(self, text: str, config: EmbeddingConfig) -> list:
        """
        创建单个文本的嵌入向量
        
        参数:
            text: 需要嵌入的文本
            config: 嵌入配置对象
            
        返回:
            嵌入向量列表
        """
        embedding_function = self.embedding_factory.create_embedding_function(config)
        return embedding_function.embed_query(text)

    def get_document_embedding_config(self, collection_name: str) -> EmbeddingConfig:
        """
        从已存在的文档中获取嵌入配置
        
        参数:
            collection_name: 集合名称
            
        返回:
            嵌入配置对象
            
        异常:
            ValueError: 当找不到匹配的嵌入配置时抛出
        """
        try:
            # 移除 'collection_' 前缀，并分割后面的部分
            doc_name = collection_name.replace('collection_', '').split('_')[0]
            
            # 查找对应的embedding文件
            embedded_docs_dir = "02-embedded-docs"
            for filename in os.listdir(embedded_docs_dir):
                if filename.endswith('.json'):
                    with open(os.path.join(embedded_docs_dir, filename), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # 检查文档名称是否匹配
                        if doc_name in data.get("filename", ""):
                            logger.info(f"找到匹配的嵌入配置：{data.get('embedding_provider')} - {data.get('embedding_model')}")
                            return EmbeddingConfig(
                                provider=data.get("embedding_provider"),
                                model_name=data.get("embedding_model")
                            )
                            
            raise ValueError(f"未找到集合的匹配嵌入配置: {collection_name}")
        except Exception as e:
            logger.error(f"获取嵌入配置时出错: {str(e)}")
            raise ValueError(f"获取嵌入配置时出错: {str(e)}")

class EmbeddingFactory:
    """
    嵌入工厂类，负责创建不同提供商的嵌入函数
    """
    @staticmethod
    def create_embedding_function(config: EmbeddingConfig):
        """
        根据配置创建嵌入函数
        
        参数:
            config: 嵌入配置对象
            
        返回:
            嵌入函数对象
            
        异常:
            ValueError: 当提供商不支持时抛出
        """
        if config.provider == EmbeddingProvider.BEDROCK:
            bedrock_client = boto3.client(
                service_name='bedrock-runtime',
                region_name=config.aws_region,
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
            )
            return BedrockEmbeddings(
                client=bedrock_client,
                model_id=config.model_name
            )
            
        elif config.provider == EmbeddingProvider.OPENAI:
            return OpenAIEmbeddings(
                model=config.model_name,
                openai_api_key=os.getenv('OPENAI_API_KEY')
            )
            
        elif config.provider == EmbeddingProvider.HUGGINGFACE:
            return HuggingFaceEmbeddings(
                model_name=config.model_name
            )
        
        elif config.provider == EmbeddingProvider.OLLAMA:
            return OllamaEmbeddings(
                model="bge-m3:latest", 
                base_url="http://localhost:11434"
            )
            
        raise ValueError(f"Unsupported embedding provider: {config.provider}")
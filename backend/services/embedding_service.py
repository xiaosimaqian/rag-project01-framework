import os
import dotenv
dotenv.load_dotenv()
import json
from datetime import datetime
from enum import Enum
# import boto3 # Removed: Not using Bedrock
import numpy as np
# from langchain_community.embeddings import BedrockEmbeddings, OpenAIEmbeddings, HuggingFaceEmbeddings # Removed
# from sentence_transformers import SentenceTransformer # Removed: Focusing on Ollama
import ollama # Keep for Ollama
# import openai # Removed
# import google.generativeai as genai # Removed
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from typing import Dict, Any, List, Optional
import math

# 从 utils.config 导入配置变量 - 只导入 OLLAMA_CONFIG
from utils.config import OLLAMA_CONFIG #, OPENAI_CONFIG, GOOGLE_CONFIG, BEDROCK_CONFIG

logger = logging.getLogger(__name__)

# model_cache 可能不再那么重要，因为我们不预加载Ollama模型，但可以保留以备将来扩展
model_cache: Dict[tuple, Any] = {}

class CompactJSONEncoder(json.JSONEncoder):
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
    OLLAMA = "ollama"
    # SENTENCE_TRANSFORMERS = "sentence_transformers" # Removed
    # OPENAI = "openai" # Removed
    # GOOGLE = "google" # Removed
    # BEDROCK = "bedrock" # Removed

class EmbeddingConfig:
    def __init__(self, 
                 provider: str = EmbeddingProvider.OLLAMA.value, 
                 model_name: Optional[str] = None, # model_name 现在可选，会从OLLAMA_CONFIG获取默认
                 api_base: Optional[str] = None):
        
        if provider != EmbeddingProvider.OLLAMA.value:
            raise ValueError(f"当前配置仅支持 Ollama 提供商，收到: {provider}")
        
        self.provider = provider
        self.api_base = api_base if api_base else OLLAMA_CONFIG.get("api_base", "http://localhost:11434")
        self.model_name = model_name if model_name else OLLAMA_CONFIG.get("default_model", "bge-m3:latest")
        
        logger.info(f"EmbeddingConfig 初始化 (Ollama only): Model={self.model_name}, API_Base={self.api_base}")

# EmbeddingFactory 变得非常简单，甚至可以移除，因为 EmbeddingService 直接处理 Ollama
# class EmbeddingFactory:
#     @staticmethod
#     def create_embedding_function(config: EmbeddingConfig) -> Any:
#         if config.provider == EmbeddingProvider.OLLAMA.value:
#             # Ollama 客户端在需要时由 EmbeddingService 创建
#             return None # 或者返回一个ollama.Client的配置，但不直接创建实例
#         else:
#             raise ValueError(f"EmbeddingFactory: 不支持的嵌入提供商: {config.provider}")

class EmbeddingService:
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        if config is None:
            self.embedding_config = EmbeddingConfig() # 默认使用Ollama配置
        elif config.provider != EmbeddingProvider.OLLAMA.value:
            logger.warning(f"提供的配置提供商为 {config.provider}，但当前服务仅配置为Ollama。将使用Ollama默认配置。")
            self.embedding_config = EmbeddingConfig() # 强制使用Ollama配置
        else:
            self.embedding_config = config
        
        # 对于Ollama，_load_model 更多是准备配置而非加载大型模型到内存
        self.model_config = self._load_model() 
        
        # 移除尝试导入 embedding_tasks 的代码
        # self.embedding_tasks_ref = None 
        # try:
        #     from backend.main import embedding_tasks 
        #     self.embedding_tasks_ref = embedding_tasks
        # except ImportError:
        #     logger.warning("无法从 backend.main 导入 embedding_tasks。取消功能可能受限。")

    def _load_model(self) -> Dict[str, Any]:
        """准备Ollama模型配置"""
        # provider 总是 Ollama
        model_name = self.embedding_config.model_name
        api_base = self.embedding_config.api_base
        
        logger.info(f"准备 Ollama 模型配置: Model={model_name}, API_Base={api_base}")
        # 返回一个配置字典，_get_ollama_embedding 将使用它
        return {"name": model_name, "api_base": api_base, "provider": EmbeddingProvider.OLLAMA.value}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _get_ollama_embedding(self, text: str, config: Dict) -> List[float]:
        try:
            client_params = {}
            if config.get('api_base'):
                client_params['host'] = config['api_base']
            
            client = ollama.Client(**client_params)
            response = client.embeddings(model=config['name'], prompt=text)
            
            # 确保返回的是密集向量
            embedding = response["embedding"]
            if not isinstance(embedding, list):
                raise ValueError(f"嵌入向量必须是列表类型，但收到: {type(embedding)}")
            
            # 确保所有元素都是浮点数
            embedding = [float(x) for x in embedding]
            
            # 确保向量是密集的（没有 None 或 NaN 值）
            if any(x is None or math.isnan(x) for x in embedding):
                raise ValueError("嵌入向量包含 None 或 NaN 值")
            
            # 转换为numpy数组以确保类型一致性
            embedding = np.array(embedding, dtype=np.float32)
            
            # 检查是否有任何零值,如果有则转换为非零值
            zero_mask = embedding == 0
            if np.any(zero_mask):
                embedding[zero_mask] = 1e-10  # 使用一个很小的非零值
            
            # 转换回列表
            embedding = embedding.tolist()
            
            logger.info(f"生成的嵌入向量类型: {type(embedding)}, 维度: {len(embedding)}")
            return embedding
        except Exception as e:
            logger.error(f"通过Ollama获取嵌入失败 (model: {config['name']}, text length: {len(text)}): {e}", exc_info=True)
            if "max sequence length" in str(e).lower():
                 logger.warning(f"文本可能超过模型 {config['name']} 的最大序列长度。文本前100字符: '{text[:100]}...'")
            raise
            
    def create_embeddings(self, chunks: List[Dict[str, Any]], metadata: Dict[str, Any], task_id: Optional[str] = None, shared_embedding_tasks: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        # 移除尝试在此处导入 embedding_tasks 的代码
        # if self.embedding_tasks_ref is None:
        #     try:
        #         from backend.main import embedding_tasks
        #         self.embedding_tasks_ref = embedding_tasks
        #         logger.info(f"任务 {task_id}: 成功在 create_embeddings 中获取 embedding_tasks 引用。")
        #     except ImportError:
        #         logger.warning(f"任务 {task_id}: 仍然无法从 backend.main 导入 embedding_tasks。取消功能将不可用。")

        embeddings_list = []
        # provider 总是 Ollama, model_config 来自 self.model_config

        doc_metadata_for_log = {
            "document_name": metadata.get("document_name", "N/A"),
            "total_chunks_in_request": len(chunks)
        }
        logger.info(f"任务 {task_id}: 开始创建嵌入向量 (Ollama) - 文档: {doc_metadata_for_log['document_name']}, 模型: {self.model_config['name']}")

        for i, chunk_data in enumerate(chunks):
            if task_id and shared_embedding_tasks: # 使用传递的 shared_embedding_tasks
                task_info = shared_embedding_tasks.get(task_id)
                if task_info and task_info.get("cancel_requested"):
                    logger.info(f"任务 {task_id}: 检测到取消请求，正在中止嵌入创建过程于块 {i+1}/{len(chunks)}。")
                    # Embedding service 应该只检查取消状态，而不直接修改它。
                    # main.py 中的 cancel_embedding_task 负责修改状态为 "cancelled"。
                    # 如果需要，这里也可以更新任务状态，但要确保逻辑一致。
                    # 例如: shared_embedding_tasks[task_id][\"status\"] = \"cancelling_in_progress\"
                    # 但为了简单起见，这里只跳出循环，依赖 cancel_embedding_task 更新最终状态。
                    break # 跳出循环，不再处理更多块

            text = chunk_data.get("text", "")
            chunk_metadata = chunk_data.get("metadata", {})
            logger.debug(f"任务 {task_id}: 处理块 {i+1}/{len(chunks)} - Chunk ID: {chunk_metadata.get('chunk_id', 'N/A')}, Text length: {len(text)}")

            if not text.strip():
                logger.warning(f"任务 {task_id}: 第 {i+1} 个块的文本为空，跳过嵌入。元数据: {chunk_metadata}")
                embedding_vector = [] 
            else:
                try:
                    embedding_vector = self._get_ollama_embedding(text, self.model_config)
                except Exception as e:
                    logger.error(f"任务 {task_id}: 为块 {i+1} (Chunk ID: {chunk_metadata.get('chunk_id', 'N/A')}) 创建嵌入失败 (Ollama): {e}", exc_info=True)
                    embedding_vector = []
            
            # 合并文档级metadata，chunk级优先
            merged_metadata = {
                **metadata,  # 文档级
                **chunk_metadata  # chunk级
            }
            # 补全必要字段
            merged_metadata["chunk_id"] = merged_metadata.get("chunk_id", i + 1)
            merged_metadata["file_name"] = merged_metadata.get("file_name", metadata.get("file_name", metadata.get("document_name", "未知文件")))
            merged_metadata["page_number"] = merged_metadata.get("page_number", 1)
            embeddings_list.append({
                "chunk_id": merged_metadata["chunk_id"],
                "text": text,
                "embedding": embedding_vector,
                "metadata": merged_metadata
            })

        logger.info(f"任务 {task_id}: 嵌入创建完成 (Ollama)。共生成 {len(embeddings_list)} 个嵌入向量 (请求处理 {len(chunks)} 个块)。")
        return embeddings_list

    def save_embeddings(self, doc_name: str, embeddings: Dict[str, Any]) -> str:
        try:
            save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "02-embedded-docs")
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            # 文件名逻辑保持，因为可能之前有其他方式生成的文件
            if '_' in doc_name and doc_name.split('_')[-1].isdigit() and len(doc_name.split('_')[-1]) == 14:
                filename = f"{doc_name}.json"
            else:
                filename = f"{doc_name}_{timestamp}.json"
            file_path = os.path.join(save_dir, filename)
            
            if not isinstance(embeddings, dict):
                raise ValueError("embeddings must be a dictionary")
            if "embeddings" not in embeddings:
                raise ValueError("embeddings must contain an 'embeddings' key")
            
            current_provider = self.embedding_config.provider
            current_model = self.embedding_config.model_name
            current_dimension = 0
            if embeddings.get("embeddings") and isinstance(embeddings["embeddings"], list) and \
               len(embeddings["embeddings"]) > 0 and embeddings["embeddings"][0].get("embedding") and \
               isinstance(embeddings["embeddings"][0]["embedding"], list):
                current_dimension = len(embeddings["embeddings"][0]["embedding"])

            embeddings["metadata"] = {
                **embeddings.get("metadata", {}),
                "filename": filename,
                "timestamp": timestamp, 
                "total_vectors": len(embeddings.get("embeddings", [])),
                "document_name": doc_name, 
                "embedding_provider": current_provider,
                "embedding_model": current_model,
                "created_at": datetime.now().isoformat(),
                "vector_dimension": current_dimension
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(embeddings, f, ensure_ascii=False, indent=2, cls=CompactJSONEncoder)
                
            logger.info(f"Successfully saved embeddings to {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving embeddings: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to save embeddings: {str(e)}")

    # create_single_embedding 和 create_embedding 可以简化或调整为只使用 Ollama
    def create_embedding(self, text: str) -> List[float]:
        """创建单个文本的嵌入向量 (仅限Ollama)"""
        logger.info(f"为文本创建单个嵌入 (Ollama): 模型 {self.model_config['name']}")
        try:
            return self._get_ollama_embedding(text, self.model_config)
        except Exception as e:
            logger.error(f"使用 Ollama 创建单个嵌入失败: {e}")
            raise

    async def create_single_embedding(self, text: str, config: Optional[EmbeddingConfig] = None) -> list:
        """异步创建单个文本的嵌入向量 (仅限Ollama)"""
        # 如果未提供配置，则使用服务实例的默认配置
        active_config = config if config and config.provider == EmbeddingProvider.OLLAMA.value else self.embedding_config
        if active_config.provider != EmbeddingProvider.OLLAMA.value:
            raise ValueError("此方法当前配置仅支持Ollama的单个嵌入创建。")
        
        # _load_model 返回的是配置字典，不是异步方法
        model_settings = {"name": active_config.model_name, "api_base": active_config.api_base}
        logger.info(f"为文本创建单个异步嵌入 (Ollama): 模型 {model_settings['name']}")
        try:
            # _get_ollama_embedding 本身是同步的，如果需要异步包装，需要额外处理或使用异步ollama客户端
            # 这里我们保持同步调用，因为 tenacity 的 @retry 是同步的
            # 如果真的需要异步，需要将 _get_ollama_embedding 改为 async def 并使用异步 ollama 客户端
            return self._get_ollama_embedding(text, model_settings)
        except Exception as e:
            logger.error(f"使用 Ollama 创建单个异步嵌入失败: {e}")
            raise

    def get_document_embedding_config(self, collection_name: str) -> EmbeddingConfig:
        """尝试从已保存的嵌入文件中获取元数据，如果找不到则返回当前服务的默认Ollama配置"""
        try:
            embedded_docs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "02-embedded-docs")
            if not os.path.exists(embedded_docs_dir):
                logger.warning(f"Embedded docs directory not found: {embedded_docs_dir}. Returning default Ollama config.")
                return EmbeddingConfig() 
            
            # 简化匹配逻辑：如果集合名（通常是原始文件名）出现在嵌入文件名中
            # （嵌入文件名通常是 原始文件名_by_类型_时间戳.json 或 原始文件名_时间戳.json）
            for filename in os.listdir(embedded_docs_dir):
                if filename.endswith('.json'):
                    # 提取文件名中的文档名部分进行比较
                    doc_part_from_filename = filename.split('_by_')[0] if '_by_' in filename else filename.rsplit('_', 1)[0] if filename.count('_') > 0 and filename.rsplit('_', 1)[-1].replace('.json','').isdigit() else os.path.splitext(filename)[0]
                    if collection_name in doc_part_from_filename or doc_part_from_filename in collection_name:
                        try:
                            with open(os.path.join(embedded_docs_dir, filename), 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                metadata = data.get("metadata", {})
                                provider = metadata.get("embedding_provider")
                                model_name = metadata.get("embedding_model")
                                if provider == EmbeddingProvider.OLLAMA.value and model_name:
                                    logger.info(f"从文件 {filename} 为集合 {collection_name} 找到匹配的Ollama嵌入配置：{model_name}")
                                    # api_base 不一定存储在每个文件中，使用当前服务的默认值或OLLAMA_CONFIG
                                    return EmbeddingConfig(model_name=model_name, api_base=self.embedding_config.api_base)
                        except Exception as e:
                            logger.error(f"读取或解析嵌入文件 {filename} 元数据失败: {e}")
                            continue 
            logger.warning(f"未找到集合 {collection_name} 的匹配Ollama嵌入配置，使用当前服务默认Ollama配置。")
            return self.embedding_config # 返回当前服务实例的配置 (已是Ollama)
        except Exception as e:
            logger.error(f"获取文档嵌入配置时出错: {str(e)}")
            return self.embedding_config # 出错时返回当前服务实例的配置
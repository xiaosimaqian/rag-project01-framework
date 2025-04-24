import os
from datetime import datetime
import json
from typing import List, Dict, Any
import logging
from pathlib import Path
from pymilvus import connections, utility, Collection, DataType, FieldSchema, CollectionSchema, MilvusException
from utils.config import VectorDBProvider, MILVUS_CONFIG, CHROMA_CONFIG, MILVUS_LITE_FILE
from services.chroma_service import ChromaService
import time

logger = logging.getLogger(__name__)

class VectorDBConfig:
    """
    向量数据库配置类，用于存储和管理向量数据库的配置信息
    """
    def __init__(self, provider: str, index_mode: str, target_collection_name: str = None, rebuild_index: bool = False):
        """
        初始化向量数据库配置
        
        参数:
            provider: 向量数据库提供商名称
            index_mode: 索引模式
            target_collection_name: 目标集合名称（用于 append 操作）
            rebuild_index: 是否重建索引
        """
        self.provider = provider
        self.index_mode = index_mode
        self.target_collection_name = target_collection_name
        self.rebuild_index = rebuild_index
        self.host = None
        self.port = None
        self.uri = None
        self.collection_metadata = None
        self.index_params = {}

        if provider == VectorDBProvider.MILVUS.value:
            # 从 MILVUS_CONFIG 获取配置
            self.uri = MILVUS_CONFIG.get("uri", MILVUS_LITE_FILE)
            self.host = MILVUS_CONFIG.get("host", "localhost")
            self.port = MILVUS_CONFIG.get("port", 19530)
            self.index_params = MILVUS_CONFIG.get("index_params", {}).get(index_mode, {})
            
            # 确保 URI 存在
            if not self.uri:
                self.uri = MILVUS_LITE_FILE
            
            # 确保目录存在
            os.makedirs(os.path.dirname(self.uri), exist_ok=True)
            logger.info(f"Initialized Milvus config with URI: {self.uri}")

        elif provider == VectorDBProvider.CHROMA.value:
            if CHROMA_CONFIG:
                self.uri = CHROMA_CONFIG.get("uri", "03-vector-store/chroma_db")
                self.collection_metadata = CHROMA_CONFIG.get("collection_metadata", {})
                self.index_params = CHROMA_CONFIG.get("index_params", {}).get(index_mode, {})
            else:
                self.uri = "03-vector-store/chroma_db"
            logger.info(f"Initialized Chroma config with URI: {self.uri}")
        else:
            logger.warning(f"Unsupported provider '{provider}' during VectorDBConfig init.")

    def _get_milvus_index_type(self, index_mode: str) -> str:
        """
        从配置对象获取Milvus索引类型
        
        参数:
            index_mode: 索引模式
            
        返回:
            Milvus索引类型
        """
        if index_mode in MILVUS_CONFIG["index_types"]:
            return MILVUS_CONFIG["index_types"].get(index_mode, "FLAT")
        return "FLAT"  # 默认返回 FLAT 索引类型
    
    def _get_milvus_index_params(self, index_mode: str) -> Dict[str, Any]:
        """
        从配置对象获取Milvus索引参数
        
        参数:
            index_mode: 索引模式
            
        返回:
            对应的Milvus索引参数字典
        """
        return MILVUS_CONFIG["index_params"].get(index_mode, {})


class VectorStoreService:
    """
    向量存储服务类，提供向量数据的索引、查询和管理功能
    """
    def __init__(self):
        """
        初始化向量存储服务
        """
        self.initialized_dbs = {}
        self.chroma_service = None  # 延迟初始化
        self.milvus_alias = "default"
        self.metadata_file = os.path.join(os.path.dirname(__file__), "..", "collection_metadata.json")
        self._init_metadata_file()
        # 确保存储目录存在 (Chroma 可能需要)
        chroma_db_path = Path(CHROMA_CONFIG.get("uri", "03-vector-store/chroma_db"))
        if chroma_db_path:
             os.makedirs(chroma_db_path.parent, exist_ok=True)
        
        # 检查并初始化数据库
        if not self._check_and_init_db():
            raise RuntimeError("无法初始化 Milvus Lite 数据库")
        
        # 不再在初始化时强制连接 Milvus
        logger.info("VectorStoreService initialized. Connections will be established on demand.")
    
    def _init_metadata_file(self):
        """初始化元数据文件"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.metadata_file), exist_ok=True)
            
            # 如果文件不存在，创建并初始化
            if not os.path.exists(self.metadata_file):
                logger.info(f"创建新的元数据文件: {self.metadata_file}")
                with open(self.metadata_file, 'w', encoding='utf-8') as f:
                    json.dump({}, f, indent=2, ensure_ascii=False)
            else:
                # 验证文件格式
                try:
                    with open(self.metadata_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if not isinstance(data, dict):
                            logger.warning(f"元数据文件格式不正确，将重新初始化: {self.metadata_file}")
                            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                                json.dump({}, f, indent=2, ensure_ascii=False)
                except json.JSONDecodeError:
                    logger.warning(f"元数据文件损坏，将重新初始化: {self.metadata_file}")
                    with open(self.metadata_file, 'w', encoding='utf-8') as f:
                        json.dump({}, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"初始化元数据文件时出错: {e}", exc_info=True)
            raise

    def _save_collection_metadata(self, collection_name: str, metadata: dict):
        """保存集合元数据"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.metadata_file), exist_ok=True)
            
            # 读取现有元数据
            current_metadata = {}
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    current_metadata = json.load(f)
            
            # 更新元数据
            current_metadata[collection_name] = metadata
            
            # 保存更新后的元数据
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(current_metadata, f, indent=2, ensure_ascii=False)
                
            logger.info(f"成功保存集合 '{collection_name}' 的元数据: {metadata}")
        except Exception as e:
            logger.error(f"保存集合 '{collection_name}' 的元数据时出错: {e}", exc_info=True)
            raise

    def _get_collection_metadata(self, collection_name: str) -> dict:
        """获取集合元数据"""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    logger.info(f"从文件 {self.metadata_file} 读取元数据: {metadata.get(collection_name)}")
                    return metadata.get(collection_name, {})
            else:
                logger.warning(f"元数据文件不存在: {self.metadata_file}")
                return {}
        except Exception as e:
            logger.error(f"读取集合元数据时出错: {e}", exc_info=True)
            return {}
    
    def _get_milvus_index_type(self, config: VectorDBConfig) -> str:
        """
        从配置对象获取Milvus索引类型
        
        参数:
            config: 向量数据库配置对象
            
        返回:
            Milvus索引类型
        """
        return config._get_milvus_index_type(config.index_mode)
    
    def _get_milvus_index_params(self, config: VectorDBConfig) -> Dict[str, Any]:
        """
        从配置对象获取Milvus索引参数
        
        参数:
            config: 向量数据库配置对象
            
        返回:
            Milvus索引参数字典
        """
        return config._get_milvus_index_params(config.index_mode)

    def _init_chroma_service(self, config: VectorDBConfig):
        """
        初始化 ChromaService，传入配置
        """
        if self.chroma_service is None:
            self.chroma_service = ChromaService(config)
    
    def index_embeddings(self, embeddings_data: dict, config: VectorDBConfig) -> Dict[str, Any]:
        """索引嵌入向量到向量数据库"""
        start_time = time.time()
        result = {}
        
        try:
            # 根据不同的数据库进行索引
            if config.provider == VectorDBProvider.MILVUS.value:
                logger.info(f"[index_embeddings] Using Milvus provider with config: {config.__dict__}")
                self._connect_milvus(config)
                result = self._index_to_milvus(embeddings_data, config.target_collection_name, config.index_mode)
                if result is None:
                    raise ValueError("_index_to_milvus returned None")
                logger.info(f"[index_embeddings] Successfully indexed to Milvus: {result}")
            elif config.provider == VectorDBProvider.CHROMA.value:
                logger.info(f"[index_embeddings] Using Chroma provider")
                self._init_chroma_service(config)
                result = self.chroma_service.index_embeddings(embeddings_data, config.index_mode)
                if result is None:
                    raise ValueError("Chroma index_embeddings returned None")
                logger.info(f"[index_embeddings] Successfully indexed to Chroma: {result}")
            else:
                raise ValueError(f"不支持的向量数据库提供商: {config.provider}")
            
        except Exception as e:
            logger.error(f"索引过程中发生错误: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"索引失败: {str(e)}",
                "details": {
                    "database": config.provider,
                    "index_mode": config.index_mode,
                    "total_vectors": 0,
                    "index_size": "N/A",
                    "processing_time": time.time() - start_time,
                    "collection_name": "N/A",
                    "action": "error",
                    "total_entities": 0,
                    "error": str(e)
                }
            }
        finally:
            if config.provider == VectorDBProvider.MILVUS.value:
                self._disconnect_milvus()
        
        # 确保返回的数据结构符合前端期望
        return {
            "status": "success",
            "message": "索引完成",
            "details": {
                "database": result.get("database", config.provider),
                "index_mode": result.get("index_mode", config.index_mode),
                "total_vectors": result.get("details", {}).get("total_vectors", len(embeddings_data.get("embeddings", []))),
                "index_size": result.get("details", {}).get("index_size", "N/A"),
                "processing_time": result.get("details", {}).get("processing_time", time.time() - start_time),
                "collection_name": result.get("details", {}).get("collection_name", config.target_collection_name),
                "action": result.get("details", {}).get("action", "create"),
                "total_entities": result.get("details", {}).get("total_entities", len(embeddings_data.get("embeddings", []))),
                "index_type": result.get("details", {}).get("index_type", self._get_milvus_index_type(config)),
                "metric_type": result.get("details", {}).get("metric_type", "L2"),
                "index_params": result.get("details", {}).get("index_params", self._get_milvus_index_params(config))
            }
        }
    
    def _load_embeddings(self, file_path: str) -> Dict[str, Any]:
        """
        加载embedding文件，返回配置信息和embeddings
        
        参数:
            file_path: 嵌入向量文件路径
            
        返回:
            包含嵌入向量、元数据以及文件路径的字典
        """
        try:
            # 确保文件路径是绝对路径
            if not os.path.isabs(file_path):
                file_path = os.path.abspath(file_path)
                
            logger.info(f"Loading embeddings from {file_path}")
            
            # 检查文件是否存在
            if not os.path.exists(file_path):
                logger.error(f"文件不存在: {file_path}")
                # 列出目录内容以便调试
                dir_path = os.path.dirname(file_path)
                if os.path.exists(dir_path):
                    logger.info(f"目录内容: {os.listdir(dir_path)}")
                raise FileNotFoundError(f"嵌入文件不存在: {file_path}")
            
            # 检查文件是否可读
            if not os.access(file_path, os.R_OK):
                logger.error(f"文件不可读: {file_path}")
                raise PermissionError(f"无法读取文件: {file_path}")
            
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON解析错误: {str(e)}")
                    raise ValueError(f"无效的JSON格式: {str(e)}")
                
                # 验证数据格式
                if not isinstance(data, dict):
                    logger.error(f"无效的数据格式: 期望字典，得到 {type(data)}")
                    raise ValueError("Invalid embedding file format: expected dictionary")
                    
                if "embeddings" not in data:
                    logger.error("缺少 'embeddings' 键")
                    raise ValueError("Invalid embedding file format: missing 'embeddings' key")
                    
                # 将文件路径添加到返回的字典中
                data['file_path'] = file_path 
                
                logger.info(f"成功加载 {len(data['embeddings'])} 个嵌入向量")
                return data
                
        except Exception as e:
            logger.error(f"加载嵌入向量时出错: {str(e)}", exc_info=True)
            raise
    
    def _index_to_milvus(self, embeddings_data: dict, collection_name: str, index_mode: str = "flat") -> dict:
        """索引到Milvus数据库"""
        try:
            # 记录输入数据的结构
            logger.info(f"Received embeddings_data keys: {embeddings_data.keys()}")
            
            # 从embeddings_data中获取必要信息
            embeddings = embeddings_data.get("embeddings", [])
            metadata = embeddings_data.get("metadata", {})
            
            logger.info(f"Number of embeddings: {len(embeddings)}")
            logger.info(f"Metadata: {metadata}")
            
            if not embeddings:
                raise ValueError("没有可用的嵌入向量数据")
            
            # 从metadata中获取向量维度
            vector_dimension = metadata.get("vector_dimension", 1024)
            logger.info(f"Vector dimension from metadata: {vector_dimension}")
            
            # 检查目标集合是否存在
            collection_exists = utility.has_collection(collection_name, using=self.milvus_alias)
            logger.info(f"Collection {collection_name} exists: {collection_exists}")
            
            # 如果集合不存在，创建新集合
            if not collection_exists:
                # 定义集合的schema
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dimension),
                    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="document_name", dtype=DataType.VARCHAR, max_length=256),
                    FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=256),
                    FieldSchema(name="page_number", dtype=DataType.INT64),
                    FieldSchema(name="word_count", dtype=DataType.INT64)
                ]
                schema = CollectionSchema(fields=fields, description=f"Collection for {collection_name}")
                
                # 使用Collection类创建集合
                collection = Collection(name=collection_name, schema=schema, using=self.milvus_alias)
                logger.info(f"Created new collection: {collection_name}")
            else:
                collection = Collection(name=collection_name, using=self.milvus_alias)
                # 获取现有集合的schema
                schema = collection.schema
                # 验证向量维度是否匹配
                for field in schema.fields:
                    if field.name == "vector":
                        if field.dim != vector_dimension:
                            raise ValueError(f"向量维度不匹配：现有集合维度为 {field.dim}，新数据维度为 {vector_dimension}")
            
            # 准备数据
            vectors = []
            texts = []
            metadata_list = []
            
            # 从embeddings_data中提取数据
            for i, embedding in enumerate(embeddings):
                # 确保向量数据存在
                vector = embedding.get("embedding", [])
                if not vector:
                    logger.warning(f"Skipping embedding {i} due to missing vector data")
                    continue
                    
                # 确保文本内容存在
                text = embedding.get("metadata", {}).get("content", "")
                if not text:
                    logger.warning(f"Skipping embedding {i} due to missing text content")
                    continue
                
                vectors.append(vector)
                texts.append(text)
                
                # 确保所有元数据字段都有默认值
                metadata_list.append({
                    "document_name": embedding.get("metadata", {}).get("document_name", metadata.get("document_name", "unknown")),
                    "chunk_id": str(embedding.get("metadata", {}).get("chunk_id", f"chunk_{i}")),  # 确保是字符串
                    "page_number": int(embedding.get("metadata", {}).get("page_number", 0) or 0),  # 处理None值
                    "word_count": int(embedding.get("metadata", {}).get("word_count", len(text.split())) or 0)  # 处理None值
                })
            
            if not vectors:
                raise ValueError("没有有效的向量数据可以插入")
                
            logger.info(f"Prepared {len(vectors)} vectors for insertion")
            logger.info(f"First vector sample: {vectors[0][:5] if vectors else 'No vectors'}")
            
            # 插入数据到集合
            insert_result = collection.insert([
                vectors,  # 向量数据
                texts,   # 文本内容
                [m["document_name"] for m in metadata_list],  # 文档名称
                [m["chunk_id"] for m in metadata_list],       # 块ID
                [m["page_number"] for m in metadata_list],    # 页码
                [m["word_count"] for m in metadata_list]      # 词数
            ])
            
            logger.info(f"Insert result: {insert_result}")
            
            # 创建索引
            index_params = {
                "metric_type": "L2",
                "index_type": MILVUS_CONFIG["index_types"].get(index_mode, "FLAT")
            }
            logger.info(f"Creating index with params: {index_params}")
            collection.create_index(field_name="vector", index_params=index_params)
            
            # 获取集合信息
            collection_info = collection.describe()
            num_entities = collection.num_entities
            logger.info(f"Collection info after insertion: {collection_info}")
            logger.info(f"Number of entities: {num_entities}")
            
            # 保存集合元数据
            collection_metadata = {
                "dimension": vector_dimension,
                "index_type": index_params["index_type"],
                "metric_type": index_params["metric_type"],
                "creation_time": datetime.now().isoformat(),
                "embedding_model": metadata.get("embedding_model", ""),
                "source_file": metadata.get("document_name", ""),
                "total_vectors": len(vectors)
            }
            self._save_collection_metadata(collection_name, collection_metadata)
            
            # 返回结果
            result = {
                "status": "success",
                "message": "索引完成",
                "details": {
                    "database": "milvus",
                    "collection_name": collection_name,
                    "index_mode": index_mode,
                    "action": "create" if not collection_exists else "append",
                    "total_vectors": len(vectors),
                    "total_entities": len(vectors),
                    "processing_time": 0,
                    "index_size": 0,
                    "index_type": index_params["index_type"],
                    "metric_type": index_params["metric_type"],
                    "index_params": index_params,
                    "dimension": vector_dimension,
                    "creation_time": collection_metadata["creation_time"]
                }
            }
            
            logger.info(f"Returning result: {json.dumps(result, indent=2)}")
            return result
            
        except Exception as e:
            logger.error(f"索引到Milvus失败: {str(e)}", exc_info=True)
            raise ValueError(f"索引到Milvus失败: {str(e)}")

    def list_collections(self, provider: str = "milvus") -> List[Dict[str, Any]]:
        """
        列出所有可用的集合
        """
        try:
            if provider == VectorDBProvider.MILVUS.value:
                config = VectorDBConfig(provider=provider, index_mode='')
                self._connect_milvus(config)
                
                collections = []
                for name in utility.list_collections(using=self.milvus_alias):
                    try:
                        collection = Collection(name=name, using=self.milvus_alias)
                        collection.load()  # 加载集合到内存
                        metadata = self._get_collection_metadata(name)
                        
                        # 获取索引信息
                        index_info = None
                        try:
                            index_info = collection.index().params
                        except Exception as e:
                            logger.warning(f"获取集合 {name} 的索引信息失败: {e}")
                        
                        collections.append({
                            "id": name,
                            "name": name,
                            "count": collection.num_entities,
                            "dimension": metadata.get("dimension"),
                            "index_type": index_info.get("index_type") if index_info else metadata.get("index_type"),
                            "metric_type": index_info.get("metric_type") if index_info else metadata.get("metric_type"),
                            "creation_time": metadata.get("creation_time", "N/A"),
                            "embedding_model": metadata.get("embedding_model"),
                            "source_file": metadata.get("source_file")
                        })
                    except Exception as e:
                        logger.warning(f"获取集合 {name} 信息时出错: {e}")
                        collections.append({
                            "id": name,
                            "name": name,
                            "count": 0,
                            "creation_time": "N/A"
                        })
                    finally:
                        if 'collection' in locals():
                            collection.release()
                        
                return collections
                
            elif provider == VectorDBProvider.CHROMA.value:
                if self.chroma_service is None:
                    self._init_chroma_service(None)
                return self.chroma_service.list_collections()
                
            else:
                raise ValueError(f"不支持的向量数据库提供商: {provider}")
            
        except Exception as e:
            logger.error(f"列出集合时出错: {e}")
            raise
        
    def get_collection_info(self, provider: str, collection_name: str) -> Dict[str, Any]:
        """
        获取集合的详细信息
        """
        try:
            if provider == VectorDBProvider.MILVUS.value:
                config = VectorDBConfig(provider=provider, index_mode='')
                self._connect_milvus(config)
                
                try:
                    collection = Collection(name=collection_name, using=self.milvus_alias)
                    collection.load()  # 加载集合到内存
                    
                    # 获取集合信息
                    collection_info = collection.describe()
                    num_entities = collection.num_entities
                    
                    # 获取索引信息
                    index_info = None
                    try:
                        index_info = collection.index().params
                        logger.info(f"获取到索引信息: {index_info}")
                    except Exception as e:
                        logger.warning(f"获取索引信息失败: {e}")
                        
                    # 获取存储的元数据
                    metadata = self._get_collection_metadata(collection_name)
                    logger.info(f"获取到的集合元数据: {metadata}")
                    
                    # 构造详细信息
                    info = {
                        "基本信息": {
                            "数据库类型": "milvus",
                            "集合名称": collection_name,
                            "向量总数": f"{num_entities:,}",
                            "创建时间": metadata.get("creation_time", "N/A")
                        },
                        "索引信息": {
                            "索引类型": index_info.get("index_type") if index_info else metadata.get("index_type", "FLAT"),
                            "距离度量": index_info.get("metric_type") if index_info else metadata.get("metric_type", "L2"),
                            "索引参数": index_info.get("params") if index_info else metadata.get("index_params", {})
                        },
                        "嵌入信息": {
                            "嵌入模型": metadata.get("embedding_model", "unknown"),
                            "源文件": metadata.get("source_file", "unknown")
                        }
                    }
                    
                    logger.info(f"集合信息: {info}")
                    return info
                    
                except Exception as e:
                    logger.error(f"获取集合信息时出错: {e}", exc_info=True)
                    raise
                finally:
                    if 'collection' in locals():
                        collection.release()
                    self._disconnect_milvus()

            elif provider == VectorDBProvider.CHROMA.value:
                self._init_chroma_service(None)
                return self.chroma_service.get_collection_info(collection_name)
            else:
                raise ValueError(f"Unsupported vector database provider: {provider}")
            
        except Exception as e:
            logger.error(f"Error getting collection info for '{collection_name}': {str(e)}", exc_info=True)
            raise

    def delete_collection(self, provider: str, collection_name: str) -> bool:
        """
        删除指定的集合
        
        参数:
            provider: 向量数据库提供商名称
            collection_name: 集合名称
            
        返回:
            是否删除成功
        """
        try:
            if provider == VectorDBProvider.CHROMA.value:
                self._init_chroma_service(None)
                return self.chroma_service.delete_collection(collection_name)
            elif provider == VectorDBProvider.MILVUS.value:
                # 确保断开所有现有连接
                try:
                    self._disconnect_milvus()
                except Exception as e:
                    logger.warning(f"Error during disconnect: {e}", exc_info=True)
                
                # 重新连接
                config = VectorDBConfig(provider=provider, index_mode='')
                self._connect_milvus(config)
                
                try:
                    # 检查集合是否存在
                    if not utility.has_collection(collection_name, using=self.milvus_alias):
                        logger.warning(f"Collection '{collection_name}' does not exist")
                        return True
                    
                    try:
                        # 尝试获取并释放集合
                        collection = Collection(name=collection_name, using=self.milvus_alias)
                        collection.release()
                        logger.info(f"Released collection resources for '{collection_name}'")
                    except Exception as e:
                        logger.warning(f"Error releasing collection: {e}", exc_info=True)
                    
                    # 等待资源释放
                    time.sleep(2)
                    
                    # 尝试删除集合
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            logger.info(f"Attempt {attempt + 1} to drop collection '{collection_name}'")
                            utility.drop_collection(collection_name, using=self.milvus_alias)
                            logger.info(f"Successfully deleted collection '{collection_name}'")
                            return True
                        except Exception as e:
                            if attempt < max_retries - 1:
                                logger.warning(f"Attempt {attempt + 1} failed: {e}", exc_info=True)
                                time.sleep(2)  # 在重试之前等待
                                # 尝试重新连接
                                self._disconnect_milvus()
                                self._connect_milvus(config)
                            else:
                                raise  # 最后一次尝试失败时抛出异常
                except Exception as e:
                    logger.error(f"Error during collection deletion: {str(e)}", exc_info=True)
                    raise
                finally:
                    try:
                        self._disconnect_milvus()
                    except Exception as e:
                        logger.warning(f"Error during final disconnect: {e}", exc_info=True)
            else:
                raise ValueError(f"Unsupported vector database provider: {provider}")
                
        except Exception as e:
            logger.error(f"Failed to delete collection '{collection_name}': {str(e)}", exc_info=True)
            return False

    def _connect_milvus(self, config: VectorDBConfig):
        """
        连接到 Milvus
        """
        try:
            # 先检查是否已经存在连接
            try:
                connections.disconnect(alias=self.milvus_alias)
            except Exception:
                pass
            
            # 确保 URI 正确设置
            if not config.uri:
                config.uri = MILVUS_CONFIG.get("uri", MILVUS_LITE_FILE)
            
            # 确保目录存在
            os.makedirs(os.path.dirname(config.uri), exist_ok=True)
            
            logger.info(f"正在连接到 Milvus，URI: {config.uri}")
            connections.connect(
                alias=self.milvus_alias,
                uri=config.uri
            )
            logger.info(f"成功连接到 Milvus (alias: {self.milvus_alias})")
        except Exception as e:
            logger.error(f"连接 Milvus 时出错: {e}")
            raise

    def _disconnect_milvus(self):
        """
        断开与 Milvus 的连接
        """
        try:
            logger.info(f"正在断开 Milvus 连接 (alias: {self.milvus_alias})")
            connections.disconnect(alias=self.milvus_alias)
        except Exception as e:
            logger.warning(f"断开 Milvus 连接时出错: {e}")

    def search(
        self,
        query_embedding: List[float],
        collection_name: str,
        provider: str,
        config: VectorDBConfig,
        top_k: int = 3,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        在指定集合中搜索最相似的向量
        """
        try:
            logger.info(f"在集合 '{collection_name}' 中初始化搜索，提供商 '{provider}'")
            
            # 确保连接已建立
            self._connect_milvus(config)
            
            # 检查集合是否存在
            collections = utility.list_collections()
            if collection_name not in collections:
                raise ValueError(
                    f"集合 '{collection_name}' 不存在。\n"
                    f"可用的集合有: {collections}\n"
                    "请确保：\n"
                    "1. 集合名称输入正确\n"
                    "2. 已经完成文档的索引过程\n"
                    "3. Milvus Lite 数据库文件未被删除或损坏"
                )
            
            try:
                collection = Collection(name=collection_name, using=self.milvus_alias)
                collection.load()
                
                search_params = {
                    "metric_type": "L2",
                    "params": {"nprobe": 10}
                }
                
                logger.info(f"使用参数执行搜索: {search_params}")
                results = collection.search(
                    data=[query_embedding],
                    anns_field="vector",
                    param=search_params,
                    limit=top_k,
                    output_fields=["content", "document_name", "chunk_id", "page_number", "word_count"]
                )
                
                search_results = []
                for hits in results:
                    for hit in hits:
                        if hit.score <= threshold:  # 对于L2距离，较小的值表示更相似
                            search_results.append({
                                "content": hit.entity.get("content", ""),
                                "score": 1 - hit.score,  # 转换分数为相似度
                                "metadata": {
                                    "document_name": hit.entity.get("document_name", ""),
                                    "chunk_id": hit.entity.get("chunk_id", -1),
                                    "page_number": hit.entity.get("page_number", -1),
                                    "word_count": hit.entity.get("word_count", 0)
                                }
                            })
                
                return search_results
                
            finally:
                if 'collection' in locals():
                    collection.release()
                
        except Exception as e:
            logger.error(f"搜索操作出错: {e}")
            raise
        finally:
            self._disconnect_milvus()

    def _check_and_init_db(self):
        """
        检查并初始化数据库
        """
        try:
            db_path = Path(MILVUS_CONFIG["uri"])
            if not db_path.exists():
                logger.warning(f"Milvus Lite 数据库文件不存在: {db_path}")
                # 确保目录存在
                db_path.parent.mkdir(parents=True, exist_ok=True)
                # 创建空数据库文件
                db_path.touch()
                logger.info(f"已创建新的 Milvus Lite 数据库文件: {db_path}")
            return True
        except Exception as e:
            logger.error(f"检查/初始化数据库时出错: {e}")
        return False

    def _ensure_collection_exists(self, collection_name: str, dimension: int = 1024) -> bool:
        """
        确保集合存在，如果不存在则创建
        
        参数:
            collection_name: 集合名称
            dimension: 向量维度
        
        返回:
            bool: 集合是否存在或创建成功
        """
        try:
            if utility.has_collection(collection_name):
                logger.info(f"集合已存在: {collection_name}")
                return True
            
            logger.warning(f"集合不存在，将创建新集合: {collection_name}")
            
            # 定义集合字段
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="document_name", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="chunk_id", dtype=DataType.INT64),
                FieldSchema(name="page_number", dtype=DataType.INT64),
                FieldSchema(name="word_count", dtype=DataType.INT64)
            ]
            
            schema = CollectionSchema(fields=fields, description=f"Collection for {collection_name}")
            Collection(name=collection_name, schema=schema)
            
            logger.info(f"成功创建集合: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"确保集合存在时出错: {e}")
            return False

    def _verify_index_creation(self, collection: Collection, collection_name: str):
        """验证索引是否正确创建"""
        try:
            # 获取索引信息
            index_info = collection.index()
            if not index_info:
                logger.warning(f"集合 '{collection_name}' 的索引信息为空")
                return False
            
            # 验证索引参数
            index_params = index_info.params
            if not index_params.get("index_type") or not index_params.get("metric_type"):
                logger.warning(f"集合 '{collection_name}' 的索引参数不完整: {index_params}")
                return False
            
            logger.info(f"集合 '{collection_name}' 的索引验证成功: {index_params}")
            return True
        except Exception as e:
            logger.error(f"验证索引时出错: {e}", exc_info=True)
            return False
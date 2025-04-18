import os
from datetime import datetime
import json
from typing import List, Dict, Any
import logging
from pathlib import Path
from pymilvus import connections, utility, Collection, DataType, FieldSchema, CollectionSchema, MilvusException
from utils.config import VectorDBProvider, MILVUS_CONFIG, CHROMA_CONFIG # Updated import
from services.chroma_service import ChromaService
import time

logger = logging.getLogger(__name__)

# 定义 Milvus Lite 数据库文件路径
# 将其放在 backend 目录下，确保路径相对于项目根目录或 backend 目录是正确的
MILVUS_LITE_FILE = "./milvus_lite.db" # Assuming running from backend dir, or adjust as needed

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
        self.uri = None  # 用于 Milvus Lite 或 Chroma
        self.collection_metadata = None
        self.index_params = {}

        logger.debug(f"Initializing VectorDBConfig for provider: '{provider}', target collection: '{target_collection_name}'")

        if provider == VectorDBProvider.MILVUS.value:
            # 使用 Milvus Lite: 设置 URI 为本地文件路径
            self.uri = MILVUS_LITE_FILE
            logger.debug(f"Provider is Milvus Lite. Using URI: {self.uri}")
            # 保留索引参数的加载逻辑
            self.index_params = MILVUS_CONFIG.get("index_params", {}).get(index_mode, {})

        elif provider == VectorDBProvider.CHROMA.value:
            logger.debug("Provider is Chroma.")
            if CHROMA_CONFIG and "uri" in CHROMA_CONFIG:
                # Chroma 也使用 URI
                self.uri = CHROMA_CONFIG["uri"]
                self.collection_metadata = CHROMA_CONFIG.get("collection_metadata", {})
                self.index_params = CHROMA_CONFIG.get("index_params", {}).get(index_mode, {})
            else:
                logger.error("CHROMA_CONFIG is missing or incomplete. Using defaults.")
                self.uri = "03-vector-store/chroma_db" # Default Chroma URI
        else:
            logger.warning(f"Unsupported provider '{provider}' during VectorDBConfig init. No specific config loaded.")

    def _get_milvus_index_type(self, index_mode: str) -> str:
        """
        根据索引模式获取Milvus索引类型
        
        参数:
            index_mode: 索引模式
            
        返回:
            对应的Milvus索引类型
        """
        if MILVUS_CONFIG and "index_types" in MILVUS_CONFIG:
            return MILVUS_CONFIG["index_types"].get(index_mode, "FLAT")
        logger.warning("MILVUS_CONFIG missing 'index_types', defaulting to FLAT.")
        return "FLAT"
    
    def _get_milvus_index_params(self, index_mode: str) -> Dict[str, Any]:
        """
        根据索引模式获取Milvus索引参数
        
        参数:
            index_mode: 索引模式
            
        返回:
            对应的Milvus索引参数字典
        """
        return self.index_params


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
        self.metadata_file = os.path.join("backend", "collection_metadata.json")
        self._init_metadata_file()
        # 确保存储目录存在 (Chroma 可能需要)
        chroma_db_path = Path(CHROMA_CONFIG.get("uri", "03-vector-store/chroma_db"))
        if chroma_db_path:
             os.makedirs(chroma_db_path.parent, exist_ok=True)
        
        # 不再在初始化时强制连接 Milvus
        logger.info("VectorStoreService initialized. Connections will be established on demand.")
    
    def _init_metadata_file(self):
        """初始化元数据文件"""
        if not os.path.exists(self.metadata_file):
            os.makedirs(os.path.dirname(self.metadata_file), exist_ok=True)
            with open(self.metadata_file, 'w') as f:
                json.dump({}, f)

    def _save_collection_metadata(self, collection_name: str, metadata: dict):
        """保存集合元数据"""
        try:
            current_metadata = {}
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    current_metadata = json.load(f)
            
            current_metadata[collection_name] = metadata
            
            with open(self.metadata_file, 'w') as f:
                json.dump(current_metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save collection metadata: {e}")

    def _get_collection_metadata(self, collection_name: str) -> dict:
        """获取集合元数据"""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                return metadata.get(collection_name, {})
        except Exception as e:
            logger.warning(f"Failed to read collection metadata: {e}")
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
    
    def index_embeddings(self, embedding_file: str, config: VectorDBConfig) -> Dict[str, Any]:
        """
        将嵌入向量索引到向量数据库
        
        参数:
            embedding_file: 嵌入向量文件路径
            config: 向量数据库配置对象
            
        返回:
            索引结果信息字典
        """
        start_time = datetime.now()
        result = {}
        
        # 读取embedding文件
        embeddings_data = self._load_embeddings(embedding_file)
        
        try:
            # 根据不同的数据库进行索引
            if config.provider == VectorDBProvider.MILVUS.value:
                # 确保 Milvus 已连接
                self._connect_milvus(config)
                result = self._index_to_milvus(embeddings_data, config)
            elif config.provider == VectorDBProvider.CHROMA.value:
                self._init_chroma_service(config)  # 确保 ChromaService 已初始化
                result = self.chroma_service.index_embeddings(embeddings_data, config.index_mode)
            else:
                raise ValueError(f"Unsupported vector database provider: {config.provider}")
        finally:
            # 如果是 Milvus，操作完成后断开连接
            if config.provider == VectorDBProvider.MILVUS.value:
                self._disconnect_milvus()
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        return {
            "database": config.provider,
            "index_mode": config.index_mode,
            "total_vectors": len(embeddings_data["embeddings"]),
            "index_size": result.get("index_size", "N/A"),
            "processing_time": processing_time,
            "collection_name": result.get("collection_name", "N/A")
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
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"Loading embeddings from {file_path}")
                
                if not isinstance(data, dict) or "embeddings" not in data:
                    raise ValueError("Invalid embedding file format: missing 'embeddings' key")
                    
                # 将文件路径添加到返回的字典中
                data['file_path'] = file_path 
                
                logger.info(f"Found {len(data['embeddings'])} embeddings in {file_path}")
                return data
                
        except Exception as e:
            logger.error(f"Error loading embeddings from {file_path}: {str(e)}")
            raise
    
    def _index_to_milvus(self, embeddings_data: Dict[str, Any], config: VectorDBConfig) -> Dict[str, Any]:
        """
        将嵌入向量索引到Milvus向量数据库
        """
        start_time = time.time()
        
        try:
            conn_addr = connections.get_connection_addr(alias=self.milvus_alias)
            logger.info(f"[_index_to_milvus] Connection check OK. Alias '{self.milvus_alias}' connected to: {conn_addr}")
        except Exception as e:
            logger.error(f"[_index_to_milvus] Connection check FAILED for alias '{self.milvus_alias}': {e}", exc_info=True)
            raise

        # 使用目标集合名称（如果有）或生成新的集合名称
        if config.target_collection_name:
            collection_name = config.target_collection_name
            logger.info(f"[_index_to_milvus] Using target collection name: {collection_name}")
        else:
            base_name = embeddings_data.get("chunked_doc_name", "").split('_')[0]
            if not base_name:
                base_name = Path(embeddings_data.get("filename", "unknown")).stem
            collection_name = f"collection_{base_name}_{config.index_mode}"
            collection_name = collection_name.replace('.', '_').replace('-', '_')
            logger.info(f"[_index_to_milvus] Generated new collection name: {collection_name}")

        if not embeddings_data.get("embeddings"):
            raise ValueError("No embeddings found in data")
        
        first_embedding = embeddings_data["embeddings"][0]
        if "embedding" not in first_embedding or not first_embedding["embedding"]:
            raise ValueError("First embedding is missing 'embedding' key or embedding is empty")
        
        embedding_dim = len(first_embedding["embedding"])
        collection = None

        # 检查集合是否存在
        try:
            has_col = utility.has_collection(collection_name, using=self.milvus_alias)
            logger.info(f"[_index_to_milvus] Checking existence of collection '{collection_name}': {has_col}")
            
            if has_col:
                collection = Collection(name=collection_name, using=self.milvus_alias)
                logger.info(f"[_index_to_milvus] Retrieved existing collection: {collection}")
                
                # 验证现有集合的维度
                existing_schema = collection.schema
                existing_dim = next((field.params["dim"] for field in existing_schema.fields if field.name == "vector"), None)
                if existing_dim != embedding_dim:
                    raise ValueError(f"Dimension mismatch: existing collection has dim={existing_dim}, new embeddings have dim={embedding_dim}")
            else:
                if config.target_collection_name:
                    raise ValueError(f"Target collection '{config.target_collection_name}' does not exist")
                
                # 创建新集合
                fields = [
                    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=255),
                    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="document_name", dtype=DataType.VARCHAR, max_length=255),
                    FieldSchema(name="chunk_id", dtype=DataType.INT64),
                    FieldSchema(name="total_chunks", dtype=DataType.INT64),
                    FieldSchema(name="word_count", dtype=DataType.INT64),
                    FieldSchema(name="page_number", dtype=DataType.INT64),
                    FieldSchema(name="page_range", dtype=DataType.VARCHAR, max_length=50),
                    FieldSchema(name="embedding_provider", dtype=DataType.VARCHAR, max_length=100),
                    FieldSchema(name="embedding_model", dtype=DataType.VARCHAR, max_length=100),
                    FieldSchema(name="embedding_timestamp", dtype=DataType.VARCHAR, max_length=50),
                    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim)
                ]
                
                schema = CollectionSchema(fields, f"Collection for {Path(embeddings_data['file_path']).name}")
                logger.info(f"[_index_to_milvus] Creating collection '{collection_name}' with schema: {schema}")
                collection = Collection(name=collection_name, schema=schema, using=self.milvus_alias)
                logger.info(f"[_index_to_milvus] Collection created successfully")
                
                collection.flush()
                time.sleep(1)

        except Exception as e:
            logger.error(f"[_index_to_milvus] Error with collection '{collection_name}': {e}", exc_info=True)
            raise

        if collection is None:
            raise MilvusException(message=f"Failed to get or create collection '{collection_name}'")

        # 准备并插入数据
        entities = []
        skipped_count = 0
        for i, embedding_item in enumerate(embeddings_data["embeddings"]):
            if "embedding" not in embedding_item or not isinstance(embedding_item["embedding"], list) or len(embedding_item["embedding"]) != embedding_dim:
                logger.warning(f"Skipping embedding {i} due to invalid format")
                skipped_count += 1
                continue

            entity = {
                "id": f"{Path(embeddings_data['file_path']).stem}_chunk_{embedding_item.get('chunk_id', i)}",
                "content": embedding_item.get("content", ""),
                "document_name": Path(embeddings_data['file_path']).name,
                "chunk_id": embedding_item.get("chunk_id", i),
                "total_chunks": embeddings_data.get("total_chunks", -1),
                "word_count": len(embedding_item.get("content", "").split()),
                "page_number": embedding_item.get("page_number", -1),
                "page_range": embedding_item.get("page_range", ""),
                "embedding_provider": embeddings_data.get("embedding_config", {}).get("provider", "Unknown"),
                "embedding_model": embeddings_data.get("embedding_config", {}).get("model_name", "Unknown"),
                "embedding_timestamp": embeddings_data.get("embedding_timestamp", "Unknown"),
                "vector": embedding_item["embedding"]
            }
            entities.append(entity)

        if not entities:
            return {
                "collection_name": collection_name,
                "index_size": "N/A - No entities inserted",
                "error": "No valid entities to insert"
            }

        # 插入数据
        logger.info(f"[_index_to_milvus] Inserting {len(entities)} entities into '{collection_name}'")
        insert_result = collection.insert(entities)
        inserted_count = insert_result.insert_count
        logger.info(f"[_index_to_milvus] Inserted {inserted_count} entities")

        # 只在创建新集合时创建索引
        if not config.target_collection_name or config.rebuild_index:
            index_type = self._get_milvus_index_type(config)
            index_params = self._get_milvus_index_params(config)
            logger.info(f"[_index_to_milvus] Creating index '{index_type}' with params: {index_params}")
            collection.create_index(
                field_name="vector",
                index_params={
                    "index_type": index_type,
                    "params": index_params,
                    "metric_type": "L2"
                }
            )
            logger.info(f"[_index_to_milvus] Index created successfully")

        # 在创建新集合时保存元数据
        if not config.target_collection_name:
            metadata = {
                "creation_time": datetime.now().isoformat(),
                "embedding_provider": embeddings_data.get("embedding_config", {}).get("provider", "Unknown"),
                "embedding_model": embeddings_data.get("embedding_config", {}).get("model_name", "Unknown"),
                "source_file": embeddings_data.get("file_path", "Unknown"),
                "dimension": embedding_dim,
                "index_type": self._get_milvus_index_type(config),
                "index_params": self._get_milvus_index_params(config)
            }
            self._save_collection_metadata(collection_name, metadata)

        # 获取最终统计信息
        collection.flush()
        total_entities = collection.num_entities
        processing_time = time.time() - start_time

        result = {
            "collection_name": collection_name,
            "total_vectors": inserted_count,  # 本次插入的向量数
            "total_entities": total_entities,  # 集合中的总向量数
            "processing_time": processing_time,
            "index_size": f"总行数: {total_entities}",
            "action": "append" if config.target_collection_name else "create"
        }
        
        logger.info(f"[_index_to_milvus] Operation completed: {result}")
        return result

    def list_collections(self, provider: str) -> List[Dict[str, Any]]:
        """
        列出指定提供商的所有集合
        
        参数:
            provider: 向量数据库提供商名称
            
        返回:
            集合信息列表
        """
        try:
            if provider == VectorDBProvider.CHROMA.value:
                self._init_chroma_service(None)
                return self.chroma_service.list_collections()
            elif provider == VectorDBProvider.MILVUS.value:
                config = VectorDBConfig(provider=provider, index_mode='')
                if not config.uri:
                    raise ValueError("Milvus Lite configuration (URI) could not be loaded correctly.")
                self._connect_milvus(config)
                try:
                    logger.info(f"[list_collections] Listing Milvus collections using alias '{self.milvus_alias}'")
                    collections_info = []
                    collection_names = utility.list_collections(using=self.milvus_alias)
                    logger.info(f"[list_collections] Found collections: {collection_names}")
                    for name in collection_names:
                        try:
                            collection = Collection(name, using=self.milvus_alias)
                            collections_info.append({
                                "id": name,
                                "name": name,
                                "count": collection.num_entities
                            })
                        except Exception as e:
                            logger.warning(f"[list_collections] Could not retrieve info for Milvus collection '{name}': {e}")
                            collections_info.append({"id": name, "name": name, "count": "Error"})
                    return collections_info
                finally:
                    self._disconnect_milvus()
            else:
                raise ValueError(f"Unsupported vector database provider: {provider}")
                
        except Exception as e:
            logger.error(f"Error listing collections for provider {provider}: {type(e).__name__} - {str(e)}", exc_info=True)
            raise e

    def get_collection_info(self, provider: str, collection_name: str) -> Dict[str, Any]:
        """
        获取集合的详细信息
        
        参数:
            provider: 向量数据库提供商名称
            collection_name: 集合名称
            
        返回:
            集合详细信息字典
        """
        try:
            if provider == VectorDBProvider.MILVUS.value:
                # 确保 Milvus 已连接
                config = VectorDBConfig(provider=provider, index_mode='')
                self._connect_milvus(config)
                
                try:
                    collection = Collection(name=collection_name, using=self.milvus_alias)
                    num_entities = collection.num_entities
                    
                    # 获取存储的元数据
                    metadata = self._get_collection_metadata(collection_name)
                    
                    # 字段说明映射
                    field_descriptions = {
                        "id": "文档块的唯一标识符",
                        "content": "文档块的实际文本内容",
                        "document_name": "源文档的文件名",
                        "chunk_id": "文档块在文档中的序号",
                        "total_chunks": "文档被分割的总块数",
                        "word_count": "当前文档块的词数",
                        "page_number": "文档块在PDF中的页码",
                        "page_range": "文档块跨越的页码范围",
                        "embedding_provider": "生成向量嵌入的服务提供商",
                        "embedding_model": "生成向量嵌入使用的模型",
                        "embedding_timestamp": "向量嵌入的生成时间",
                        "vector": "文档块的向量表示"
                    }

                    # 数据类型说明映射
                    dtype_descriptions = {
                        "DataType.VARCHAR": "可变长度字符串",
                        "DataType.INT64": "64位整数",
                        "DataType.FLOAT_VECTOR": "浮点数向量",
                        "VARCHAR": "可变长度字符串",
                        "INT64": "64位整数",
                        "FLOAT_VECTOR": "浮点数向量"
                    }

                    # 获取集合的schema信息
                    schema_info = {"fields": []}
                    for field in collection.schema.fields:
                        field_info = {
                            "name": field.name,
                            "dtype": str(field.dtype),
                            "description": field_descriptions.get(field.name, field.description or ""),
                            "type_description": dtype_descriptions.get(str(field.dtype), ""),
                            "is_primary": field.is_primary,
                            "auto_id": field.auto_id
                        }
                        
                        # 添加字段的特殊属性
                        if hasattr(field, 'params') and field.params:
                            field_info["params"] = field.params
                            # 对于向量字段，添加维度信息的说明
                            if "dim" in field.params:
                                field_info["dimension"] = field.params["dim"]
                                field_info["description"] += f"，维度：{field.params['dim']}"
                        
                            # 添加最大长度信息（对于VARCHAR类型）
                            if hasattr(field, 'max_length') and field.max_length:
                                field_info["max_length"] = field.max_length
                                if "VARCHAR" in str(field.dtype):
                                    field_info["description"] += f"，最大长度：{field.max_length}字符"
                        
                        schema_info["fields"].append(field_info)
                    
                    # 获取索引信息
                    index_params_list = collection.indexes
                    index_info = {}
                    if index_params_list:
                        first_index = index_params_list[0]
                        index_info["index_type"] = first_index.params.get("index_type", "N/A")
                        index_info["metric_type"] = first_index.params.get("metric_type", "N/A")
                        index_params = first_index.params.get("params", {})
                        index_info["params"] = index_params if isinstance(index_params, dict) else {}

                    # 构造详细信息
                    info = {
                        "name": collection.name,
                        "num_entities": num_entities,
                        "schema": schema_info,
                        "index_type": index_info.get("index_type", "N/A"),
                        "index_params": index_info.get("params", {}),
                        "metric_type": index_info.get("metric_type", "N/A"),
                        "description": collection.description or f"Collection '{collection_name}' with {num_entities} entities",
                        "creation_time": metadata.get("creation_time"),  # 从元数据中获取创建时间
                        "embedding_provider": metadata.get("embedding_provider", "Unknown"),
                        "embedding_model": metadata.get("embedding_model", "Unknown"),
                        "source_file": metadata.get("source_file", "Unknown")
                    }
                    
                    logger.info(f"[get_collection_info] Retrieved info: {info}")
                    return info
                    
                except Exception as e:
                    logger.error(f"[get_collection_info] Error getting info: {e}", exc_info=True)
                    raise
                finally:
                    try:
                        if 'collection' in locals() and collection.is_empty == False:
                            collection.release()
                            logger.info(f"Released collection '{collection_name}' resources.")
                    except Exception as release_error:
                        logger.warning(f"Error releasing collection '{collection_name}': {release_error}")
                    
                    self._disconnect_milvus()
            else:
                if provider == VectorDBProvider.CHROMA.value:
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
        连接到Milvus实例 (服务器或Lite)
        """
        # 确保之前的连接已断开
        self._disconnect_milvus()
        
        try:
            # 等待一小段时间确保之前的连接完全释放
            time.sleep(1)
            
            if config and config.uri and config.provider == VectorDBProvider.MILVUS.value:
                logger.info(f"Connecting to Milvus Lite using URI: {config.uri}")
                db_path = Path(config.uri)
                os.makedirs(db_path.parent, exist_ok=True)
                connections.connect(alias=self.milvus_alias, uri=config.uri)
            else:
                # 使用默认的 Milvus Lite 配置
                default_uri = MILVUS_LITE_FILE
                logger.info(f"Connecting to Milvus Lite using default URI: {default_uri}")
                db_path = Path(default_uri)
                os.makedirs(db_path.parent, exist_ok=True)
                connections.connect(alias=self.milvus_alias, uri=default_uri)

            logger.info(f"Successfully connected to Milvus (alias: {self.milvus_alias})")
            self.initialized_dbs[self.milvus_alias] = True

        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}", exc_info=True)
            self.initialized_dbs[self.milvus_alias] = False
            raise

    def _disconnect_milvus(self):
        """
        断开与Milvus实例的连接
        """
        try:
            # 获取所有连接的别名
            aliases = connections.list_connections()
            
            # 断开所有连接
            for alias in aliases:
                try:
                    logger.info(f"Disconnecting from Milvus (alias: {alias})...")
                    connections.disconnect(alias=alias)
                    logger.info(f"Successfully disconnected from Milvus (alias: {alias})")
                except Exception as e:
                    logger.warning(f"Error disconnecting from Milvus (alias: {alias}): {e}")
            
            # 重置初始化状态
            self.initialized_dbs = {}
            
        except Exception as e:
            logger.error(f"Error during Milvus disconnect: {e}", exc_info=True)
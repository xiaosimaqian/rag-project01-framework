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
    def __init__(self, provider: str, index_mode: str):
        """
        初始化向量数据库配置
        
        参数:
            provider: 向量数据库提供商名称
            index_mode: 索引模式
        """
        self.provider = provider
        self.index_mode = index_mode
        self.host = None
        self.port = None
        self.uri = None  # 用于 Milvus Lite 或 Chroma
        self.collection_metadata = None
        self.index_params = {}

        logger.debug(f"Initializing VectorDBConfig for provider: '{provider}'")

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
        # 确保存储目录存在 (Chroma 可能需要)
        chroma_db_path = Path(CHROMA_CONFIG.get("uri", "03-vector-store/chroma_db"))
        if chroma_db_path:
             os.makedirs(chroma_db_path.parent, exist_ok=True)
        
        # 不再在初始化时强制连接 Milvus
        logger.info("VectorStoreService initialized. Connections will be established on demand.")

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
        # --- Connection Check at Start ---
        try:
            conn_addr = connections.get_connection_addr(alias=self.milvus_alias)
            logger.info(f"[_index_to_milvus] Connection check OK. Alias '{self.milvus_alias}' connected to: {conn_addr}")
        except Exception as e:
            logger.error(f"[_index_to_milvus] Connection check FAILED for alias '{self.milvus_alias}': {e}", exc_info=True)
            raise  # Re-raise connection error
        # --- End Connection Check ---

        base_name = embeddings_data.get("chunked_doc_name", "").split('_')[0]
        if not base_name:
            base_name = Path(embeddings_data.get("filename", "unknown")).stem
        
        collection_name = f"collection_{base_name}_{config.index_mode}"
        collection_name = collection_name.replace('.', '_').replace('-', '_') # Ensure name is valid

        logger.info(f"[_index_to_milvus] Target collection name: '{collection_name}'")

        if not embeddings_data.get("embeddings"):
            raise ValueError("No embeddings found in data")
        
        first_embedding = embeddings_data["embeddings"][0]
        if "embedding" not in first_embedding or not first_embedding["embedding"]:
            raise ValueError("First embedding is missing 'embedding' key or embedding is empty")
        
        embedding_dim = len(first_embedding["embedding"])
        
        collection = None # Initialize collection variable

        # 检查集合是否存在
        try:
            has_col = utility.has_collection(collection_name, using=self.milvus_alias)
            logger.info(f"[_index_to_milvus] Checking existence of collection '{collection_name}': {has_col}")
            if has_col:
                logger.warning(f"[_index_to_milvus] Collection '{collection_name}' already exists. Retrieving it.")
                collection = Collection(name=collection_name, using=self.milvus_alias)
                logger.info(f"[_index_to_milvus] Retrieved existing collection: {collection}") # Log collection object
            else:
                # 定义字段 (Simplified Schema)
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
                
                logger.info(f"[_index_to_milvus] Creating collection '{collection_name}' with SIMPLIFIED schema: {schema}")
                collection = Collection(name=collection_name, schema=schema, using=self.milvus_alias)
                logger.info(f"[_index_to_milvus] Collection '{collection_name}' created successfully: {collection}")

                # --- 新增：尝试 flush 和 sleep ---
                try:
                    logger.info(f"[_index_to_milvus] Flushing collection '{collection_name}' immediately after creation.")
                    collection.flush()
                    logger.info(f"[_index_to_milvus] Flush complete. Adding short sleep.")
                    time.sleep(1) # 等待 1 秒
                except Exception as flush_err:
                     # 记录 flush 错误，但可能仍尝试继续
                     logger.warning(f"[_index_to_milvus] Error during flush for collection '{collection_name}': {flush_err}", exc_info=True)
                # --- 结束新增 ---

        except Exception as e:
             logger.error(f"[_index_to_milvus] Error checking or creating collection '{collection_name}': {e}", exc_info=True)
             raise

        if collection is None:
            logger.error(f"[_index_to_milvus] Failed to obtain a valid collection object for '{collection_name}'")
            raise MilvusException(message=f"Failed to get or create collection '{collection_name}'")

        # 准备插入数据
        entities = []
        skipped_count = 0
        for i, embedding_item in enumerate(embeddings_data["embeddings"]): # Renamed variable to avoid confusion
            # --- 修改：检查 "embedding" 键 ---
            if "embedding" not in embedding_item or not isinstance(embedding_item["embedding"], list) or len(embedding_item["embedding"]) != embedding_dim:
                logger.warning(f"Skipping embedding index {i} due to missing, invalid, or dimension-mismatched 'embedding' key. Chunk ID: {embedding_item.get('chunk_id', 'N/A')}")
                skipped_count += 1
                continue
            # --- 结束检查 ---

            # --- 注意：JSON 文件片段中似乎缺少 chunk_id, content 等字段，添加 .get() 以增强健壮性 ---
            chunk_id = embedding_item.get("chunk_id", i) # 使用索引作为备用 chunk_id
            content = embedding_item.get("content", "")   # 如果缺少 content，设为空字符串

            entity = {
                "id": f"{Path(embeddings_data['file_path']).stem}_chunk_{chunk_id}",
                "content": content,
                "document_name": Path(embeddings_data['file_path']).name,
                "chunk_id": chunk_id,
                "total_chunks": embeddings_data.get("total_chunks", -1),
                "word_count": len(content.split()),
                "page_number": embedding_item.get("page_number", -1),
                "page_range": embedding_item.get("page_range", ""),
                "embedding_provider": embeddings_data.get("embedding_config", {}).get("provider", "Unknown"),
                "embedding_model": embeddings_data.get("embedding_config", {}).get("model_name", "Unknown"),
                "embedding_timestamp": embeddings_data.get("embedding_timestamp", "Unknown"),
                "vector": embedding_item["embedding"]
            }
            entities.append(entity)
        
        if skipped_count > 0:
             logger.warning(f"Skipped {skipped_count} out of {len(embeddings_data['embeddings'])} embeddings due to missing/invalid vectors.")

        if not entities:
             logger.error(f"[_index_to_milvus] No valid entities to insert into '{collection_name}' after filtering.")
             return {"collection_name": collection_name, "index_size": "N/A - No entities inserted", "error": "No valid entities to insert."}

        # --- Log Before Insert ---
        logger.info(f"[_index_to_milvus] Attempting to insert {len(entities)} entities into collection '{collection.name}' (Object: {collection}) using alias '{self.milvus_alias}'")
        try:
            # 插入数据
            insert_result = collection.insert(entities)
            logger.info(f"[_index_to_milvus] Successfully inserted {insert_result.insert_count} vectors into '{collection.name}'.")
        except Exception as e:
            logger.error(f"[_index_to_milvus] Error during insertion into collection '{collection_name}': {e}", exc_info=True)
            raise # Re-raise insertion error
        # --- End Log Before Insert ---

        # --- Log Before Index Creation ---
        index_type = self._get_milvus_index_type(config)
        index_params = self._get_milvus_index_params(config)
        logger.info(f"[_index_to_milvus] Attempting to create index '{index_type}' on field 'vector' for collection '{collection.name}' with params: {index_params}")
        try:
            # 创建索引
            collection.create_index(field_name="vector", index_params={"index_type": index_type, "params": index_params, "metric_type": "L2"})
            logger.info(f"[_index_to_milvus] Index '{index_type}' created successfully for collection '{collection.name}'.")
        except Exception as e:
            logger.error(f"[_index_to_milvus] Error creating index for collection '{collection_name}': {e}", exc_info=True)
            # Decide if this should be fatal or just logged
            # raise # Optionally re-raise index creation error
        # --- End Log Before Index Creation ---

        # --- Log Before Stats ---
        index_size_info = "N/A"
        logger.info(f"[_index_to_milvus] Attempting to get stats for collection '{collection_name}' using alias '{self.milvus_alias}'")
        try:
            stats = utility.get_collection_stats(collection_name=collection_name, using=self.milvus_alias)
            index_size_info = f"Row count: {stats.get('row_count', 'Unknown')}"
            logger.info(f"[_index_to_milvus] Got stats for collection '{collection_name}': {stats}")
        except Exception as e:
            logger.warning(f"[_index_to_milvus] Could not get stats for collection '{collection_name}': {e}")
        # --- End Log Before Stats ---

        return {
            "collection_name": collection_name,
            "index_size": index_size_info
        }

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
        获取指定集合的详细信息

        参数:
            provider: 向量数据库提供商名称
            collection_name: 集合名称

        返回:
            集合详细信息
        """
        try:
            if provider == VectorDBProvider.CHROMA.value:
                self._init_chroma_service(None) # Assuming config isn't needed just to get info
                return self.chroma_service.get_collection_info(collection_name)
            elif provider == VectorDBProvider.MILVUS.value:
                # --- 修改：需要传递有效的 Config 对象来连接 ---
                config = VectorDBConfig(provider=provider, index_mode='')
                if not config.uri:
                    raise ValueError("Milvus Lite configuration (URI) could not be loaded correctly.")
                self._connect_milvus(config)
                # --- 结束修改 ---
                try:
                    logger.info(f"[get_collection_info] Getting info for Milvus collection '{collection_name}' using alias '{self.milvus_alias}'")
                    collection = Collection(name=collection_name, using=self.milvus_alias)
                    # 尝试加载集合以获取更准确的信息，如果需要的话
                    # collection.load() # Uncomment if load is necessary before getting schema/num_entities
                    info = {
                        "name": collection.name,
                        "num_entities": collection.num_entities,
                        # .schema 可能返回一个复杂的对象，可能需要转换为字典或简化
                        "schema": str(collection.schema) # Convert schema to string for basic info
                    }
                    logger.info(f"[get_collection_info] Retrieved info: {info}")
                    return info
                except Exception as e:
                     logger.error(f"[get_collection_info] Error getting info for Milvus collection '{collection_name}': {e}", exc_info=True)
                     raise # Re-raise error after logging
                finally:
                    self._disconnect_milvus()
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
                self._init_chroma_service()
                return self.chroma_service.delete_collection(collection_name)
            elif provider == VectorDBProvider.MILVUS.value:
                # 连接到 Milvus
                self._connect_milvus(None)
                try:
                    utility.drop_collection(collection_name)
                    return True
                finally:
                    self._disconnect_milvus()
            else:
                raise ValueError(f"Unsupported vector database provider: {provider}")
                
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            return False

    def _connect_milvus(self, config: VectorDBConfig):
        """
        连接到Milvus实例 (服务器或Lite)
        
        参数:
            config: 向量数据库配置对象
        """
        # 如果已经连接，则先断开旧连接
        if self.initialized_dbs.get(self.milvus_alias, False):
             self._disconnect_milvus()

        try:
            if config.uri and config.provider == VectorDBProvider.MILVUS.value: # Milvus Lite 使用 URI
                 logger.info(f"Connecting to Milvus Lite using URI: {config.uri}")
                 # 确保存储 Milvus Lite 文件的目录存在
                 db_path = Path(config.uri)
                 os.makedirs(db_path.parent, exist_ok=True)
                 connections.connect(alias=self.milvus_alias, uri=config.uri)
            elif config.host and config.port and config.provider == VectorDBProvider.MILVUS.value: # Milvus Server 使用 host/port
                 logger.info(f"Connecting to Milvus server at {config.host}:{config.port}")
                 connections.connect(alias=self.milvus_alias, host=config.host, port=str(config.port)) # Ensure port is string
            else:
                 # 当前逻辑下，如果 provider 是 Milvus，必然有 uri
                 # 如果 provider 不是 Milvus，则不应调用此方法
                 logger.error(f"Invalid configuration for Milvus connection provider: {config.provider}, uri: {config.uri}, host: {config.host}")
                 raise ValueError("Invalid Milvus configuration for connection.")

            logger.info(f"Successfully connected to Milvus (alias: {self.milvus_alias}).")
            self.initialized_dbs[self.milvus_alias] = True # 标记为已初始化/连接

        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            self.initialized_dbs[self.milvus_alias] = False
            raise  # 重新引发异常，以便上层知道连接失败

    def _disconnect_milvus(self):
        """
        断开与Milvus实例的连接
        """
        if self.initialized_dbs.get(self.milvus_alias, False):
            try:
                logger.info(f"Disconnecting from Milvus (alias: {self.milvus_alias})...")
                connections.disconnect(alias=self.milvus_alias)
                self.initialized_dbs[self.milvus_alias] = False
                logger.info("Successfully disconnected from Milvus.")
            except Exception as e:
                logger.error(f"Error disconnecting from Milvus: {e}")
        else:
            logger.debug("Milvus connection alias not initialized or already disconnected.")
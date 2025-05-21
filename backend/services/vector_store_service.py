import os
from datetime import datetime
import json
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
from pymilvus import connections, utility, Collection, DataType, FieldSchema, CollectionSchema, MilvusException
from utils.config import VectorDBProvider, MILVUS_CONFIG, CHROMA_CONFIG, MILVUS_LITE_FILE, MILVUS_LITE_CONFIG, MILVUS_STANDALONE_CONFIG
import time
import math
import uuid
import numpy as np
import chromadb
from chromadb.config import Settings
from langchain.embeddings import OllamaEmbeddings
from pymilvus import MilvusClient

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
        self.metadata_dir = None

        if provider == "milvus_lite":
            config = MILVUS_LITE_CONFIG
            self.uri = config.get("uri", MILVUS_LITE_FILE)
            self.host = config.get("host", "localhost")
            self.port = config.get("port", 19530)
            self.index_params = MILVUS_CONFIG.get("index_params", {}).get(index_mode, {})
            self.metadata_dir = os.path.join(os.path.dirname(__file__), "..", "metadata", "milvus_lite")
            os.makedirs(os.path.dirname(self.uri), exist_ok=True)
            logger.info(f"Initialized Milvus Lite config with URI: {self.uri}")
        elif provider == "milvus_standalone":
            config = MILVUS_STANDALONE_CONFIG
            self.uri = config.get("uri", "tcp://localhost:19530")
            self.host = config.get("host", "localhost")
            self.port = config.get("port", 19530)
            self.index_params = MILVUS_CONFIG.get("index_params", {}).get(index_mode, {})
            self.metadata_dir = os.path.join(os.path.dirname(__file__), "..", "metadata", "milvus_standalone")
            logger.info(f"Initialized Milvus Standalone config with URI: {self.uri}")
        elif provider == VectorDBProvider.MILVUS.value:
            # 兼容原有 milvus
            self.uri = MILVUS_CONFIG.get("uri", MILVUS_LITE_FILE)
            self.host = MILVUS_CONFIG.get("host", "localhost")
            self.port = MILVUS_CONFIG.get("port", 19530)
            self.index_params = MILVUS_CONFIG.get("index_params", {}).get(index_mode, {})
            self.metadata_dir = os.path.join(os.path.dirname(__file__), "..", "metadata", "milvus")
            if not self.uri:
                self.uri = MILVUS_LITE_FILE
            os.makedirs(os.path.dirname(self.uri), exist_ok=True)
            logger.info(f"Initialized Milvus config with URI: {self.uri}")
        elif provider == VectorDBProvider.CHROMA.value:
            if CHROMA_CONFIG:
                self.uri = CHROMA_CONFIG.get("uri", "03-vector-store/chroma_db")
                self.collection_metadata = CHROMA_CONFIG.get("collection_metadata", {})
                self.index_params = CHROMA_CONFIG.get("index_params", {}).get(index_mode, {})
                self.metadata_dir = os.path.join(os.path.dirname(__file__), "..", "metadata", "chroma")
            else:
                self.uri = "03-vector-store/chroma_db"
                self.metadata_dir = os.path.join(os.path.dirname(__file__), "..", "metadata", "chroma")
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
        if index_mode == "flat":
            return "IVF_FLAT"  # Milvus Standalone 不支持 FLAT，使用 IVF_FLAT 替代
        elif index_mode in MILVUS_CONFIG["index_types"]:
            return MILVUS_CONFIG["index_types"].get(index_mode, "IVF_FLAT")
        return "IVF_FLAT"  # 默认返回 IVF_FLAT 索引类型
    
    def _get_milvus_index_params(self, index_mode: str) -> Dict[str, Any]:
        """
        从配置对象获取Milvus索引参数
        
        参数:
            index_mode: 索引模式
            
        返回:
            对应的Milvus索引参数字典
        """
        if index_mode == "flat":
            return {
                "nlist": 1024  # IVF_FLAT 的聚类中心数量
            }
        return MILVUS_CONFIG["index_params"].get(index_mode, {
            "nlist": 1024
        })


class VectorStoreService:
    """
    向量存储服务类，整合了 Chroma 和 Milvus 的功能
    
    提供以下功能：
    1. 向量索引：支持多种索引模式
    2. 向量搜索：支持相似度搜索
    3. 集合管理：支持创建、删除、查询集合
    4. 元数据管理：支持存储和查询文档元数据
    """
    
    def __init__(self):
        """
        初始化向量存储服务
        """
        self.initialized_dbs = {}
        self.provider = None
        self.milvus_alias = "default"
        self.milvus_uri = MILVUS_CONFIG.get("uri", MILVUS_LITE_FILE)
        self.metadata_file = os.path.join(os.path.dirname(__file__), "..", "collection_metadata.json")
        self._init_metadata_file()
        
        # 确保存储目录存在
        chroma_db_path = Path(CHROMA_CONFIG.get("uri", "03-vector-store/chroma_db"))
        if chroma_db_path:
             os.makedirs(chroma_db_path.parent, exist_ok=True)
        
        # 检查并初始化数据库
        if not self._check_and_init_db():
            raise RuntimeError("无法初始化 Milvus Lite 数据库")
        
        logger.info("VectorStoreService initialized. Connections will be established on demand.")
        
        self._chroma_client = None
        self._milvus_client = None
        self._embeddings = OllamaEmbeddings(
            model="bge-m3:latest",
            base_url="http://localhost:11434"
        )
    
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

    def _format_creation_time(self, creation_time: Optional[str]) -> str:
        """格式化创建时间
        
        Args:
            creation_time: ISO8601格式的时间字符串或None
            
        Returns:
            格式化后的时间字符串，格式为 'YYYY/MM/DD HH:mm:ss'
        """
        if not creation_time:
            return "N/A"
        
        try:
            # 尝试解析ISO8601格式
            if isinstance(creation_time, str):
                # 处理已经是目标格式的情况
                if len(creation_time) == 19 and creation_time[4] == '/' and creation_time[7] == '/':
                    return creation_time
                
                # 处理ISO8601格式
                dt = datetime.fromisoformat(creation_time.replace('Z', '+00:00'))
                return dt.strftime("%Y/%m/%d %H:%M:%S")
            elif isinstance(creation_time, datetime):
                return creation_time.strftime("%Y/%m/%d %H:%M:%S")
        except (ValueError, AttributeError) as e:
            logger.warning(f"无法解析创建时间 {creation_time}: {e}")
            return "N/A"
        
        return "N/A"

    def _save_collection_metadata(self, collection_name: str, metadata: dict, provider: str):
        """保存集合元数据"""
        try:
            # 确保元数据目录存在
            metadata_dir = os.path.join(os.path.dirname(__file__), "..", "metadata", provider)
            os.makedirs(metadata_dir, exist_ok=True)
            
            # 格式化创建时间
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 不含/和:
            metadata_path = f"{metadata_dir}/collection_metadata_{collection_name}_{timestamp}.json"
            
            # 确保目录存在
            os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
            
            # 保存元数据
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"[save_collection_metadata] provider={provider}, path={metadata_path}, metadata={metadata}")
        except Exception as e:
            logger.error(f"保存集合 '{collection_name}' 的元数据时出错: {e}", exc_info=True)
            raise

    def _get_collection_metadata(self, collection_name: str, provider: str) -> dict:
        """获取集合元数据"""
        try:
            metadata_dir = os.path.join(os.path.dirname(__file__), "..", "metadata", provider)
            if not os.path.exists(metadata_dir):
                return {}
            
            pattern = f"collection_metadata_{collection_name}"
            metadata_files = [f for f in os.listdir(metadata_dir) if f.startswith(pattern)]
            
            if not metadata_files:
                logger.warning(f"[get_collection_metadata] 未找到元数据文件: {pattern}")
                return {}
            
            # 按文件名排序，获取最新的元数据文件
            latest_file = sorted(metadata_files)[-1]
            metadata_path = os.path.join(metadata_dir, latest_file)
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                
                # 确保创建时间存在且格式正确
                if not metadata.get("creation_time"):
                    # 尝试从文件名中提取创建时间
                    try:
                        time_str = latest_file.split("_")[-1].replace(".json", "")
                        metadata["creation_time"] = datetime.fromisoformat(time_str).strftime("%Y/%m/%d %H:%M:%S")
                    except (ValueError, IndexError):
                        metadata["creation_time"] = "N/A"
                
                logger.info(f"[get_collection_metadata] provider={provider}, path={metadata_path}, metadata={metadata}")
                return metadata
        except Exception as e:
            logger.error(f"读取集合元数据时出错: {e}", exc_info=True)
            return {}
    
    def _get_milvus_index_type(self, config: VectorDBConfig) -> str:
        # 只允许 HNSW、DISKANN
        if config.provider == "milvus_standalone":
            if config.index_mode in ["hnsw", "diskann"]:
                return config.index_mode.upper()
            return "HNSW"
        # 其他 provider 保持原逻辑
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

    def _init_chroma_client(self):
        """初始化 Chroma 客户端"""
        if self._chroma_client is None:
            self._chroma_client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=CHROMA_CONFIG.get("uri", "03-vector-store/chroma_db")
            ))

    def _safe_creation_time(self, val):
        if not val or val == "N/A":
            return ""
        if isinstance(val, str) and "T" in val:
            return val
        if isinstance(val, datetime.datetime):
            return val.isoformat()
        return str(val)

    def list_collections(self, provider: str = "milvus") -> List[Dict[str, Any]]:
        """
        列出所有可用的集合
        """
        logger.info(f"[list_collections] 查询 provider: {provider}")
        try:
            if provider in ["milvus", "milvus_lite", "milvus_standalone"]:
                config = VectorDBConfig(provider=provider, index_mode='')
                try:
                    self._connect_milvus(config)
                except Exception as e:
                    logger.error(f"[list_collections] 连接 {provider} 失败: {e}")
                    return []
                collections = []
                try:
                    for name in utility.list_collections(using=self.milvus_alias):
                        try:
                            collection = Collection(name=name, using=self.milvus_alias)
                            collection.load()
                            metadata = self._get_collection_metadata(name, provider)
                            index_info = None
                            try:
                                index_info = collection.index().params
                            except Exception as e:
                                logger.warning(f"获取集合 {name} 的索引信息失败: {e}")
                            creation_time = self._format_creation_time(metadata.get("creation_time"))
                            collections.append({
                                "id": name,
                                "name": name,
                                "count": collection.num_entities,
                                "dimension": metadata.get("dimension") or collection.schema.fields[1].params.get("dim", 0),
                                "index_type": index_info.get("index_type") if index_info else metadata.get("index_type", "FLAT"),
                                "metric_type": index_info.get("metric_type") if index_info else metadata.get("metric_type", "L2"),
                                "creation_time": creation_time,
                                "embedding_model": metadata.get("embedding_model", "unknown"),
                                "source_file": metadata.get("source_file", "unknown"),
                                "provider": provider
                            })
                            collection.release()
                        except Exception as e:
                            logger.error(f"获取集合 {name} 的信息失败: {e}")
                    logger.info(f"[list_collections] provider={provider}, collections={collections}")
                    return collections
                except Exception as e:
                    logger.error(f"[list_collections] 遍历集合失败: {e}")
                    return []
            elif provider == VectorDBProvider.CHROMA.value:
                try:
                    self._init_chroma_client()
                    collections = []
                    for name in self._chroma_client.list_collections():
                        collection = self._chroma_client.get_collection(name)
                        metadata = collection.metadata or {}
                        collections.append({
                            "id": name,
                            "name": name,
                            "count": collection.count(),
                            "dimension": metadata.get("dimension", 0),
                            "index_type": metadata.get("index_type", "unknown"),
                            "metric_type": metadata.get("metric_type", "unknown"),
                            "creation_time": self._format_creation_time(metadata.get("creation_time")),
                            "embedding_model": metadata.get("embedding_model", "unknown"),
                            "source_file": metadata.get("source_file", "unknown"),
                            "provider": provider
                        })
                    logger.info(f"[list_collections] provider={provider}, collections={collections}")
                    return collections
                except Exception as e:
                    logger.error(f"[list_collections] CHROMA 遍历集合失败: {e}")
                    return []
            else:
                logger.error(f"[list_collections] 不支持的向量数据库提供商: {provider}")
                return []
        except Exception as e:
            logger.error(f"列出集合时出错: {e}")
            return []
        
    def get_collection_info(self, provider: str, collection_name: str) -> Dict[str, Any]:
        logger.info(f"[get_collection_info] provider={provider}, collection_name={collection_name}")
        try:
            if provider in ["milvus", "milvus_lite", "milvus_standalone"]:
                config = VectorDBConfig(provider=provider, index_mode='')
                self._connect_milvus(config)
                try:
                    collection = Collection(name=collection_name, using=self.milvus_alias)
                    collection.load()
                    metadata = self._get_collection_metadata(collection_name, provider)
                    num_entities = collection.num_entities
                    
                    # 格式化创建时间
                    creation_time = self._format_creation_time(metadata.get("creation_time"))
                    
                    info = {
                        "基本信息": {
                            "数据库类型": provider,
                            "集合名称": collection_name,
                            "向量总数": f"{num_entities:,}",
                            "创建时间": creation_time
                        },
                        "索引信息": {
                            "索引类型": metadata.get("index_type", "FLAT"),
                            "距离度量": metadata.get("metric_type", "L2"),
                            "索引参数": metadata.get("index_params", {})
                        },
                        "嵌入信息": {
                            "嵌入模型": metadata.get("embedding_model", "unknown"),
                            "源文件": metadata.get("source_file", "unknown")
                        }
                    }
                    logger.info(f"[get_collection_info] info={info}")
                    return info
                except Exception as e:
                    logger.error(f"获取集合信息时出错: {e}", exc_info=True)
                    raise
                finally:
                    if 'collection' in locals():
                        collection.release()
                    self._disconnect_milvus()
            elif provider == VectorDBProvider.CHROMA.value:
                self._init_chroma_client()
                collection = self._chroma_client.get_collection(collection_name)
                return {
                    "基本信息": {
                        "数据库类型": provider,
                        "集合名称": collection_name,
                        "向量总数": collection.count(),
                        "创建时间": self._format_creation_time(collection.metadata.get("creation_time"))
                    },
                    "元数据": collection.metadata
                }
            else:
                raise ValueError(f"不支持的向量数据库类型: {provider}")
        except Exception as e:
            logger.error(f"获取集合信息时出错: {e}", exc_info=True)
            raise

    def delete_collection(self, provider: str, collection_name: str) -> bool:
        """
        删除指定的集合
        """
        try:
            if provider == VectorDBProvider.CHROMA.value:
                self._init_chroma_client()
                self._chroma_client.delete_collection(collection_name)
                return True
            elif provider in ["milvus", "milvus_lite", "milvus_standalone"]:
                try:
                    self._disconnect_milvus()
                except Exception as e:
                    logger.warning(f"Error during disconnect: {e}", exc_info=True)
                
                config = VectorDBConfig(provider=provider, index_mode='')
                self._connect_milvus(config)
                
                try:
                    if not utility.has_collection(collection_name, using=self.milvus_alias):
                        logger.warning(f"Collection '{collection_name}' does not exist")
                        return True
                    
                    try:
                        collection = Collection(name=collection_name, using=self.milvus_alias)
                        collection.release()
                        logger.info(f"Released collection resources for '{collection_name}'")
                    except Exception as e:
                        logger.warning(f"Error releasing collection: {e}", exc_info=True)
                    
                    time.sleep(2)
                    
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
                                time.sleep(2)
                                self._disconnect_milvus()
                                self._connect_milvus(config)
                            else:
                                raise
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
        """连接到 Milvus 服务"""
        max_retries = 3
        retry_delay = 2  # 秒
        
        for attempt in range(max_retries):
            try:
                # 生成唯一的连接别名
                alias = f"{config.provider}_{config.uri.replace('://', '_').replace(':', '_')}"
                
                # 如果已经存在连接，先断开
                try:
                    connections.disconnect(alias=alias)
                    logger.info(f"[_connect_milvus] 已断开旧连接: {alias}")
                except Exception as e:
                    logger.debug(f"[_connect_milvus] 断开旧连接时出错: {e}")
                
                # 检查并删除可能存在的锁文件
                if config.provider in ["milvus", "milvus_lite"]:
                    lock_file = os.path.join(os.path.dirname(config.uri), ".milvus_lite.db.lock")
                    if os.path.exists(lock_file):
                        try:
                            os.remove(lock_file)
                            logger.info(f"[_connect_milvus] 已删除锁文件: {lock_file}")
                        except Exception as e:
                            logger.warning(f"[_connect_milvus] 删除锁文件失败: {e}")
                
                # 确保使用正确的连接参数
                if config.provider == "milvus_standalone":
                    uri = config.uri or "tcp://localhost:19530"
                    logger.info(f"[_connect_milvus] 正在连接到 Milvus Standalone，URI: {uri}")
                    
                    if not uri.startswith("tcp://"):
                        raise ValueError(f"Milvus Standalone URI 必须以 tcp:// 开头，当前: {uri}")
                        
                    host_port = uri.replace("tcp://", "")
                    host, port = host_port.split(":")
                    
                    connections.connect(
                        alias=alias,
                        host=host,
                        port=int(port)
                    )
                    logger.info(f"[_connect_milvus] 成功连接到 Milvus Standalone (alias: {alias})")
                    
                elif config.provider in ["milvus", "milvus_lite"]:
                    uri = config.uri or os.path.join(os.path.dirname(__file__), "..", "milvus_lite.db")
                    logger.info(f"[_connect_milvus] 正在连接到 Milvus Lite，URI: {uri}")
                    
                    connections.connect(
                        alias=alias,
                        uri=uri
                    )
                    logger.info(f"[_connect_milvus] 成功连接到 Milvus Lite (alias: {alias})")
                    
                else:
                    raise ValueError(f"不支持的 Milvus provider: {config.provider}")
                
                # 保存当前使用的别名
                self.milvus_alias = alias
                
                # 测试连接
                utility.list_collections(using=alias)
                logger.info(f"[_connect_milvus] 连接测试成功")
                
                return  # 连接成功，退出重试循环
                
            except Exception as e:
                logger.error(f"[_connect_milvus] 第 {attempt + 1} 次连接失败: {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"[_connect_milvus] {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                    continue
                else:
                    raise RuntimeError(f"连接到 Milvus 失败，已重试 {max_retries} 次: {str(e)}")

    def _disconnect_milvus(self):
        """断开与 Milvus 的连接"""
        try:
            if hasattr(self, 'milvus_alias'):
                logger.info(f"正在断开 Milvus 连接 (alias: {self.milvus_alias})")
                connections.disconnect(alias=self.milvus_alias)
                
                # 删除锁文件
                if hasattr(self, 'provider') and self.provider in ["milvus", "milvus_lite"]:
                    lock_file = os.path.join(os.path.dirname(self.milvus_uri), ".milvus_lite.db.lock")
                    if os.path.exists(lock_file):
                        try:
                            os.remove(lock_file)
                            logger.info(f"已删除锁文件: {lock_file}")
                        except Exception as e:
                            logger.warning(f"删除锁文件失败: {e}")
        except Exception as e:
            logger.warning(f"断开 Milvus 连接时出错: {e}")
            # 不抛出异常，因为这是清理操作

    def search(self, query: str, collection_name: str, provider: str, top_k: int = 5, threshold: float = 0.7) -> Dict:
        """
        搜索相似文档
        
        参数:
            query: 查询文本
            collection_name: 集合名称
            provider: 向量数据库提供商
            top_k: 返回结果数量
            threshold: 相似度阈值
            
        返回:
            搜索结果
        """
        try:
            # 获取查询向量
            query_embedding = self._embeddings.embed_query(query)
            
            if provider == "chroma":
                return self._search_in_chroma(query_embedding, collection_name, top_k)
            elif provider == "milvus":
                return self._search_in_milvus(query_embedding, collection_name, top_k)
            else:
                raise ValueError(f"不支持的向量数据库提供商: {provider}")
        except Exception as e:
            logger.error(f"搜索失败: {str(e)}")
            raise

    def _search_in_chroma(self, query_embedding: List[float], collection_name: str, top_k: int) -> Dict:
        """在 Chroma 中搜索"""
        try:
            self._init_chroma_client()
            collection = self._chroma_client.get_collection(collection_name)
            
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            # 处理结果
            processed_results = []
            if results["documents"] and len(results["documents"]) > 0:
                for i, doc in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    processed_results.append({
                        "document": doc,
                        "distance": float(results["distances"][0][i]) if results["distances"] else 0.0,
                        "metadata": metadata
                    })
            
            return {
                "results": processed_results,
                "total": len(processed_results)
            }
            
        except Exception as e:
            logger.error(f"Chroma 搜索失败: {str(e)}")
            raise

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

    def get_collection_schema(self, collection_name: str) -> Dict:
        """获取集合的schema"""
        try:
            # 连接到Milvus
            self._connect_milvus(VectorDBConfig(provider="milvus", index_mode="flat"))
            
            # 获取集合
            collection = Collection(name=collection_name, using=self.milvus_alias)
            schema = collection.schema
            return schema
        except Exception as e:
            logger.error(f"获取集合schema失败: {str(e)}")
            raise
        finally:
            self._disconnect_milvus()

    def index_embeddings(self, embeddings_data, vector_db_config):
        """索引嵌入向量到向量数据库"""
        try:
            provider = vector_db_config.provider
            if provider == "milvus_lite":
                return self._index_embeddings_milvus_lite(embeddings_data, vector_db_config)
            elif provider == "milvus_standalone":
                return self._index_embeddings_milvus_standalone(embeddings_data, vector_db_config)
            elif provider == "chroma":
                return self._index_embeddings_chroma(embeddings_data, vector_db_config)
            else:
                raise ValueError(f"不支持的向量数据库类型: {provider}")
        except Exception as e:
            logger.error(f"索引嵌入向量时出错: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"索引嵌入向量时出错: {str(e)}",
                "details": {}
            }

    def _index_embeddings_milvus_lite(self, embeddings_data, config: VectorDBConfig) -> dict:
        """使用 Milvus Lite 索引嵌入向量"""
        try:
            logger.info(f"[_index_embeddings_milvus_lite] 开始索引，provider={config.provider}, collection={config.target_collection_name}")

            # 先连接，确保 self.milvus_alias 正确
            self._connect_milvus(config)

            # 确保 vectors 是二维数组
            vectors = embeddings_data.get("vectors") or embeddings_data.get("embeddings") or []
            if not vectors:
                raise ValueError("嵌入数据中没有向量")
            
            # 如果 vectors 是字典，转换为数组
            if isinstance(vectors, dict):
                vectors = list(vectors.values())
            
            # 确保 vectors 是二维数组
            if not isinstance(vectors, list) or not all(isinstance(v, list) for v in vectors):
                raise ValueError("向量数据格式错误，应为二维数组")

            dimension = len(vectors[0])
            logger.info(f"[_index_embeddings_milvus_lite] 向量维度: {dimension}")

            logger.info(f"【vector_store_service.py】收到的vectors类型: {type(vectors)}")
            logger.info(f"【vector_store_service.py】vectors长度: {len(vectors)}")
            if isinstance(vectors, list) and len(vectors) > 0:
                logger.info(f"【vector_store_service.py】第一个向量类型: {type(vectors[0])}, 内容: {vectors[0]}")
                if isinstance(vectors[0], list):
                    logger.info(f"【vector_store_service.py】第一个向量长度: {len(vectors[0])}")

            collection_name = config.target_collection_name
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension)
            ]
            schema = CollectionSchema(fields=fields, description=f"Collection for {collection_name}")

            # 检查集合是否存在，使用别名
            if utility.has_collection(collection_name, using=self.milvus_alias):
                utility.drop_collection(collection_name, using=self.milvus_alias)
                logger.info(f"[_index_embeddings_milvus_lite] 已删除已存在的集合: {collection_name}")

            # 创建集合，指定 using
            collection = Collection(name=collection_name, schema=schema, using=self.milvus_alias)
            logger.info(f"[_index_embeddings_milvus_lite] 成功创建集合: {collection_name}")

            index_params = {
                "index_type": config.index_mode or "FLAT",
                "metric_type": "L2",
                "params": {}
            }
            collection.create_index(field_name="vector", index_params=index_params)
            logger.info(f"[_index_embeddings_milvus_lite] 成功创建索引: {index_params}")

            collection.load()
            logger.info("[_index_embeddings_milvus_lite] 成功加载集合")

            # 构造正确的插入数据格式
            insert_data = [{"vector": vector} for vector in vectors]
            logger.info(f"[_index_embeddings_milvus_lite] 开始插入向量，数量: {len(vectors)}，连接别名: {self.milvus_alias}")
            collection.insert(insert_data)
            logger.info(f"[_index_embeddings_milvus_lite] 成功插入 {len(vectors)} 个向量")

            metadata = {
                "dimension": dimension,
                "creation_time": datetime.now().isoformat(),
                "index_type": index_params["index_type"],
                "metric_type": index_params["metric_type"],
                "embedding_model": embeddings_data.get("metadata", {}).get("embedding_model", "unknown"),
                "source_file": embeddings_data.get("source_file", "unknown"),
                "index_params": index_params
            }
            self._save_collection_metadata(collection_name, metadata, config.provider)
            logger.info(f"[_index_embeddings_milvus_lite] 成功保存元数据")

            return {
                "status": "success",
                "message": f"成功索引 {len(vectors)} 个向量到集合 {collection_name}",
                "provider": config.provider,
                "count": len(vectors),
                "total_vectors": len(vectors),
                "details": {
                    "collection_name": collection_name,
                    "vector_count": len(vectors),
                    "dimension": dimension,
                    "index_type": index_params["index_type"],
                    "provider": config.provider
                }
            }
        except Exception as e:
            error_msg = f"索引向量时出错: {str(e)}"
            logger.error(f"[_index_embeddings_milvus_lite] {error_msg}")
            return {
                "status": "error",
                "message": error_msg,
                "details": {}
            }
        finally:
            try:
                if 'collection' in locals():
                    collection.release()
            except Exception:
                pass
            self._disconnect_milvus()

    def _index_embeddings_milvus_standalone(self, embeddings_data, config: VectorDBConfig) -> dict:
        """使用 Milvus Standalone 索引嵌入向量"""
        try:
            logger.info(f"[_index_embeddings_milvus_standalone] 开始索引，provider={config.provider}, collection={config.target_collection_name}")

            # 先连接，确保 self.milvus_alias 正确
            self._connect_milvus(config)

            # 确保 vectors 是二维数组
            vectors = embeddings_data.get("vectors") or embeddings_data.get("embeddings") or []
            if not vectors:
                raise ValueError("嵌入数据中没有向量")
            
            # 如果 vectors 是字典，转换为数组
            if isinstance(vectors, dict):
                vectors = list(vectors.values())
            
            # 如果 vectors 是字符串，尝试解析 JSON
            if isinstance(vectors, str):
                try:
                    vectors = json.loads(vectors)
                except json.JSONDecodeError:
                    raise ValueError("向量数据格式错误，无法解析 JSON")
            
            # 确保 vectors 是列表
            if not isinstance(vectors, list):
                raise ValueError("向量数据格式错误，应为列表")
            
            # 如果 vectors 是嵌套字典，转换为二维数组
            if vectors and isinstance(vectors[0], dict):
                vectors = [v.get("vector", v.get("embedding", [])) for v in vectors]
            
            # 确保所有向量都是列表
            vectors = [v if isinstance(v, list) else [] for v in vectors]
            
            # 过滤掉空向量
            vectors = [v for v in vectors if v]
            
            # 确保 vectors 是二维数组
            if not vectors or not all(isinstance(v, list) for v in vectors):
                raise ValueError("向量数据格式错误，应为二维数组")
            
            # 确保所有向量维度一致
            dimension = len(vectors[0])
            if not all(len(v) == dimension for v in vectors):
                raise ValueError("向量维度不一致")

            logger.info(f"[_index_embeddings_milvus_standalone] 向量维度: {dimension}, 向量数量: {len(vectors)}")

            logger.info(f"【vector_store_service.py】收到的vectors类型: {type(vectors)}")
            logger.info(f"【vector_store_service.py】vectors长度: {len(vectors)}")
            if isinstance(vectors, list) and len(vectors) > 0:
                logger.info(f"【vector_store_service.py】第一个向量类型: {type(vectors[0])}, 内容: {vectors[0]}")
                if isinstance(vectors[0], list):
                    logger.info(f"【vector_store_service.py】第一个向量长度: {len(vectors[0])}")

            collection_name = config.target_collection_name
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension)
            ]
            schema = CollectionSchema(fields=fields, description=f"Collection for {collection_name}")

            # 检查集合是否存在，使用别名
            if utility.has_collection(collection_name, using=self.milvus_alias):
                utility.drop_collection(collection_name, using=self.milvus_alias)
                logger.info(f"[_index_embeddings_milvus_standalone] 已删除已存在的集合: {collection_name}")

            # 创建集合，指定 using
            collection = Collection(name=collection_name, schema=schema, using=self.milvus_alias)
            logger.info(f"[_index_embeddings_milvus_standalone] 成功创建集合: {collection_name}")

            index_params = {
                "index_type": config.index_mode or "FLAT",
                "metric_type": "L2",
                "params": {}
            }
            collection.create_index(field_name="vector", index_params=index_params)
            logger.info(f"[_index_embeddings_milvus_standalone] 成功创建索引: {index_params}")

            collection.load()
            logger.info("[_index_embeddings_milvus_standalone] 成功加载集合")

            # 构造正确的插入数据格式
            insert_data = [{"vector": vector} for vector in vectors]
            logger.info(f"[_index_embeddings_milvus_standalone] 开始插入向量，数量: {len(vectors)}，连接别名: {self.milvus_alias}")
            collection.insert(insert_data)
            logger.info(f"[_index_embeddings_milvus_standalone] 成功插入 {len(vectors)} 个向量")

            metadata = {
                "dimension": dimension,
                "creation_time": datetime.now().isoformat(),
                "index_type": index_params["index_type"],
                "metric_type": index_params["metric_type"],
                "embedding_model": embeddings_data.get("metadata", {}).get("embedding_model", "unknown"),
                "source_file": embeddings_data.get("source_file", "unknown"),
                "index_params": index_params
            }
            self._save_collection_metadata(collection_name, metadata, config.provider)
            logger.info(f"[_index_embeddings_milvus_standalone] 成功保存元数据")

            return {
                "status": "success",
                "message": f"成功索引 {len(vectors)} 个向量到集合 {collection_name}",
                "provider": config.provider,
                "count": len(vectors),
                "total_vectors": len(vectors),
                "details": {
                    "collection_name": collection_name,
                    "vector_count": len(vectors),
                    "dimension": dimension,
                    "index_type": index_params["index_type"],
                    "provider": config.provider
                }
            }
        except Exception as e:
            error_msg = f"索引向量时出错: {str(e)}"
            logger.error(f"[_index_embeddings_milvus_standalone] {error_msg}")
            return {
                "status": "error",
                "message": error_msg,
                "details": {}
            }
        finally:
            try:
                if 'collection' in locals():
                    collection.release()
            except Exception:
                pass
            self._disconnect_milvus()

    def _index_embeddings_chroma(self, embeddings_data, vector_db_config):
        # TODO: 实现 chroma 的向量插入逻辑
        logger.info(f"[Mock] Chroma 索引: collection={vector_db_config.target_collection_name}, 向量数={len(embeddings_data.get('embeddings', []))}")
        total_vectors = len(embeddings_data.get('embeddings', []))
        return {
            "status": "success",
            "provider": "chroma",
            "count": total_vectors,
            "total_vectors": total_vectors,
            "details": []
        }

    def _get_metadata_path(self, collection_name: str, provider: str, creation_time: str = None) -> str:
        """获取元数据文件路径"""
        if creation_time:
            filename = f"collection_metadata_{collection_name}_{creation_time}.json"
        else:
            filename = f"collection_metadata_{collection_name}.json"
        return os.path.join(os.path.dirname(__file__), "..", "metadata", provider, filename)
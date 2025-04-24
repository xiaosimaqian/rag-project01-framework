from enum import Enum
from typing import Dict, Any
import os
from pathlib import Path

class VectorDBProvider(str, Enum):
    MILVUS = "milvus"
    CHROMA = "chroma"
    OLLAMA = "ollama"
    # More providers can be added later

class VectorDBConfig:
    """向量数据库配置类"""
    def __init__(
        self,
        provider: str,
        index_mode: str = "flat",
        target_collection_name: str = None,
        rebuild_index: bool = False
    ):
        """
        初始化向量数据库配置
        
        参数:
            provider: 数据库提供商 (milvus, chroma)
            index_mode: 索引模式 (flat, ivf_flat, ivf_sq8, hnsw)，默认为 flat
            target_collection_name: 目标集合名称（可选）
            rebuild_index: 是否重建索引（可选）
        """
        self.provider = provider
        self.index_mode = index_mode
        self.target_collection_name = target_collection_name
        self.rebuild_index = rebuild_index
        
        # 根据提供商设置 URI
        if provider == VectorDBProvider.MILVUS.value:
            self.uri = MILVUS_CONFIG["uri"]
        elif provider == VectorDBProvider.CHROMA.value:
            self.uri = CHROMA_CONFIG["uri"]
        else:
            self.uri = None
            
    def _get_milvus_index_type(self, index_mode: str) -> str:
        """获取 Milvus 索引类型"""
        return MILVUS_CONFIG["index_types"].get(index_mode, "FLAT")
        
    def _get_milvus_index_params(self, index_mode: str) -> Dict[str, Any]:
        """获取 Milvus 索引参数"""
        return MILVUS_CONFIG["index_params"].get(index_mode, {})
        
    def _get_chroma_index_params(self, index_mode: str) -> Dict[str, Any]:
        """获取 Chroma 索引参数"""
        return CHROMA_CONFIG["index_params"].get(index_mode, {})

# 可以在这里添加其他配置相关的内容
MILVUS_LITE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "milvus_lite.db")

MILVUS_CONFIG = {
    "host": "localhost",  # 或者使用环境变量
    "port": 19530,       # 或者使用环境变量
    "uri": MILVUS_LITE_FILE,
    "index_types": {
        "flat": "FLAT",
        "ivf_flat": "IVF_FLAT",
        "ivf_sq8": "IVF_SQ8",
        "hnsw": "HNSW"
    },
    "index_params": {
        "flat": {},
        "ivf_flat": {"nlist": 1024},
        "ivf_sq8": {"nlist": 1024},
        "hnsw": {
            "M": 16,
            "efConstruction": 500
        }
    }
} 

CHROMA_CONFIG = {
    "uri": "03-vector-store/chroma_db",
    "settings": {
        "chroma_db_impl": "duckdb+parquet",
        "persist_directory": "03-vector-store/chroma_db",
        "anonymized_telemetry": False
    },
    "collection_metadata": {
        "hnsw:space": "cosine",
        "hnsw:M": 8,
        "hnsw:construction_ef": 100,
        "hnsw:search_ef": 10
    },
    "index_params": {
        "hnsw": {
            "M": 8,
            "construction_ef": 100,
            "search_ef": 10
        }
    }
} 
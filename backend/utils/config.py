from enum import Enum
from typing import Dict, Any

class VectorDBProvider(str, Enum):
    MILVUS = "milvus"
    CHROMA = "chroma"
    OLLAMA = "ollama"
    # More providers can be added later

# 可以在这里添加其他配置相关的内容
MILVUS_CONFIG = {
    "host": "localhost",  # 或者使用环境变量
    "port": 19530,       # 或者使用环境变量
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
import os
import json
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body, Query, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from services.loading_service import LoadingService
from services.chunking_service import ChunkingService
from services.embedding_service import EmbeddingService, EmbeddingConfig
from services.vector_store_service import VectorStoreService, VectorDBConfig, VectorDBProvider
from services.search_service import SearchService
from services.parsing_service import ParsingService
import logging
from enum import Enum
from utils.config import VectorDBProvider, MILVUS_CONFIG, CHROMA_CONFIG
import pandas as pd
from pathlib import Path
from services.generation_service import GenerationService
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
import uuid
import shutil
import asyncio
import aiohttp
from openai import AsyncOpenAI

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
vector_store_service = None
search_service = None

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# 配置请求体大小限制
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# 创建上传文件存储目录
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# 文件存储映射
file_storage = {}

# 文件存储信息文件路径
STORAGE_FILE = Path("file_storage.json")

# 配置上传参数
CHUNK_SIZE = 1024 * 1024  # 1MB chunks
MAX_UPLOAD_SIZE = 1024 * 1024 * 1024  # 1GB
UPLOAD_TIMEOUT = 300  # 5 minutes

# API限制
MAX_CONTEXT_SIZE = 1024 * 1024 * 1024  # 1GB
MAX_FILE_SIZE = 1024 * 1024 * 1024  # 1GB per file
MAX_FILES = 10  # 最多10个文件

# Ollama API配置
OLLAMA_API_BASE = "http://localhost:11434"  # Ollama默认地址

# 添加分块处理相关的常量
MAX_CHUNK_SIZE = 50 * 1024 * 1024  # 50MB per chunk
MAX_CHUNKS = 100  # 最多100个块

class GenerationRequest(BaseModel):
    provider: str
    model_name: str
    query: str
    api_key: Optional[str] = None
    show_reasoning: bool = True
    context_file_ids: List[str] = Field(default_factory=list, max_items=MAX_FILES)
    context_contents: Optional[List[str]] = Field(default=None, max_items=MAX_FILES)
    search_results: Optional[List[dict]] = None
    collection_name: Optional[str] = None

    class Config:
        extra = "allow"  # 允许额外的字段

class FileInfo(BaseModel):
    file_id: str
    name: str
    size: int
    upload_time: datetime
    status: str
    used_count: int = 0

def save_file_storage():
    """保存file_storage到文件"""
    try:
        # 将datetime对象转换为字符串
        storage_data = {}
        for file_id, info in file_storage.items():
            storage_data[file_id] = {
                "path": info["path"],
                "name": info["name"],
                "size": info["size"],
                "upload_time": info["upload_time"].isoformat(),
                "status": info["status"],
                "used_count": info["used_count"]
            }
        
        with open(STORAGE_FILE, "w", encoding="utf-8") as f:
            json.dump(storage_data, f, ensure_ascii=False, indent=2)
        logger.info(f"file_storage已保存到文件: {STORAGE_FILE}")
    except Exception as e:
        logger.error(f"保存file_storage时出错: {str(e)}")

def load_file_storage():
    """从文件加载file_storage"""
    try:
        if STORAGE_FILE.exists():
            with open(STORAGE_FILE, "r", encoding="utf-8") as f:
                storage_data = json.load(f)
            
            # 将字符串转换回datetime对象
            for file_id, info in storage_data.items():
                file_storage[file_id] = {
                    "path": info["path"],
                    "name": info["name"],
                    "size": info["size"],
                    "upload_time": datetime.fromisoformat(info["upload_time"]),
                    "status": info["status"],
                    "used_count": info["used_count"]
                }
            logger.info(f"从文件加载了 {len(file_storage)} 个文件信息")
        else:
            logger.info("file_storage文件不存在，将创建新文件")
    except Exception as e:
        logger.error(f"加载file_storage时出错: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """在应用启动时初始化服务"""
    global vector_store_service, search_service
    try:
        # 初始化向量存储服务
        vector_store_service = VectorStoreService()
        search_service = SearchService()
        collections = vector_store_service.list_collections(provider=VectorDBProvider.MILVUS.value)
        logger.info(f"应用启动时找到以下集合：{collections}")
        
        # 加载文件存储信息
        logger.info("开始加载file_storage")
        logger.info(f"uploads目录: {UPLOAD_DIR}")
        logger.info(f"uploads目录是否存在: {UPLOAD_DIR.exists()}")
        logger.info(f"uploads目录中的文件: {list(UPLOAD_DIR.glob('*'))}")
        
        # 首先尝试从file_storage.json加载
        if STORAGE_FILE.exists():
            try:
                logger.info(f"尝试从 {STORAGE_FILE} 加载文件信息")
                with open(STORAGE_FILE, "r", encoding="utf-8") as f:
                    storage_data = json.load(f)
                
                # 将字符串转换回datetime对象
                for file_id, info in storage_data.items():
                    file_storage[file_id] = {
                        "path": info["path"],
                        "name": info["name"],
                        "size": info["size"],
                        "upload_time": datetime.fromisoformat(info["upload_time"]),
                        "status": info["status"],
                        "used_count": info["used_count"]
                    }
                logger.info(f"从file_storage.json加载了 {len(file_storage)} 个文件信息")
            except Exception as e:
                logger.error(f"从file_storage.json加载失败: {str(e)}")
                # 如果加载失败，清空file_storage
                file_storage.clear()
        
        # 如果file_storage为空，从uploads目录恢复
        if not file_storage:
            logger.info("开始从uploads目录恢复file_storage")
            for file_path in UPLOAD_DIR.glob("*"):
                if file_path.is_file():
                    try:
                        # 生成文件ID（使用文件名作为ID）
                        file_id = str(uuid.uuid4())
                        
                        # 获取文件信息
                        file_size = file_path.stat().st_size
                        file_name = file_path.name
                        
                        # 存储文件信息
                        file_storage[file_id] = {
                            "path": str(file_path),
                            "name": file_name,
                            "size": file_size,
                            "upload_time": datetime.now(),
                            "status": "active",
                            "used_count": 0
                        }
                        
                        logger.info(f"恢复文件信息: {file_name} -> {file_id}")
                    except Exception as e:
                        logger.error(f"恢复文件 {file_path} 时出错: {str(e)}")
                        continue
            
            # 保存file_storage到文件
            save_file_storage()
            
            logger.info(f"从uploads目录恢复了 {len(file_storage)} 个文件信息")
        
        logger.info(f"file_storage加载完成，共 {len(file_storage)} 个文件")
        logger.info(f"file_storage内容: {file_storage}")
    except Exception as e:
        logger.error(f"应用启动时初始化服务出错: {e}")
        # 不要在这里抛出异常，让应用继续启动

@app.on_event("shutdown")
async def shutdown_event():
    """在应用关闭时清理资源"""
    global vector_store_service
    try:
        # 断开 Milvus 连接
        if vector_store_service:
            try:
                vector_store_service._disconnect_milvus()
                logger.info("应用关闭时成功断开 Milvus 连接")
            except Exception as e:
                logger.warning(f"应用关闭时断开连接出错: {e}")
        
        # 保存file_storage到文件
        logger.info("开始保存file_storage")
        save_file_storage()
        logger.info("file_storage保存完成")
    except Exception as e:
        logger.error(f"应用关闭时出错: {e}")

@app.get("/")
async def root():
    """根路径处理函数"""
    return {"message": "Welcome to RAG Framework API"}

@app.post("/process")
async def process_document(
    file: UploadFile,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    embedding_provider: str = "ollama",
    embedding_model: str = "bge-m3:latest",
    vector_db_provider: str = "milvus",
    index_mode: str = "flat"
):
    """处理上传的文档"""
    try:
        # 1. 保存文件
        file_path = os.path.join("01-original-docs", file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        # 2. 创建嵌入配置
        embedding_config = EmbeddingConfig(
            provider=embedding_provider,
            model_name=embedding_model
        )
        
        # 3. 处理文档
        processor = DocumentProcessor()
        chunks = processor.process_document(
            file_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # 4. 创建嵌入
        embedding_service = EmbeddingService()
        embeddings = await embedding_service.create_embeddings(chunks, embedding_config)
        
        # 5. 保存嵌入结果
        os.makedirs("02-embedded-docs", exist_ok=True)
        output_path = os.path.join(
            "02-embedded-docs",
            f"{os.path.splitext(file.filename)[0]}_embeddings.json"
        )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(embeddings, f, ensure_ascii=False, indent=2)
            
        # 6. 创建向量数据库配置
        vector_db_config = VectorDBConfig(
            provider=vector_db_provider,
            index_mode=index_mode
        )
        
        # 7. 索引到向量数据库
        vector_store = VectorStoreService()
        index_result = vector_store.index_embeddings(output_path, vector_db_config)
        
        return {
            "message": "文档处理和索引成功",
            "embeddings_file": output_path,
            "index_result": index_result
        }
        
    except Exception as e:
        logger.error(f"处理文档时出错: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/save")
async def save_chunks(data: dict):
    try:
        doc_name = data.get("docName")
        chunks = data.get("chunks")
        metadata = data.get("metadata", {})
        
        if not doc_name or not chunks:
            raise ValueError("Missing required fields")
        
        # 构建文件名
        filename = f"{doc_name}.json"
        filepath = os.path.join("01-chunked-docs", filename)
        
        # 保存数据
        document_data = {
            "document_name": doc_name,
            "metadata": metadata,
            "chunks": chunks
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(document_data, f, ensure_ascii=False, indent=2)
        
        return {
            "status": "success",
            "message": "Document saved successfully",
            "filepath": filepath
        }
    except Exception as e:
        logger.error(f"Error saving document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/list-docs")
async def list_documents():
    try:
        docs = []
        docs_dir = "01-chunked-docs"
        for filename in os.listdir(docs_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(docs_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    doc_data = json.load(f)
                    docs.append({
                        "id": filename,
                        "name": doc_data["document_name"]
                    })
        return {"documents": docs}
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise

@app.post("/embed")
async def embed_document(
    data: dict = Body(...)
):
    try:
        doc_name = data.get("docName")
        doc_type = data.get("docType", "chunked")  # 默认为 chunked 类型
        embedding_config = data.get("embeddingConfig", {})
        
        if not doc_name:
            raise ValueError("文档名称不能为空")
            
        # 使用绝对路径
        chunked_dir = os.path.join(BASE_DIR, "01-chunked-docs")
        file_path = os.path.join(chunked_dir, f"{doc_name}.json")
        
        # 如果文件不存在，尝试查找解析后的文件
        if not os.path.exists(file_path):
            # 查找以 "parsed_" 开头且以 ".json" 结尾的文件
            for filename in os.listdir(chunked_dir):
                if filename.startswith(f"parsed_{doc_name}") and filename.endswith(".json"):
                    file_path = os.path.join(chunked_dir, filename)
                    break
                    
        if not os.path.exists(file_path):
            # 记录更多信息以便调试
            logger.error(f"找不到文档，搜索路径: {file_path}")
            logger.error(f"目录内容: {os.listdir(chunked_dir)}")
            raise HTTPException(
                status_code=404,
                detail=f"找不到文档: {doc_name}"
            )
            
        # 读取文档内容
        with open(file_path, 'r', encoding='utf-8') as f:
            doc_data = json.load(f)
            
        # 准备元数据和文本块
        metadata = doc_data.get("metadata", {})
        chunks = []
        
        # 根据文档格式处理文本块
        if "chunks" in doc_data:
            # 标准格式
            chunks = doc_data["chunks"]
        elif "content" in doc_data:
            # parsed 格式
            chunks = [
                {
                    "content": item["content"],
                    "metadata": {
                        "chunk_id": idx + 1,
                        "page_number": item.get("page", 0),
                        "page_range": str(item.get("page", 0)),
                        "word_count": len(item["content"].split())
                    }
                }
                for idx, item in enumerate(doc_data["content"])
            ]
        
        if not chunks:
            raise ValueError("文档没有可用的文本块")
            
        # 创建嵌入配置
        embedding_config = EmbeddingConfig(
            provider=embedding_config.get("provider", "ollama"),
            model_name=embedding_config.get("model", "bge-m3:latest")
        )
        
        # 创建嵌入
        embedding_service = EmbeddingService()
        embeddings = await embedding_service.create_embeddings(
            chunks=chunks,
            metadata=metadata,
            doc_timestamp=datetime.now().strftime("%Y%m%d%H%M%S")
        )
        
        # 保存嵌入结果
        output_dir = os.path.join(BASE_DIR, "02-embedded-docs")
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(
            output_dir,
            f"{doc_name}_embeddings.json"
        )
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(embeddings, f, ensure_ascii=False, indent=2)
            
        return {
            "status": "success",
            "message": f"成功为文档 {doc_name} 创建嵌入",
            "filepath": output_path,
            "embeddings": embeddings
        }
        
    except Exception as e:
        logger.error(f"创建嵌入时出错: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/list-embedded")
async def list_embedded_docs():
    """List all embedded documents"""
    try:
        documents = []
        embedded_dir = os.path.join("02-embedded-docs")
        logger.info(f"Scanning directory: {embedded_dir}")
        
        if not os.path.exists(embedded_dir):
            logger.warning(f"Directory {embedded_dir} does not exist")
            return {"documents": []}
            
        for filename in os.listdir(embedded_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(embedded_dir, filename)
                logger.info(f"Reading file: {file_path}")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # 从metadata字段中获取信息
                        metadata = data.get("metadata", {})
                        doc_info = {
                            "name": filename,
                            "metadata": {
                                "document_name": metadata.get("document_name", filename),
                                "embedding_model": metadata.get("embedding_model", "unknown"),
                                "embedding_provider": metadata.get("embedding_provider", "unknown"),
                                "embedding_timestamp": metadata.get("created_at", ""),
                                "vector_dimension": metadata.get("vector_dimension", 0)
                            }
                        }
                        logger.info(f"Added document info: {doc_info}")
                        documents.append(doc_info)
                except Exception as e:
                    logger.error(f"Error reading file {file_path}: {str(e)}")
                    
        logger.info(f"Total documents found: {len(documents)}")
        return {"documents": documents}
    except Exception as e:
        logger.error(f"Error listing embedded documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index")
async def index_document(
    data: dict = Body(...)
):
    try:
        # 从请求中获取参数
        embeddings_file = data.get("embeddingsFile")
        if not embeddings_file:
            raise HTTPException(status_code=400, detail="未提供embeddingsFile参数")
            
        # 构建完整的文件路径
        embeddings_path = os.path.join(os.path.dirname(__file__), "02-embedded-docs", embeddings_file)
        if not os.path.exists(embeddings_path):
            raise HTTPException(status_code=404, detail=f"嵌入文件不存在: {embeddings_file}")
            
        # 读取嵌入文件
        with open(embeddings_path, 'r', encoding='utf-8') as f:
            embeddings_data = json.load(f)
            
        # 获取集合名称
        collection_name = data.get("targetCollectionName")
        if not collection_name:
            # 如果没有指定集合名称，使用嵌入文件的名称（去掉.json后缀）
            collection_name = os.path.splitext(embeddings_file)[0]
            
        # 创建向量数据库配置
        vector_db_config = VectorDBConfig(
            provider="milvus",
            index_mode="HNSW",
            target_collection_name=collection_name
        )
        
        # 创建向量存储服务实例
        vector_store = VectorStoreService()
        
        # 执行索引
        result = vector_store.index_embeddings(embeddings_data, vector_db_config)
        
        # 验证返回的数据结构
        if not isinstance(result, dict):
            raise ValueError("索引结果格式错误")
            
        if "details" not in result:
            raise ValueError("索引结果缺少details字段")
            
        if "total_vectors" not in result["details"]:
            raise ValueError("索引结果缺少total_vectors字段")
            
        if result["details"]["total_vectors"] == 0:
            raise ValueError("索引结果中向量数量为0")
            
        return {
            "status": "success",
            "message": "文档索引成功",
            "data": result
        }
        
    except Exception as e:
        logger.error(f"索引文档失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/providers")
async def get_providers():
    """获取支持的向量数据库列表"""
    try:
        search_service = SearchService()
        providers = search_service.get_providers()
        return {"providers": providers}
    except Exception as e:
        logger.error(f"Error getting providers: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/collections")
async def get_collections(
    provider: VectorDBProvider = Query(default=VectorDBProvider.MILVUS)
):
    """获取指定向量数据库中的集合"""
    try:
        vector_store_service = VectorStoreService()
        collections = vector_store_service.list_collections(provider.value)
        return {"collections": collections}
    except Exception as e:
        logger.error(f"Error getting collections: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.post("/search")
async def search(
    data: dict = Body(...)
):
    """执行向量搜索"""
    try:
        # 从请求体中获取参数
        query = data.get("query")
        collection_name = data.get("collection_name")
        provider = data.get("provider", "milvus")
        top_k = data.get("top_k", 3)
        threshold = data.get("threshold", 0.7)
        
        if not query or not collection_name:
            raise HTTPException(status_code=400, detail="Missing required parameters: query and collection_name")
            
        logger.info(f"Search request - Query: {query}, Collection: {collection_name}, Provider: {provider}, Top K: {top_k}, Threshold: {threshold}")
        
        # 创建向量数据库配置
        vector_db_config = VectorDBConfig(
            provider=provider,
            index_mode="flat"  # 使用默认的 flat 索引模式进行搜索
        )
        
        # 执行搜索
        results = await search_service.search(
            query=query,
            collection_name=collection_name,
            provider=provider,
            config=vector_db_config,
            top_k=top_k,
            threshold=threshold
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Error performing search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections/{provider}")
async def get_provider_collections(provider: str):
    """Get collections for a specific vector database provider"""
    try:
        vector_store_service = VectorStoreService()
        collections = vector_store_service.list_collections(provider)
        return {"collections": collections}
    except Exception as e:
        logger.error(f"Error getting collections for provider {provider}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/collections/{provider}/{collection_name}")
async def get_collection_info(provider: str, collection_name: str):
    """Get detailed information about a specific collection"""
    try:
        vector_store_service = VectorStoreService()
        info = vector_store_service.get_collection_info(provider, collection_name)
        return info
    except Exception as e:
        logger.error(f"Error getting collection info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.delete("/collections/{provider}/{collection_name}")
async def delete_collection(provider: str, collection_name: str):
    """Delete a specific collection"""
    try:
        vector_store_service = VectorStoreService()
        success = vector_store_service.delete_collection(provider, collection_name)
        if success:
            return {"message": f"Collection {collection_name} deleted successfully"}
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to delete collection {collection_name}"
            )
    except Exception as e:
        logger.error(f"Error deleting collection: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/documents")
async def get_documents(type: str = Query("all")):
    try:
        documents = []
        logger.info(f"开始获取文档列表,类型: {type}")
        
        # 读取loaded文档
        if type in ["all", "loaded"]:
            loaded_dir = "01-loaded-docs"
            logger.info(f"读取loaded文档目录: {loaded_dir}")
            if os.path.exists(loaded_dir):
                for filename in os.listdir(loaded_dir):
                    if filename.endswith('.json'):
                        file_path = os.path.join(loaded_dir, filename)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                doc_data = json.load(f)
                                documents.append({
                                    "id": filename,
                                    "name": filename,
                                    "type": "loaded",
                                    "metadata": {
                                        "total_pages": doc_data.get("total_pages"),
                                        "total_chunks": doc_data.get("total_chunks"),
                                        "loading_method": doc_data.get("loading_method"),
                                        "chunking_method": doc_data.get("chunking_method"),
                                        "timestamp": doc_data.get("timestamp")
                                    }
                                })
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON解析错误,文件 {filename}: {str(e)}")
                            continue
                        except Exception as e:
                            logger.error(f"读取loaded文件 {filename} 时出错: {str(e)}")
                            continue

        # 读取chunked文档(包括parsed文件)
        if type in ["all", "chunked", "parsed"]:
            chunked_dir = "01-chunked-docs"
            logger.info(f"读取chunked文档目录: {chunked_dir}")
            if os.path.exists(chunked_dir):
                for filename in os.listdir(chunked_dir):
                    if filename.endswith('.json'):
                        file_path = os.path.join(chunked_dir, filename)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                doc_data = json.load(f)
                                # 根据文件名判断文档类型
                                doc_type = "parsed" if filename.startswith("parsed_") else "chunked"
                                documents.append({
                                    "id": filename,
                                    "name": filename,
                                    "type": doc_type,
                                    "metadata": {
                                        "total_pages": doc_data.get("total_pages"),
                                        "total_chunks": doc_data.get("total_chunks"),
                                        "chunking_method": doc_data.get("chunking_method"),
                                        "timestamp": doc_data.get("timestamp")
                                    }
                                })
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON解析错误,文件 {filename}: {str(e)}")
                            continue
                        except Exception as e:
                            logger.error(f"读取chunked文件 {filename} 时出错: {str(e)}")
                            continue
            else:
                logger.warning(f"目录不存在: {chunked_dir}")
        
        # 按时间戳排序,最新的在前面
        def get_timestamp(doc):
            try:
                timestamp = doc.get("metadata", {}).get("timestamp", "")
                return timestamp if timestamp else "1970-01-01T00:00:00"
            except:
                return "1970-01-01T00:00:00"
                
        documents.sort(key=get_timestamp, reverse=True)
        
        logger.info(f"成功获取文档列表,共 {len(documents)} 个文档")
        return {"documents": documents}
    except Exception as e:
        logger.error(f"获取文档列表时出错: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{doc_name}")
async def get_document(doc_name: str, type: str = Query("loaded")):
    try:

        base_name = doc_name.replace('.json', '')
        file_name = f"{base_name}.json"
        
        # 根据类型选择不同的目录
        directory = "01-loaded-docs" if type == "loaded" else "01-chunked-docs"
        file_path = os.path.join(directory, file_name)
        
        logger.info(f"Attempting to read document from: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"Document not found at path: {file_path}")
            raise HTTPException(status_code=404, detail="Document not found")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            doc_data = json.load(f)
            
        return doc_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{doc_name}")
async def delete_document(doc_name: str, type: str = Query("loaded")):
    try:
        # 移除已有的 .json 扩展名（如果有）然后添加一个
        base_name = doc_name.replace('.json', '')
        file_name = f"{base_name}.json"
        
        # 根据类型选择不同的目录
        directory = "01-loaded-docs" if type == "loaded" else "01-chunked-docs"
        file_path = os.path.join(directory, file_name)
        
        logger.info(f"Attempting to delete document: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"Document not found at path: {file_path}")
            raise HTTPException(status_code=404, detail="Document not found")
            
        # 删除文件
        os.remove(file_path)
        
        return {
            "status": "success",
            "message": f"Document {doc_name} deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/embedded-docs/{doc_name}")
async def get_embedded_doc(doc_name: str):
    """Get specific embedded document"""
    try:
        logger.info(f"Attempting to read document: {doc_name}")
        file_path = os.path.join("02-embedded-docs", doc_name)
        
        if not os.path.exists(file_path):
            logger.error(f"Document not found: {file_path}")
            raise HTTPException(
                status_code=404,
                detail=f"Document {doc_name} not found"
            )
            
        with open(file_path, 'r', encoding='utf-8') as f:
            doc_data = json.load(f)
            logger.info(f"Successfully read document: {doc_name}")
            
            return {
                "embeddings": [
                    {
                        "embedding": embedding["embedding"],
                        "metadata": {
                            "document_name": doc_data.get("document_name", doc_name),
                            "chunk_id": idx + 1,
                            "total_chunks": len(doc_data["embeddings"]),
                            "content": embedding["metadata"].get("content", ""),
                            "page_number": embedding["metadata"].get("page_number", ""),
                            "page_range": embedding["metadata"].get("page_range", ""),
                            # "chunking_method": embedding["metadata"].get("chunking_method", ""),
                            "embedding_model": doc_data.get("embedding_model", ""),
                            "embedding_provider": doc_data.get("embedding_provider", ""),
                            "embedding_timestamp": doc_data.get("created_at", ""),
                            "vector_dimension": doc_data.get("vector_dimension", 0)
                        }
                    }
                    for idx, embedding in enumerate(doc_data["embeddings"])
                ]
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting embedded document {doc_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/embedded-docs/{doc_name}")
async def delete_embedded_doc(doc_name: str):
    """Delete specific embedded document"""
    try:
        file_path = os.path.join("02-embedded-docs", doc_name)
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404,
                detail=f"Document {doc_name} not found"
            )
            
        os.remove(file_path)
        return {"message": f"Document {doc_name} deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting embedded document {doc_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/parse")
async def parse_file(
    file: UploadFile = File(...),
    loading_method: str = Form(...),
    parsing_option: str = Form(...),
    file_type: str = Form(...)
):
    try:
        # Save uploaded file
        temp_path = os.path.join("temp", file.filename)
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Prepare metadata
        metadata = {
            "filename": file.filename,
            "loading_method": loading_method,
            "original_file_size": len(content),
            "processing_date": datetime.now().isoformat(),
            "parsing_method": parsing_option,
            "file_type": file_type
        }
        
        loading_service = LoadingService()
        raw_text = loading_service.load_document(temp_path, loading_method)
        metadata["total_pages"] = loading_service.get_total_pages()
        
        page_map = loading_service.get_page_map()
        
        parsing_service = ParsingService()
        parsed_content = parsing_service.parse_document(
            raw_text, 
            parsing_option, 
            metadata,
            page_map=page_map
        )
        
        # 保存解析结果到 01-chunked-docs 目录，添加 parsed_ 前缀
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        base_name = os.path.splitext(file.filename)[0]
        doc_name = f"parsed_{base_name}_{timestamp}"
        
        # 创建保存目录
        os.makedirs("01-chunked-docs", exist_ok=True)
        
        # 保存文件
        filepath = os.path.join("01-chunked-docs", f"{doc_name}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(parsed_content, f, ensure_ascii=False, indent=2)
        
        # Clean up temp file
        os.remove(temp_path)
        
        return {
            "parsed_content": parsed_content,
            "saved_filepath": filepath
        }
    except Exception as e:
        logger.error(f"Error parsing file: {str(e)}")
        raise

@app.post("/load")
async def load_file(
    file: UploadFile = File(...),
    loading_method: str = Form(...),
    strategy: str = Form(None),
    chunking_strategy: str = Form(None),
    chunking_options: str = Form(None)
):
    try:
        # 保存上传的文件
        temp_path = os.path.join("temp", file.filename)
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # 准备元数据
        metadata = {
            "filename": file.filename,
            "total_chunks": 0,  # 将在后面更新
            "total_pages": 0,   # 将在后面更新
            "loading_method": loading_method,
            "loading_strategy": strategy,  
            "chunking_strategy": chunking_strategy, 
            "timestamp": datetime.now().isoformat()
        }
        
        # Parse chunking options if provided
        chunking_options_dict = None
        if chunking_options:
            chunking_options_dict = json.loads(chunking_options)
        
        # 使用 LoadingService 加载文档
        loading_service = LoadingService()
        raw_text = loading_service.load_pdf(
            temp_path, 
            loading_method, 
            strategy=strategy,
            chunking_strategy=chunking_strategy,
            chunking_options=chunking_options_dict
        )
        
        metadata["total_pages"] = loading_service.get_total_pages()
        
        page_map = loading_service.get_page_map()
        
        # 转换成标准化的chunks格式
        chunks = []
        for idx, page in enumerate(page_map, 1):
            chunk_metadata = {
                "chunk_id": idx,
                "page_number": page["page"],
                "page_range": str(page["page"]),
                "word_count": len(page["text"].split())
            }
            if "metadata" in page:
                chunk_metadata.update(page["metadata"])
            
            chunks.append({
                "content": page["text"],
                "metadata": chunk_metadata
            })
        
        # 使用 LoadingService 保存文档，传递strategy参数
        filepath = loading_service.save_document(
            filename=file.filename,
            chunks=chunks,
            metadata=metadata,
            loading_method=loading_method,
            strategy=strategy,
            chunking_strategy=chunking_strategy,
        )
        
        # 读取保存的文档以返回
        with open(filepath, "r", encoding="utf-8") as f:
            document_data = json.load(f)
        
        # 清理临时文件
        os.remove(temp_path)
        
        return {"loaded_content": document_data, "filepath": filepath}
    except Exception as e:
        logger.error(f"Error loading file: {str(e)}")
        raise

@app.post("/chunk")
async def chunk_document(data: dict = Body(...)):
    try:
        doc_id = data.get("doc_id")
        chunking_option = data.get("chunking_option")
        chunk_size = data.get("chunk_size", 1000)
        doc_type = data.get("doc_type", "loaded")  # 添加文档类型参数
        
        if not doc_id or not chunking_option:
            raise HTTPException(
                status_code=400, 
                detail="Missing required parameters: doc_id and chunking_option"
            )
        
        # 根据文档类型选择目录
        if doc_type == "loaded":
            directory = "01-loaded-docs"
        elif doc_type == "parsed":
            directory = "01-chunked-docs"  # parsed 文件保存在 chunked-docs 目录
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported document type: {doc_type}"
            )
            
        file_path = os.path.join(directory, doc_id)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Document not found at {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            doc_data = json.load(f)
            
        # 构建页面映射
        page_map = []
        if doc_type == "loaded":
            page_map = [
                {
                    'page': chunk['metadata']['page_number'],
                    'text': chunk['content']
                }
                for chunk in doc_data['chunks']
            ]
        else:  # parsed
            # 检查是否是Verilog网表文件
            is_verilog = False
            if doc_id.endswith('.v'):
                is_verilog = True
            elif 'content' in doc_data:
                content_str = str(doc_data['content'])
                if 'SYNOPSYS_UNCONNECTED' in content_str or 'module' in content_str or 'endmodule' in content_str:
                    is_verilog = True
            
            if is_verilog:
                # 对Verilog网表文件进行特殊处理
                content = doc_data.get('content', [])
                if isinstance(content, list):
                    # 将每个模块定义作为一个块
                    current_module = []
                    module_count = 0
                    for item in content:
                        if isinstance(item, dict):
                            text = item.get('content', '')
                            # 按分号分割文本
                            statements = text.split(';')
                            for statement in statements:
                                statement = statement.strip()
                                if statement:  # 忽略空语句
                                    # 检查是否是模块定义
                                    if 'module' in statement or 'endmodule' in statement:
                                        if current_module:
                                            page_map.append({
                                                'page': module_count + 1,
                                                'text': '\n'.join(current_module)
                                            })
                                            module_count += 1
                                            current_module = []
                                        current_module.append(statement)
                                    else:
                                        current_module.append(statement)
                    if current_module:
                        page_map.append({
                            'page': module_count + 1,
                            'text': '\n'.join(current_module)
                        })
                else:
                    # 如果不是列表,尝试按行分割
                    text = str(content)
                    lines = text.split('\n')
                    current_chunk = []
                    chunk_count = 0
                    for line in lines:
                        # 按分号分割行
                        statements = line.split(';')
                        for statement in statements:
                            statement = statement.strip()
                            if statement:  # 忽略空语句
                                current_chunk.append(statement)
                                if len(current_chunk) >= chunk_size:
                                    page_map.append({
                                        'page': chunk_count + 1,
                                        'text': '\n'.join(current_chunk)
                                    })
                                    chunk_count += 1
                                    current_chunk = []
                    if current_chunk:
                        page_map.append({
                            'page': chunk_count + 1,
                            'text': '\n'.join(current_chunk)
                        })
            else:
                # 非Verilog文件的常规处理
                page_map = [
                    {
                        'page': item['page'],
                        'text': item['content']
                    }
                    for item in doc_data['content']
                ]
            
        # 准备元数据
        metadata = {
            "filename": doc_data.get('filename', doc_id),
            "loading_method": doc_data.get('loading_method', 'parsed'),
            "total_pages": doc_data.get('total_pages', len(page_map))
        }
            
        chunking_service = ChunkingService()
        result = chunking_service.chunk_text(
            text="",  # 不需要传递文本，因为我们使用 page_map
            method=chunking_option,
            metadata=metadata,
            page_map=page_map,
            chunk_size=chunk_size
        )
        
        # 生成输出文件名
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        # 从原始文件名中提取基础名称
        base_name = os.path.splitext(doc_id)[0]
        # 如果文件名以parsed_开头,去掉这个前缀
        if base_name.startswith('parsed_'):
            base_name = base_name[7:]
        # 构建新的文件名
        output_filename = f"{base_name}_{chunking_option}_{timestamp}.json"
        
        output_path = os.path.join("01-chunked-docs", output_filename)
        os.makedirs("01-chunked-docs", exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return result
        
    except Exception as e:
        logger.error(f"Error chunking document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate")
async def evaluate_search(
    file: UploadFile = File(...),
    collection_id: str = Form(...),
    top_k: int = Form(10),
    threshold: float = Form(0.7)
):
    try:
        # 读取CSV文件
        df = pd.read_csv(file.file)
        
        # 只合并前四列的文本内容
        df['combined_text'] = df.apply(
            lambda row: ' '.join(
                str(val) for i, val in enumerate(row) 
                if i < 4 and pd.notna(val) and val != '[]'
            ), 
            axis=1
        )
        
        # 初始化SearchService
        search_service = SearchService()
        
        results = []
        total_score_hit = 0
        total_score_find = 0
        valid_queries = 0
        
        # 处理每个查询
        for _, row in df.iterrows():
            # 跳过没有标签的行
            if pd.isna(row['LABEL']) or row['LABEL'] == '[]':
                continue
                
            try:
                # 解析标签页码列表
                label_str = str(row['LABEL']).strip('[]').replace(' ', '')
                if label_str:
                    expected_pages = [int(x.strip()) for x in label_str.split(',') if x.strip()]
                else:
                    continue
                
                # 执行搜索
                search_results = await search_service.search(
                    query=row['combined_text'],
                    collection_id=collection_id,
                    top_k=top_k,
                    threshold=threshold
                )
                
                # 提取找到的页码
                found_pages = [int(result['metadata']['page']) for result in search_results]
                
                # 计算分数
                hits = sum(1 for page in found_pages if page in expected_pages)
                score_hit = hits / len(found_pages) if found_pages else 0
                score_find = len(set(found_pages) & set(expected_pages)) / len(expected_pages)
                
                # 添加到结果列表，包括所有top_k结果的文本
                result_entry = {
                    "query": row['combined_text'],
                    "expected_pages": expected_pages,
                    "found_pages": found_pages,
                    "score_hit": score_hit,
                    "score_find": score_find
                }
                
                # 添加每个top_k结果的文本作为单独的字段
                for i, result in enumerate(search_results, 1):
                    result_entry[f"text_{i}"] = result['text']
                    result_entry[f"page_{i}"] = result['metadata']['page']
                    result_entry[f"score_{i}"] = result['score']
                
                results.append(result_entry)
                
                total_score_hit += score_hit
                total_score_find += score_find
                valid_queries += 1
                
            except Exception as e:
                logger.warning(f"Error processing row: {str(e)}")
                continue
        
        if valid_queries == 0:
            raise ValueError("No valid queries found in the CSV file")
        
        # 计算平均分数
        average_scores = {
            "score_hit": total_score_hit / valid_queries,
            "score_find": total_score_find / valid_queries
        }
        
        # 保存结果
        output_dir = Path("06-evaluation-result")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存详细的JSON结果
        output_path = output_dir / f"evaluation_results_{timestamp}.json"
        evaluation_results = {
            "results": results,
            "average_scores": average_scores,
            "total_queries": valid_queries,
            "parameters": {
                "collection_id": collection_id,
                "top_k": top_k,
                "threshold": threshold
            }
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(evaluation_results, f, indent=2)
            
        # 保存CSV格式的结果，每个top_k结果单独一列
        results_df = pd.DataFrame(results)
        
        # 重新排列列的顺序，使其更有逻辑性
        column_order = ['query', 'expected_pages', 'found_pages', 'score_hit', 'score_find']
        for i in range(1, top_k + 1):
            column_order.extend([f'page_{i}', f'score_{i}', f'text_{i}'])
        
        # 只选择存在的列
        existing_columns = [col for col in column_order if col in results_df.columns]
        results_df = results_df[existing_columns]
        
        csv_path = output_dir / f"evaluation_results_{timestamp}.csv"
        results_df.to_csv(csv_path, index=False)
        
        return evaluation_results
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/save-search")
async def save_search_results(request: Request):
    try:
        data = await request.json()
        query = data.get("query")
        collection_id = data.get("collection_id")
        results = data.get("results")
        
        if not all([query, collection_id, results]):
            raise HTTPException(status_code=400, detail="Missing required parameters")
        
        # 直接创建 SearchService 实例
        search_service = SearchService()
        filepath = search_service.save_search_results(query, collection_id, results)
        return {"saved_filepath": filepath}
        
    except Exception as e:
        logger.error(f"Error saving search results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/generation/models")
async def get_generation_models():
    """获取可用的生成模型列表"""
    try:
        generation_service = GenerationService()
        models = generation_service.get_available_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"Error getting generation models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def call_ollama_api(model_name: str, prompt: str) -> str:
    """调用Ollama API"""
    try:
        logger.info(f"调用Ollama API，模型: {model_name}")
        logger.info(f"提示词长度: {len(prompt)} 字符")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{OLLAMA_API_BASE}/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False
                }
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Ollama API错误: {error_text}")
                    raise Exception(f"Ollama API错误: {error_text}")
                
                result = await response.json()
                logger.info("Ollama API调用成功")
                return result.get("response", "")
                
    except Exception as e:
        logger.error(f"调用Ollama API失败: {str(e)}")
        raise

async def call_openai_api(model_name: str, prompt: str, api_key: str) -> str:
    """调用OpenAI API"""
    try:
        logger.info(f"调用OpenAI API，模型: {model_name}")
        logger.info(f"提示词长度: {len(prompt)} 字符")
        
        if not api_key:
            raise Exception("未提供OpenAI API密钥")
            
        client = AsyncOpenAI(api_key=api_key)
        
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "你是一个专业的助手，请基于提供的上下文回答问题。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        logger.info("OpenAI API调用成功")
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"调用OpenAI API失败: {str(e)}")
        raise

def split_content(content: str, max_chunk_size: int = MAX_CHUNK_SIZE) -> List[str]:
    """将内容分割成多个块"""
    chunks = []
    current_chunk = ""
    current_size = 0
    
    # 按行分割内容
    lines = content.split('\n')
    
    for line in lines:
        line_size = len(line.encode('utf-8'))
        
        # 如果单行超过最大块大小，需要进一步分割
        if line_size > max_chunk_size:
            # 如果当前块不为空，先保存
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
                current_size = 0
            
            # 分割大行
            words = line.split()
            current_line = ""
            for word in words:
                word_size = len(word.encode('utf-8'))
                if current_size + word_size > max_chunk_size:
                    chunks.append(current_line)
                    current_line = word
                    current_size = word_size
                else:
                    current_line += " " + word if current_line else word
                    current_size += word_size
            if current_line:
                current_chunk = current_line
        # 如果添加当前行会超出块大小限制
        elif current_size + line_size > max_chunk_size:
            chunks.append(current_chunk)
            current_chunk = line
            current_size = line_size
        else:
            current_chunk += "\n" + line if current_chunk else line
            current_size += line_size
    
    # 添加最后一个块
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

@app.post("/generate")
async def generate(request: GenerationRequest = Body(..., max_size=MAX_CONTEXT_SIZE)):
    try:
        logger.info(f"收到生成请求: {request.dict()}")
        
        # 检查文件数量
        if len(request.context_file_ids) > MAX_FILES:
            raise HTTPException(status_code=400, detail=f"文件数量超过限制: 最多{MAX_FILES}个文件")
        
        # 读取上下文文件内容
        context_contents = []
        total_size = 0
        total_chunks = 0
        
        if request.context_contents:  # 如果提供了context_contents，直接使用
            for content in request.context_contents:
                if content is None:
                    continue
                    
                # 检查内容大小
                content_size = len(content.encode('utf-8'))
                if content_size > MAX_FILE_SIZE:
                    # 如果内容超过单个文件限制，进行分块
                    chunks = split_content(content)
                    if len(chunks) > MAX_CHUNKS:
                        raise HTTPException(status_code=400, detail=f"文件内容块数超过限制: 最多{MAX_CHUNKS}个块")
                    
                    total_chunks += len(chunks)
                    if total_chunks > MAX_CHUNKS:
                        raise HTTPException(status_code=400, detail=f"总块数超过限制: 最多{MAX_CHUNKS}个块")
                    
                    # 添加分块内容
                    for i, chunk in enumerate(chunks):
                        context_contents.append(f"=== 文件块 {i+1}/{len(chunks)} ===\n{chunk}\n=== 块结束 ===\n")
                else:
                    total_size += content_size
                    if total_size > MAX_CONTEXT_SIZE:
                        raise HTTPException(status_code=400, detail=f"总内容大小超过限制: 最多{MAX_CONTEXT_SIZE/1024/1024}MB")
                    context_contents.append(content)
        else:  # 否则从文件读取
            for file_id in request.context_file_ids:
                if file_id in file_storage:
                    file_info = file_storage[file_id]
                    try:
                        # 检查文件大小
                        file_size = Path(file_info["path"]).stat().st_size
                        if file_size > MAX_FILE_SIZE:
                            # 如果文件超过大小限制，进行分块读取
                            chunks = []
                            with open(file_info["path"], "r", encoding="utf-8") as f:
                                content = f.read()
                                chunks = split_content(content)
                            
                            if len(chunks) > MAX_CHUNKS:
                                raise HTTPException(status_code=400, detail=f"文件内容块数超过限制: 最多{MAX_CHUNKS}个块")
                            
                            total_chunks += len(chunks)
                            if total_chunks > MAX_CHUNKS:
                                raise HTTPException(status_code=400, detail=f"总块数超过限制: 最多{MAX_CHUNKS}个块")
                            
                            # 添加分块内容
                            for i, chunk in enumerate(chunks):
                                context_contents.append(f"=== 文件: {file_info['name']} 块 {i+1}/{len(chunks)} ===\n{chunk}\n=== 块结束 ===\n")
                        else:
                            # 正常读取文件内容
                            with open(file_info["path"], "r", encoding="utf-8") as f:
                                content = f.read()
                                content_size = len(content.encode('utf-8'))
                                total_size += content_size
                                if total_size > MAX_CONTEXT_SIZE:
                                    raise HTTPException(status_code=400, detail=f"总内容大小超过限制: 最多{MAX_CONTEXT_SIZE/1024/1024}MB")
                                
                                context_contents.append(f"=== 文件: {file_info['name']} ===\n{content}\n=== 文件结束 ===\n")
                        
                        # 更新使用次数
                        file_storage[file_id]["used_count"] += 1
                    except Exception as e:
                        logger.error(f"读取文件 {file_id} 失败: {str(e)}")
                        raise HTTPException(status_code=500, detail=f"读取文件 {file_info['name']} 失败: {str(e)}")
                else:
                    logger.warning(f"文件 {file_id} 不存在")
                    raise HTTPException(status_code=404, detail=f"文件 {file_id} 不存在")
        
        # 构建完整的上下文
        full_context = "\n".join(context_contents)
        
        # 构建完整的提示词，包含上下文和问题
        full_prompt = f"""你是一个专业的助手。请基于以下文件内容回答问题。如果文件内容中没有相关信息，请说明无法回答。
注意：如果文件内容被分成了多个块，请确保查看所有相关块的内容。

文件内容：
{full_context}

问题：
{request.query}

请提供详细的回答，并说明你的推理过程。如果回答涉及多个文件或文件块的内容，请明确指出信息来源。"""

        # 调用LLM生成回答
        try:
            # 根据不同的provider调用相应的API
            if request.provider == "ollama":
                # 调用Ollama API
                response = await call_ollama_api(request.model_name, full_prompt)
            elif request.provider == "openai":
                # 调用OpenAI API
                response = await call_openai_api(request.model_name, full_prompt, request.api_key)
            else:
                raise HTTPException(status_code=400, detail=f"不支持的provider: {request.provider}")
            
            return {
                "data": {
                    "response": response,
                    "context": full_context,  # 返回使用的上下文内容
                    "prompt": full_prompt,    # 返回完整的提示词
                    "used_files": [           # 返回使用的文件信息
                        {
                            "file_id": file_id,
                            "name": file_storage[file_id]["name"],
                            "size": file_storage[file_id]["size"],
                            "used_count": file_storage[file_id]["used_count"]
                        }
                        for file_id in request.context_file_ids
                    ]
                }
            }
        except Exception as e:
            logger.error(f"调用LLM API失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"调用LLM API失败: {str(e)}")
            
    except HTTPException as he:
        logger.error(f"生成错误 (HTTP): {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"生成错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search-results")
async def list_search_results():
    """获取所有搜索结果文件列表"""
    try:
        search_results_dir = "04-search-results"
        if not os.path.exists(search_results_dir):
            return {"files": []}
            
        files = []
        for filename in os.listdir(search_results_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(search_results_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    files.append({
                        "id": filename,
                        "name": f"Search: {data.get('query', 'Unknown')} ({filename})",
                        "timestamp": data.get('timestamp', '')
                    })
                    
        # 按时间戳排序，最新的在前面
        files.sort(key=lambda x: x['timestamp'], reverse=True)
        return {"files": files}
        
    except Exception as e:
        logger.error(f"Error listing search results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search-results/{file_id}")
async def get_search_result(file_id: str):
    """获取特定搜索结果文件的内容"""
    try:
        file_path = os.path.join("04-search-results", file_id)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Search result file not found")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data
            
    except Exception as e:
        logger.error(f"Error reading search result file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/indexing")
async def list_embedding_files():
    """列出所有可用的嵌入文件"""
    try:
        embedding_dir = os.path.join("02-embedded-docs")
        logger.info(f"正在扫描目录: {embedding_dir}")
        logger.info(f"当前工作目录: {os.getcwd()}")
        
        if not os.path.exists(embedding_dir):
            logger.warning(f"目录不存在: {embedding_dir}")
            return {
                "message": "嵌入文件目录不存在",
                "embedding_files": []
            }
            
        embedding_files = []
        for file in os.listdir(embedding_dir):
            if file.endswith('.json'):
                file_path = os.path.join(embedding_dir, file)
                logger.info(f"正在读取文件: {file_path}")
                # 读取文件以获取更多信息
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        embedding_files.append({
                            "filename": file,
                            "filepath": file_path,
                            "document_name": data.get("filename", ""),
                            "embedding_provider": data.get("embedding_provider", "unknown"),
                            "embedding_model": data.get("embedding_model", "unknown"),
                            "vector_dimension": data.get("vector_dimension", 0),
                            "total_vectors": len(data.get("embeddings", [])),
                            "created_at": data.get("created_at", "")
                        })
                        logger.info(f"成功添加文件信息: {file}")
                    except json.JSONDecodeError:
                        logger.warning(f"无法解析文件 {file}")
                        continue
        
        if not embedding_files:
            logger.warning("没有找到嵌入文件")
            return {
                "message": "没有找到嵌入文件",
                "embedding_files": []
            }
            
        # 按创建时间排序
        embedding_files.sort(key=lambda x: x["created_at"], reverse=True)
        logger.info(f"找到 {len(embedding_files)} 个嵌入文件")
        
        return {
            "message": f"找到 {len(embedding_files)} 个嵌入文件",
            "embedding_files": embedding_files
        }
        
    except Exception as e:
        logger.error(f"列出嵌入文件时出错: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def get_available_collections():
    """获取可用的集合列表"""
    vector_store_service = VectorStoreService()
    try:
        collections = vector_store_service.list_collections(provider=VectorDBProvider.MILVUS.value)
        logger.info(f"找到以下集合：{collections}")
        return collections
    except Exception as e:
        logger.error(f"获取集合列表时出错: {e}")
        return []

# 在应用启动时使用这个函数
collections = get_available_collections()
print(f"可用的集合：{collections}")

@app.post("/upload-context")
async def upload_context(file: UploadFile = File(...)):
    """上传文件接口"""
    try:
        logger.info(f"开始接收文件: {file.filename}, 大小: {file.size} bytes")
        logger.info(f"文件类型: {file.content_type}")
        logger.info(f"文件头信息: {file.headers}")
        
        # 检查文件大小
        if file.size > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=400, detail="文件大小超过限制")
        
        # 使用原始文件名
        file_name = file.filename
        file_path = UPLOAD_DIR / file_name
        
        logger.info(f"上传目录: {UPLOAD_DIR}")
        logger.info(f"上传目录是否存在: {UPLOAD_DIR.exists()}")
        logger.info(f"上传目录是否为目录: {UPLOAD_DIR.is_dir()}")
        logger.info(f"上传目录权限: {oct(UPLOAD_DIR.stat().st_mode)}")
        logger.info(f"目标文件路径: {file_path}")
        
        # 检查文件是否已存在
        if file_path.exists():
            logger.info(f"文件已存在: {file_path}")
            # 查找已存在的文件ID
            existing_file_id = None
            for fid, info in file_storage.items():
                if info["name"] == file_name:
                    existing_file_id = fid
                    break
            
            if existing_file_id:
                logger.info(f"返回已存在的文件ID: {existing_file_id}")
                return {"fileId": existing_file_id, "status": "exists"}
        
        # 生成文件ID（使用原始文件名）
        file_id = file_name
        
        logger.info(f"保存文件到: {file_path}")
        
        try:
            # 确保目录存在
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存文件
            logger.info("开始写入文件...")
            with file_path.open("wb") as buffer:
                # 分块读取和写入
                total_size = 0
                while True:
                    try:
                        # 设置超时
                        chunk = await asyncio.wait_for(file.read(CHUNK_SIZE), timeout=UPLOAD_TIMEOUT)
                        if not chunk:
                            break
                        buffer.write(chunk)
                        total_size += len(chunk)
                        logger.info(f"已写入: {total_size} bytes")
                    except asyncio.TimeoutError:
                        raise HTTPException(status_code=408, detail="上传超时")
            
            logger.info("文件写入完成")
                
            # 验证文件是否成功保存
            if not file_path.exists():
                raise Exception("文件保存失败")
                
            # 验证文件大小
            actual_size = file_path.stat().st_size
            if actual_size != file.size:
                raise Exception(f"文件大小不匹配: 预期 {file.size} bytes, 实际 {actual_size} bytes")
                
            # 验证文件内容
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    logger.info(f"文件内容长度: {len(content)} 字符")
                    logger.info(f"文件内容前100个字符: {content[:100]}")
            except UnicodeDecodeError:
                logger.warning("文件不是UTF-8编码，尝试使用latin-1编码")
                with open(file_path, "r", encoding="latin-1") as f:
                    content = f.read()
                    logger.info(f"使用latin-1编码读取成功，内容长度: {len(content)} 字符")
                
        except Exception as e:
            logger.error(f"保存文件失败: {e}")
            # 清理可能的部分文件
            if file_path.exists():
                file_path.unlink()
            raise HTTPException(status_code=500, detail=f"保存文件失败: {str(e)}")
        
        # 存储文件信息
        file_storage[file_id] = {
            "path": str(file_path),
            "name": file_name,
            "size": file.size,
            "upload_time": datetime.now(),
            "status": "active",
            "used_count": 0
        }
        
        # 保存file_storage到文件
        save_file_storage()
        
        logger.info(f"文件上传成功: {file_id}")
        logger.info(f"文件存储信息: {file_storage[file_id]}")
        
        return {"fileId": file_id, "status": "success"}
    except HTTPException as he:
        logger.error(f"上传错误 (HTTP): {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"上传错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files")
async def list_files():
    """获取所有文件列表"""
    try:
        logger.info("开始获取文件列表")
        logger.info(f"当前file_storage中的文件数量: {len(file_storage)}")
        logger.info(f"file_storage内容: {file_storage}")
        
        files = []
        for file_id, info in file_storage.items():
            try:
                # 验证文件是否存在
                file_path = Path(info["path"])
                logger.info(f"检查文件: {file_path}")
                logger.info(f"文件是否存在: {file_path.exists()}")
                
                if not file_path.exists():
                    logger.warning(f"文件不存在: {file_path}")
                    continue
                    
                # 验证文件大小
                actual_size = file_path.stat().st_size
                if actual_size != info["size"]:
                    logger.warning(f"文件大小不匹配: {file_path}, 预期: {info['size']}, 实际: {actual_size}")
                    info["size"] = actual_size
                
                files.append({
                    "file_id": file_id,
                    "name": info["name"],
                    "size": info["size"],
                    "upload_time": info["upload_time"],
                    "status": info.get("status", "active"),
                    "used_count": info.get("used_count", 0)
                })
                logger.info(f"添加文件到列表: {info['name']}")
            except Exception as e:
                logger.error(f"处理文件 {file_id} 时出错: {str(e)}")
                continue
                
        logger.info(f"文件列表获取成功，共 {len(files)} 个文件")
        return {"files": files}
    except Exception as e:
        logger.error(f"获取文件列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files/{file_id}")
async def get_file_content(file_id: str):
    """获取文件内容"""
    try:
        logger.info(f"开始获取文件内容: {file_id}")
        
        if file_id not in file_storage:
            logger.error(f"文件不存在: {file_id}")
            raise HTTPException(status_code=404, detail="文件不存在")
            
        file_info = file_storage[file_id]
        logger.info(f"找到文件信息: {file_info}")
        
        file_path = Path(file_info["path"])
        logger.info(f"文件路径: {file_path}")
        logger.info(f"文件路径是否存在: {file_path.exists()}")
        logger.info(f"文件路径是否为文件: {file_path.is_file()}")
        logger.info(f"文件路径的父目录: {file_path.parent}")
        logger.info(f"文件路径的父目录是否存在: {file_path.parent.exists()}")
        logger.info(f"文件路径的父目录是否为目录: {file_path.parent.is_dir()}")
        logger.info(f"文件权限: {oct(file_path.stat().st_mode)}")
        
        if not file_path.exists():
            logger.error(f"文件路径不存在: {file_path}")
            raise HTTPException(status_code=404, detail=f"文件路径不存在: {file_path}")
            
        try:
            # 检查文件大小
            file_size = file_path.stat().st_size
            logger.info(f"文件大小: {file_size} bytes")
            
            if file_size == 0:
                logger.warning(f"文件为空: {file_path}")
                return {
                    "file_id": file_id,
                    "name": file_info["name"],
                    "content": "",
                    "size": 0,
                    "status": "empty"
                }
            
            # 尝试不同的编码方式读取文件
            encodings = ['utf-8', 'latin-1', 'gbk', 'gb2312', 'utf-16']
            content = None
            used_encoding = None
            
            for encoding in encodings:
                try:
                    logger.info(f"尝试使用 {encoding} 编码读取文件")
                    with open(file_path, "r", encoding=encoding) as f:
                        content = f.read()
                        content_size = len(content.encode('utf-8'))
                        logger.info(f"使用 {encoding} 编码读取成功，内容大小: {content_size} bytes")
                        logger.info(f"内容前100个字符: {content[:100]}")
                        used_encoding = encoding
                        break
                except UnicodeDecodeError:
                    logger.warning(f"使用 {encoding} 编码读取失败")
                    continue
                except Exception as e:
                    logger.error(f"使用 {encoding} 编码读取时发生错误: {str(e)}")
                    continue
            
            if content is None:
                logger.error("所有编码方式都读取失败")
                raise HTTPException(status_code=400, detail="无法读取文件内容，请检查文件编码")
            
            if not content:
                logger.warning(f"文件内容为空: {file_path}")
                return {
                    "file_id": file_id,
                    "name": file_info["name"],
                    "content": "",
                    "size": 0,
                    "status": "empty"
                }
            
            return {
                "file_id": file_id,
                "name": file_info["name"],
                "content": content,
                "size": len(content.encode('utf-8')),
                "status": "success",
                "encoding": used_encoding
            }
            
        except Exception as e:
            logger.error(f"读取文件失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"读取文件失败: {str(e)}")
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"获取文件内容失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/files/{file_id}")
async def delete_file(file_id: str):
    """删除文件"""
    try:
        logger.info(f"尝试删除文件: {file_id}")
        
        if file_id not in file_storage:
            logger.error(f"文件不存在: {file_id}")
            raise HTTPException(status_code=404, detail="File not found")
        
        # 获取文件信息
        file_info = file_storage[file_id]
        file_path = file_info["path"]
        
        logger.info(f"文件路径: {file_path}")
        
        # 删除文件
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"文件已删除: {file_path}")
            except Exception as e:
                logger.error(f"删除文件时出错: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")
        else:
            logger.warning(f"文件不存在: {file_path}")
        
        # 从存储中移除
        del file_storage[file_id]
        logger.info(f"文件信息已从存储中移除: {file_id}")
        
        return {"status": "success", "message": "File deleted"}
    except HTTPException as he:
        logger.error(f"删除文件错误 (HTTP): {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"删除文件错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 
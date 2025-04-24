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
from fastapi import FastAPI, Body, HTTPException
from utils.config import VectorDBProvider
from services.search_service import SearchService
import logging
from pymilvus import connections, utility
from services.document_processor_service import DocumentProcessor

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
vector_store_service = None
search_service = None

# 确保必要的目录存在
BASE_DIR = os.path.dirname(__file__)  # 指向 backend 目录
os.makedirs(os.path.join(BASE_DIR, "temp"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "01-chunked-docs"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "02-embedded-docs"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "01-original-docs"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "03-vector-store"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "04-search-results"), exist_ok=True)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    # allow_origins=["*"],
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """在应用启动时初始化服务"""
    global vector_store_service, search_service
    try:
        vector_store_service = VectorStoreService()
        search_service = SearchService()
        collections = vector_store_service.list_collections(provider=VectorDBProvider.MILVUS.value)
        logger.info(f"应用启动时找到以下集合：{collections}")
    except Exception as e:
        logger.error(f"应用启动时初始化服务出错: {e}")
        # 不要在这里抛出异常，让应用继续启动

@app.on_event("shutdown")
async def shutdown_event():
    """在应用关闭时清理资源"""
    global vector_store_service
    if vector_store_service:
        try:
            vector_store_service._disconnect_milvus()
            logger.info("应用关闭时成功断开 Milvus 连接")
        except Exception as e:
            logger.warning(f"应用关闭时断开连接出错: {e}")

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
    """为文档创建嵌入向量"""
    try:
        # 从请求中获取参数
        document_id = data.get("documentId")
        embedding_provider = data.get("embeddingProvider", "ollama")
        embedding_model = data.get("embeddingModel", "bge-m3:latest")
        
        if not document_id:
            raise ValueError("Missing documentId")
            
        # 构建文件路径 - 从 chunked-docs 目录读取
        chunked_file_path = os.path.join("01-chunked-docs", document_id)
        if not os.path.exists(chunked_file_path):
            raise FileNotFoundError(f"Document not found: {chunked_file_path}")
            
        # 读取分块后的文档
        with open(chunked_file_path, 'r', encoding='utf-8') as f:
            chunked_data = json.load(f)
            
        # 准备元数据
        metadata = {
            "document_name": os.path.splitext(document_id)[0],
            "embedding_provider": embedding_provider,
            "embedding_model": embedding_model,
            "created_at": datetime.now().isoformat()
        }
        
        # 如果分块数据中有元数据，则合并
        if "metadata" in chunked_data:
            metadata.update(chunked_data["metadata"])
        
        # 创建嵌入配置
        embedding_config = EmbeddingConfig(
            provider=embedding_provider,
            model_name=embedding_model
        )
        
        # 创建嵌入
        embedding_service = EmbeddingService()
        embeddings = embedding_service.create_embeddings(
            chunks=chunked_data["chunks"],
            metadata=metadata
        )
        
        # 保存嵌入向量
        doc_name = os.path.splitext(document_id)[0]  # 移除 .json 后缀
        filepath = embedding_service.save_embeddings(doc_name, embeddings)
        
        return {
            "status": "success",
            "message": "Embedding completed successfully",
            "filepath": filepath,
            "embeddings": embeddings["embeddings"],  # 只返回嵌入向量部分
            "documentId": document_id,
            "embeddingProvider": embedding_provider,
            "embeddingModel": embedding_model
        }
        
    except Exception as e:
        logger.error(f"Error creating embeddings: {str(e)}", exc_info=True)
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
        
        # 读取loaded文档
        if type in ["all", "loaded"]:
            loaded_dir = "01-loaded-docs"
            if os.path.exists(loaded_dir):
                for filename in os.listdir(loaded_dir):
                    if filename.endswith('.json'):
                        file_path = os.path.join(loaded_dir, filename)
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

        # 读取chunked文档
        if type in ["all", "chunked"]:
            chunked_dir = "01-chunked-docs"
            if os.path.exists(chunked_dir):
                for filename in os.listdir(chunked_dir):
                    if filename.endswith('.json'):
                        file_path = os.path.join(chunked_dir, filename)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            doc_data = json.load(f)
                            documents.append({
                                "id": filename,
                                "name": filename,  # 保持原始文件名
                                "type": "chunked"
                            })
        
        return {"documents": documents}
    except Exception as e:
        logger.error(f"Error getting documents: {str(e)}")
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
    parsing_option: str = Form(...)
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
        }
        
        loading_service = LoadingService()
        raw_text = loading_service.load_pdf(temp_path, loading_method)
        metadata["total_pages"] = loading_service.get_total_pages()
        
        page_map = loading_service.get_page_map()
        
        parsing_service = ParsingService()
        parsed_content = parsing_service.parse_pdf(
            raw_text, 
            parsing_option, 
            metadata,
            page_map=page_map
        )
        
        # Clean up temp file
        os.remove(temp_path)
        
        return {"parsed_content": parsed_content}
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
        
        if not doc_id or not chunking_option:
            raise HTTPException(
                status_code=400, 
                detail="Missing required parameters: doc_id and chunking_option"
            )
        
        # 读取已加载的文档
        file_path = os.path.join("01-loaded-docs", doc_id)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Document not found")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            doc_data = json.load(f)
            
        # 构建页面映射
        page_map = [
            {
                'page': chunk['metadata']['page_number'],
                'text': chunk['content']
            }
            for chunk in doc_data['chunks']
        ]
            
        # 准备元数据
        metadata = {
            "filename": doc_data['filename'],
            "loading_method": doc_data['loading_method'],
            "total_pages": doc_data['total_pages']
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
        base_name = doc_data['filename'].replace('.pdf', '').split('_')[0]
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

@app.post("/generate")
async def generate_response(
    query: str = Body(...),
    provider: str = Body(...),
    model_name: str = Body(...),
    search_results: List[Dict] = Body(...),
    api_key: Optional[str] = Body(None)
):
    """生成回答"""
    try:
        generation_service = GenerationService()
        result = generation_service.generate(
            provider=provider,
            model_name=model_name,
            query=query,
            search_results=search_results,
            api_key=api_key
        )
        return result
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
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
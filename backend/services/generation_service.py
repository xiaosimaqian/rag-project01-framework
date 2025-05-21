import os
import json
import logging
import requests
import openai
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from openai import OpenAI
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pymilvus import connections, Collection
from utils.config import MILVUS_CONFIG

logger = logging.getLogger(__name__)

class GenerationService:
    """
    生成服务类：负责调用不同的模型提供商（HuggingFace、OpenAI、DeepSeek、Ollama）生成回答
    支持本地模型和API调用，并将生成结果保存到文件
    """
    def __init__(self):
        """
        初始化生成服务，配置支持的模型列表和创建输出目录
        """
        self.models = {
            "huggingface": {
                "Llama-2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
                "DeepSeek-7b": "deepseek-ai/deepseek-llm-7b-chat",
                "DeepSeek-R1-Distill-Qwen": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
            },
            "openai": {
                "gpt-3.5-turbo": "gpt-3.5-turbo",
                "gpt-4": "gpt-4",
            },
            "deepseek": {
                "deepseek-v3": "deepseek-chat",
                "deepseek-r1": "deepseek-reasoner",
            },
            "ollama": {
                "DeepSeek-r1:14b": "DeepSeek-r1:14b",
            },
        }
        
        # 确保输出目录存在
        os.makedirs("05-generation-results", exist_ok=True)
        
    def _load_huggingface_model(self, model_name: str):
        """
        加载HuggingFace模型
        
        参数:
            model_name: 模型名称，对应self.models["huggingface"]中的键
            
        返回:
            model: 加载的模型
            tokenizer: 对应的分词器
        """
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.models["huggingface"][model_name],
                torch_dtype=torch.float16,
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(
                self.models["huggingface"][model_name]
            )
            return model, tokenizer
        except Exception as e:
            logger.error(f"Error loading HuggingFace model: {str(e)}")
            raise

    def _generate_with_huggingface(
        self,
        model_name: str,
        query: str,
        context: str,
        max_length: int = 512
    ) -> str:
        """
        使用HuggingFace模型生成回答
        
        参数:
            model_name: 模型名称
            query: 用户查询
            context: 上下文信息
            max_length: 生成文本的最大长度
            
        返回:
            生成的回答文本
        """
        try:
            model, tokenizer = self._load_huggingface_model(model_name)
            
            # 构建提示
            prompt = f"""请基于以下上下文回答问题。如果上下文中没有相关信息，请说明无法回答。

                        问题：{query}

                        上下文：
                        {context}

                        回答："""
        
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.split("回答：")[-1].strip()
            
        except Exception as e:
            logger.error(f"Error generating with HuggingFace: {str(e)}")
            raise

    def _generate_with_openai(
        self,
        model_name: str,
        query: str,
        context: str,
        api_key: Optional[str] = None
    ) -> str:
        """
        使用OpenAI API生成回答
        
        参数:
            model_name: 模型名称
            query: 用户查询
            context: 上下文信息
            api_key: OpenAI API密钥，如不提供则从环境变量获取
            
        返回:
            生成的回答文本
        """
        try:
            if not api_key:
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OpenAI API key not provided")
                    
            client = OpenAI(api_key=api_key)
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer the question."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
            ]
            
            response = client.chat.completions.create(
                model=self.models["openai"][model_name],
                messages=messages,
                temperature=0.7,
                max_tokens=512
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating with OpenAI: {str(e)}")
            raise

    def _generate_with_deepseek(
        self,
        model_name: str,
        query: str,
        context: str,
        api_key: Optional[str] = None,
        show_reasoning: bool = True
    ) -> str:
        """
        使用DeepSeek API生成回答
        
        参数:
            model_name: 模型名称
            query: 用户查询
            context: 上下文信息
            api_key: DeepSeek API密钥，如不提供则从环境变量获取
            show_reasoning: 是否显示推理过程（仅对推理模型有效）
            
        返回:
            生成的回答文本，对于推理模型可能包含思维过程
        """
        try:
            if not api_key:
                api_key = os.getenv("DEEPSEEK_API_KEY")
                if not api_key:
                    raise ValueError("DeepSeek API key not provided")
                    
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com"
            )
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer the question."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
            ]
            
            response = client.chat.completions.create(
                model=self.models["deepseek"][model_name],
                messages=messages,
                max_tokens=512,
                stream=False
            )
            
            # 如果是推理模型，处理思维链输出
            if model_name == "deepseek-r1":
                message = response.choices[0].message
                reasoning = message.reasoning_content
                answer = message.content
                
                if show_reasoning and reasoning:
                    return f"【思维过程】\n{reasoning}\n\n【最终答案】\n{answer}"
                return answer
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating with DeepSeek: {str(e)}")
            raise

    def _generate_with_ollama(
        self,
        model_name: str,
        query: str,
        context: str,
        show_reasoning: bool = True
    ) -> str:
        """
        使用Ollama API生成回答
        
        参数:
            model_name: 模型名称
            query: 用户查询
            context: 上下文信息
            show_reasoning: 是否显示推理过程（仅对推理模型有效）
            
        返回:
            生成的回答文本，对于推理模型可能包含思维过程
        """
        try:
            # 设置超时时间
            timeout = 60  # 增加到60秒超时
            
            # 创建 Ollama 客户端
            client = OllamaLLM(
                model=self.models["ollama"][model_name],
                base_url="http://localhost:11434",
                timeout=timeout
            )

            # 构建提示模板
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. Use the provided context to answer the question. If you cannot find the answer in the context, say so."),
                ("user", "{input}")
            ])
            
            # 创建输出解析器
            output_parser = StrOutputParser()
            
            # 构建处理链
            chain = prompt | client | output_parser

            # 构建输入
            input_text = f"Context: {context}\n\nQuestion: {query}"
            
            # 调用模型
            try:
                logger.debug(f"开始调用 Ollama 模型: {model_name}")
                response = chain.invoke({"input": input_text})
                if not response:
                    raise ValueError("模型返回空响应")
                logger.debug("Ollama 模型调用成功")
                return response
            except Exception as e:
                logger.error(f"调用 Ollama 模型时出错: {str(e)}")
                raise
            
        except Exception as e:
            logger.error(f"使用 Ollama 生成回答时出错: {str(e)}")
            raise

    def generate(self, provider: str, model_name: str, query: str, search_results: List[Dict] = None, collection_name: str = None, api_key: str = None, additional_context: str = None, context_file_ids: List[str] = None, context_contents: List[str] = None, chunked_file_ids: List[str] = None, generation_mode: str = 'search', show_reasoning: bool = True) -> Dict:
        """生成回答"""
        try:
            logger.info(f"开始生成回答 - 提供商: {provider}, 模型: {model_name}, 模式: {generation_mode}")
            
            # 获取模型配置
            model_config = self.get_model_config(provider, model_name)
            if not model_config:
                raise ValueError(f"不支持的模型: {provider}/{model_name}")
            
            # 准备上下文
            context_parts = []
            
            # 如果是直接生成模式，使用空上下文
            if generation_mode == 'direct':
                context = ""
                logger.info("使用直接生成模式，不提供上下文")
            else:
                # 添加搜索结果上下文
                if search_results:
                    logger.debug(f"处理搜索结果: {len(search_results)} 条")
                    # 从搜索结果中提取上下文
                    for result in search_results:
                        if isinstance(result, dict) and 'content' in result:
                            context_parts.append(f"搜索结果：\n{result['content']}\n")
                        else:
                            logger.warning(f"搜索结果格式不正确: {result}")
                        
                # 添加额外的上下文文件内容
                if additional_context:
                    logger.debug("处理额外上下文")
                    context_parts.append(f"补充上下文：\n{additional_context}\n")
                
                # 添加直接提供的上下文内容
                if context_contents:
                    logger.debug(f"处理直接提供的上下文内容: {len(context_contents)} 条")
                    for content in context_contents:
                        if content:  # 确保内容不为空
                            context_parts.append(f"文件内容：\n{content}\n")
                
                # 从文件中读取上下文
                if context_file_ids:
                    logger.debug(f"处理上下文文件: {context_file_ids}")
                    for file_id in context_file_ids:
                        try:
                            file_path = os.path.join("uploads", file_id)
                            logger.debug(f"尝试读取文件: {file_path}")
                            if os.path.exists(file_path):
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                    if content:
                                        context_parts.append(f"文件内容：\n{content}\n")
                                    else:
                                        logger.warning(f"文件内容为空: {file_path}")
                            else:
                                logger.warning(f"文件不存在: {file_path}")
                        except Exception as e:
                            logger.warning(f"读取文件 {file_id} 失败: {str(e)}")
                
                # 从分块文件中读取上下文
                if chunked_file_ids:
                    logger.debug(f"处理分块文件: {chunked_file_ids}")
                    for file_id in chunked_file_ids:
                        try:
                            file_path = os.path.join("01-chunked-docs", file_id)
                            logger.debug(f"尝试读取分块文件: {file_path}")
                            if os.path.exists(file_path):
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    data = json.load(f)
                                    if "chunks" in data:
                                        for chunk in data["chunks"]:
                                            if isinstance(chunk, dict):
                                                content = chunk.get("text", chunk.get("content", ""))
                                                if content:
                                                    context_parts.append(f"分块内容：\n{content}\n")
                                                else:
                                                    logger.warning(f"分块内容为空: {chunk}")
                                    else:
                                        logger.warning(f"分块文件格式不正确: {file_path}")
                            else:
                                logger.warning(f"分块文件不存在: {file_path}")
                        except Exception as e:
                            logger.warning(f"读取分块文件 {file_id} 失败: {str(e)}")
                        
                # 从collection中获取上下文
                if collection_name:
                    try:
                        logger.debug(f"从集合 {collection_name} 获取上下文")
                        # 连接到 Milvus
                        connections.connect(
                            alias="default",
                            uri=MILVUS_CONFIG["uri"]
                        )
                        
                        # 获取集合
                        collection = Collection(collection_name)
                        collection.load()
                        
                        # 执行向量搜索
                        search_params = {
                            "metric_type": "L2",
                            "params": {"nprobe": 10}
                        }
                        
                        # 获取查询向量
                        query_vector = self.get_query_embedding(query, provider, model_name, api_key)
                        logger.info(f"生成的查询向量维度: {len(query_vector)}")
                        
                        # 执行搜索
                        results = collection.search(
                            data=[query_vector],
                            anns_field="embedding",
                            param=search_params,
                            limit=5,  # 获取前5个最相似的结果
                            output_fields=["content", "metadata"]
                        )
                        
                        logger.info(f"向量搜索返回结果数量: {len(results[0]) if results else 0}")
                        
                        # 解析 metadata 字段，提取 chunk_id、page_number 等
                        for hit in results[0]:
                            content = getattr(hit.entity, "content", "")
                            metadata_str = getattr(hit.entity, "metadata", "{}")
                            logger.info(f"搜索结果 - 相似度: {hit.distance}, 内容长度: {len(content)}")
                            try:
                                metadata = json.loads(metadata_str)
                                logger.info(f"搜索结果元数据: {metadata}")
                            except Exception as e:
                                logger.error(f"解析元数据失败: {e}")
                                metadata = {}
                            # 可根据需要提取 chunk_id/page_number/document_name 等
                            context_parts.append(f"集合搜索结果：\n{content}\n")
                        
                    except Exception as e:
                        logger.error(f"从 collection 获取上下文时出错: {str(e)}")
                        raise
                    finally:
                        try:
                            connections.disconnect("default")
                        except Exception as e:
                            logger.warning(f"断开 Milvus 连接时出错: {str(e)}")
                
                # 合并所有上下文
                context = "\n".join(context_parts)
                logger.debug(f"合并后的上下文长度: {len(context)}")
                
                # 如果不是直接生成模式，且没有上下文，则报错
                if not context:
                    logger.error("没有找到任何上下文信息")
                    raise ValueError("无法获取上下文信息")
            
            # 根据不同的提供商调用相应的生成方法
            try:
                if provider == "huggingface":
                    response = self._generate_with_huggingface(model_name, query, context)
                elif provider == "openai":
                    response = self._generate_with_openai(model_name, query, context, api_key)
                elif provider == "deepseek":
                    response = self._generate_with_deepseek(model_name, query, context, api_key, show_reasoning)
                elif provider == "ollama":
                    response = self._generate_with_ollama(model_name, query, context, show_reasoning)
                else:
                    raise ValueError(f"不支持的提供商: {provider}")
                
                if not response:
                    raise ValueError("模型返回空响应")
                
                logger.info("成功生成回答")
                return {
                    "status": "success",
                    "data": {
                        "response": response,
                        "context": context if generation_mode != 'direct' else None
                    }
                }
                
            except Exception as e:
                logger.error(f"调用模型生成回答时出错: {str(e)}")
                raise
            
        except Exception as e:
            logger.error(f"生成回答时出错: {str(e)}")
            raise
        
    def get_query_embedding(self, query: str, provider: str, model_name: str, api_key: str = None) -> List[float]:
        """获取查询的向量表示"""
        try:
            # 获取模型配置
            model_config = self.get_model_config(provider, model_name)
            if not model_config:
                raise ValueError(f"不支持的模型: {provider}/{model_name}")
            
            # 调用模型获取向量
            if provider == "openai":
                response = openai.Embedding.create(
                    input=query,
                    model="text-embedding-3-small",  # 使用与集合相同的嵌入模型
                    api_key=api_key
                )
                # 确保返回的是密集向量（dense vector）
                embedding = response["data"][0]["embedding"]
                return [float(x) for x in embedding]  # 确保所有值都是 float 类型
            elif provider == "ollama":
                response = requests.post(
                    f"{model_config['base_url']}/api/embeddings",
                    json={
                        "model": "bge-m3:latest",  # 使用与集合相同的嵌入模型
                        "prompt": query
                    },
                    timeout=30  # 添加超时设置
                )
                if response.status_code != 200:
                    raise ValueError(f"获取向量失败: {response.text}")
                    
                # 确保返回的是密集向量（dense vector）
                embedding = response.json()["embedding"]
                if not isinstance(embedding, list):
                    raise ValueError("返回的向量格式不正确")
                    
                # 确保所有值都是 float 类型
                dense_vector = [float(x) for x in embedding]
                
                # 验证向量维度
                if len(dense_vector) != 1024:  # 假设维度是1024
                    raise ValueError(f"向量维度不正确: {len(dense_vector)}")
                    
                return dense_vector
            else:
                raise ValueError(f"不支持的提供商: {provider}")
            
        except Exception as e:
            logger.error(f"获取查询向量时出错: {str(e)}")
            raise ValueError(f"获取查询向量失败: {str(e)}")

    def get_available_models(self) -> Dict:
        """
        获取可用的模型列表
        
        返回:
            包含所有支持模型的字典
        """
        return self.models 

    def get_model_config(self, provider: str, model_name: str) -> Dict:
        """获取模型配置"""
        try:
            if provider not in self.models:
                raise ValueError(f"不支持的提供商: {provider}")
            
            if model_name not in self.models[provider]:
                raise ValueError(f"不支持的模型: {model_name}")
            
            # 获取模型配置
            model_config = {
                "provider": provider,
                "model_name": model_name,
                "base_url": "http://localhost:11434" if provider == "ollama" else None
            }
            
            return model_config
            
        except Exception as e:
            logger.error(f"获取模型配置时出错: {str(e)}")
            raise ValueError(f"获取模型配置失败: {str(e)}")

    def _build_prompt(self, query: str, context: str, model_config: Dict) -> str:
        """构建提示词"""
        return f"""请基于以下上下文回答问题。如果上下文中没有相关信息，请说明无法回答。

问题：{query}

上下文：
{context}

回答："""

    def _call_model(self, prompt: str, model_config: Dict, api_key: str = None) -> str:
        """调用模型生成回答"""
        try:
            provider = model_config["provider"]
            model_name = model_config["model_name"]
            
            # 从提示词中提取问题和上下文
            parts = prompt.split("\n\n")
            query = parts[0].replace("问题：", "").strip()
            context = parts[1].replace("上下文：", "").strip()
            
            if provider == "huggingface":
                return self._generate_with_huggingface(model_name, query, context)
            elif provider == "openai":
                return self._generate_with_openai(model_name, query, context, api_key)
            elif provider == "deepseek":
                return self._generate_with_deepseek(model_name, query, context, api_key)
            elif provider == "ollama":
                return self._generate_with_ollama(model_name, query, context)
            else:
                raise ValueError(f"不支持的提供商: {provider}")
            
        except Exception as e:
            logger.error(f"调用模型时出错: {str(e)}")
            raise ValueError(f"调用模型失败: {str(e)}")

    async def generate_stream(self, provider: str, model_name: str, query: str, search_results: List[Dict] = None, collection_name: str = None, api_key: str = None, additional_context: str = None, context_file_ids: List[str] = None, context_contents: List[str] = None, chunked_file_ids: List[str] = None, generation_mode: str = 'search', show_reasoning: bool = True):
        """流式生成回答"""
        try:
            logger.info(f"开始流式生成 - 提供商: {provider}, 模型: {model_name}")
            logger.info(f"查询: {query}")
            logger.info(f"集合名称: {collection_name}")
            logger.info(f"生成模式: {generation_mode}")
            
            # 获取模型配置
            model_config = self.get_model_config(provider, model_name)
            logger.info(f"模型配置: {model_config}")
            
            # 构建上下文
            context = ""
            if generation_mode == 'search' and collection_name:
                try:
                    logger.info("开始向量搜索...")
                    # 获取查询向量
                    query_vector = self.get_query_embedding(query, provider, model_name, api_key)
                    logger.info(f"查询向量生成成功，维度: {len(query_vector)}")
                    
                    # 执行向量搜索
                    search_params = {
                        "metric_type": "L2",
                        "params": {"nprobe": 10}
                    }
                    
                    # 连接到 Milvus
                    connections.connect(
                        alias="default",
                        uri=MILVUS_CONFIG["uri"]
                    )
                    logger.info("成功连接到 Milvus")
                    
                    # 获取集合
                    collection = Collection(collection_name)
                    collection.load()
                    logger.info(f"成功加载集合: {collection_name}")
                    
                    # 执行搜索
                    results = collection.search(
                        data=[query_vector],
                        anns_field="embedding",
                        param=search_params,
                        limit=5,
                        output_fields=["content", "metadata"]
                    )
                    
                    logger.info(f"搜索完成，返回结果数量: {len(results[0]) if results else 0}")
                    
                    # 处理搜索结果
                    search_contents = []
                    for hit in results[0]:
                        content = getattr(hit.entity, "content", "")
                        metadata_str = getattr(hit.entity, "metadata", "{}")
                        logger.info(f"搜索结果 - 相似度: {hit.distance}, 内容长度: {len(content)}")
                        
                        try:
                            metadata = json.loads(metadata_str)
                            logger.info(f"搜索结果元数据: {metadata}")
                        except Exception as e:
                            logger.error(f"解析元数据失败: {e}")
                            metadata = {}
                        
                        if content:
                            search_contents.append(content)
                    
                    # 构建上下文
                    if search_contents:
                        context = "\n".join(search_contents)
                        logger.info(f"构建的上下文长度: {len(context)}")
                    else:
                        logger.warning("未找到相关搜索结果")
                        
                except Exception as e:
                    logger.error(f"向量搜索失败: {str(e)}")
                    raise
                finally:
                    try:
                        connections.disconnect("default")
                    except Exception as e:
                        logger.warning(f"断开 Milvus 连接时出错: {e}")
            
            # 根据不同的提供商调用相应的生成方法
            try:
                if provider == "huggingface":
                    async for chunk in self._generate_with_huggingface_stream(model_name, query, context):
                        yield chunk
                elif provider == "openai":
                    async for chunk in self._generate_with_openai_stream(model_name, query, context, api_key):
                        yield chunk
                elif provider == "deepseek":
                    async for chunk in self._generate_with_deepseek_stream(model_name, query, context, api_key, show_reasoning):
                        yield chunk
                elif provider == "ollama":
                    async for chunk in self._generate_with_ollama_stream(model_name, query, context, show_reasoning):
                        yield chunk
                else:
                    raise ValueError(f"不支持的提供商: {provider}")
                
            except Exception as e:
                logger.error(f"调用模型生成回答时出错: {str(e)}")
                yield json.dumps({"type": "error", "error": str(e)})
            
        except Exception as e:
            logger.error(f"生成回答时出错: {str(e)}")
            yield json.dumps({"type": "error", "error": str(e)})

    async def _generate_with_ollama_stream(self, model_name: str, query: str, context: str, show_reasoning: bool = True):
        """使用Ollama API流式生成回答"""
        try:
            # 设置超时时间
            timeout = 60  # 60秒超时
            
            # 创建 Ollama 客户端
            client = OllamaLLM(
                model=self.models["ollama"][model_name],
                base_url="http://localhost:11434",
                timeout=timeout
            )

            # 构建提示模板
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. Use the provided context to answer the question. If you cannot find the answer in the context, say so."),
                ("user", "{input}")
            ])
            
            # 构建输入
            input_text = f"Context: {context}\n\nQuestion: {query}"
            
            # 调用模型
            try:
                logger.debug(f"开始调用 Ollama 模型: {model_name}")
                
                # 发送开始状态
                yield json.dumps({"type": "status", "status": "开始生成回答..."})
                
                # 流式生成回答
                async for chunk in client.astream(input_text):
                    if chunk:
                        yield json.dumps({"type": "content", "content": chunk})
                
                # 发送完成状态
                yield json.dumps({"type": "done"})
                
                logger.debug("Ollama 模型调用成功")
                
            except Exception as e:
                logger.error(f"调用 Ollama 模型时出错: {str(e)}")
                yield json.dumps({"type": "error", "error": str(e)})
            
        except Exception as e:
            logger.error(f"使用 Ollama 生成回答时出错: {str(e)}")
            yield json.dumps({"type": "error", "error": str(e)})

    async def _generate_with_openai_stream(self, model_name: str, query: str, context: str, api_key: str = None):
        """使用OpenAI API流式生成回答"""
        try:
            if not api_key:
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OpenAI API key not provided")
                    
            client = OpenAI(api_key=api_key)
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer the question."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
            ]
            
            # 发送开始状态
            yield json.dumps({"type": "status", "status": "开始生成回答..."})
            
            # 流式生成回答
            stream = await client.chat.completions.create(
                model=self.models["openai"][model_name],
                messages=messages,
                temperature=0.7,
                max_tokens=512,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield json.dumps({"type": "content", "content": chunk.choices[0].delta.content})
            
            # 发送完成状态
            yield json.dumps({"type": "done"})
            
        except Exception as e:
            logger.error(f"Error generating with OpenAI: {str(e)}")
            yield json.dumps({"type": "error", "error": str(e)})

    async def _generate_with_deepseek_stream(self, model_name: str, query: str, context: str, api_key: str = None, show_reasoning: bool = True):
        """使用DeepSeek API流式生成回答"""
        try:
            if not api_key:
                api_key = os.getenv("DEEPSEEK_API_KEY")
                if not api_key:
                    raise ValueError("DeepSeek API key not provided")
                    
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com"
            )
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer the question."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
            ]
            
            # 发送开始状态
            yield json.dumps({"type": "status", "status": "开始生成回答..."})
            
            # 流式生成回答
            stream = await client.chat.completions.create(
                model=self.models["deepseek"][model_name],
                messages=messages,
                max_tokens=512,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield json.dumps({"type": "content", "content": chunk.choices[0].delta.content})
            
            # 发送完成状态
            yield json.dumps({"type": "done"})
            
        except Exception as e:
            logger.error(f"Error generating with DeepSeek: {str(e)}")
            yield json.dumps({"type": "error", "error": str(e)})

    async def _generate_with_huggingface_stream(self, model_name: str, query: str, context: str, max_length: int = 512):
        """使用HuggingFace模型流式生成回答"""
        try:
            model, tokenizer = self._load_huggingface_model(model_name)
            
            # 构建提示
            prompt = f"""请基于以下上下文回答问题。如果上下文中没有相关信息，请说明无法回答。

                        问题：{query}

                        上下文：
                        {context}

                        回答："""
        
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # 发送开始状态
            yield json.dumps({"type": "status", "status": "开始生成回答..."})
            
            # 流式生成回答
            async for output in model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                stream=True
            ):
                if output:
                    text = tokenizer.decode(output, skip_special_tokens=True)
                    yield json.dumps({"type": "content", "content": text})
            
            # 发送完成状态
            yield json.dumps({"type": "done"})
            
        except Exception as e:
            logger.error(f"Error generating with HuggingFace: {str(e)}")
            yield json.dumps({"type": "error", "error": str(e)}) 
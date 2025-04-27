import os
import json
from datetime import datetime
from typing import List, Dict, Optional
import logging
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from openai import OpenAI
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import requests
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
            client = OllamaLLM(
                model=self.models["ollama"][model_name],
                base_url="http://localhost:11434"
            )

            prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a helpful assistant. Use the provided context to answer the question."),
                    ("user", "{input}" )
            ])
            output_parser = StrOutputParser()
            chain = prompt | client | output_parser

            #messages = {"model": "deepseek-r1:14b" , "prompt": f"Context: {context}\n\nQuestion: {query}"}
            
 #           response = client.chat.completions.create(
            response = chain.invoke({"input": f"Context: {context}\n\nQuestion: {query}"})
            
            # 如果是推理模型，处理思维链输出
 #           if response.status_code == 200:
 #               response_text = response.text
 #               data = json.loads(response_text)
 
 #               answer = data["response"]
#                reasoning = message.reasoning_content
                
#                if show_reasoning and reasoning:
#                    return f"【思维过程】\n{reasoning}\n\n【最终答案】\n{answer}"
#                return answer
 #           else:
 #               answer = "no answer"
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating with Ollama: {str(e)}")
            raise


    def generate(self, provider: str, model_name: str, query: str, search_results: List[Dict] = None, collection_name: str = None, api_key: str = None, additional_context: str = None) -> Dict:
        """生成回答"""
        try:
            # 获取模型配置
            model_config = self.get_model_config(provider, model_name)
            if not model_config:
                raise ValueError(f"不支持的模型: {provider}/{model_name}")
            
            # 准备上下文
            context_parts = []
            
            # 添加搜索结果上下文
            if search_results:
                # 从搜索结果中提取上下文
                for result in search_results:
                    context_parts.append(f"搜索结果：\n{result['content']}\n")
                    
            # 添加额外的上下文文件内容
            if additional_context:
                context_parts.append(f"补充上下文：\n{additional_context}\n")
                    
            # 从collection中获取上下文
            if collection_name:
                try:
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
                    
                    # 执行搜索
                    results = collection.search(
                        data=[query_vector],
                        anns_field="vector",
                        param=search_params,
                        limit=5,  # 获取前5个最相似的结果
                        output_fields=["content", "document_name", "chunk_id", "page_number"]
                    )
                    
                    # 提取上下文
                    for hit in results[0]:
                        context_parts.append(f"集合搜索结果：\n{hit.entity.content}\n")
                    
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
            
            if not context:
                raise ValueError("无法获取上下文信息")
            
            # 构建提示词
            prompt = self._build_prompt(query, context, model_config)
            
            # 调用模型生成回答
            response = self._call_model(prompt, model_config, api_key)
            
            return {
                "status": "success",
                "message": "生成成功",
                "data": {
                    "query": query,
                    "response": response,
                    "model": model_name,
                    "provider": provider,
                    "context": context[:1000]  # 只返回前1000个字符的上下文
                }
            }
            
        except Exception as e:
            logger.error(f"生成回答时出错: {str(e)}")
            raise ValueError(f"生成回答失败: {str(e)}")
        
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
                return response["data"][0]["embedding"]
            elif provider == "ollama":
                response = requests.post(
                    f"{model_config['base_url']}/api/embeddings",
                    json={
                        "model": "bge-m3:latest",  # 使用与集合相同的嵌入模型
                        "prompt": query
                    }
                )
                return response.json()["embedding"]
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
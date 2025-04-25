import logging
from typing import Dict, List
import fitz  # PyMuPDF
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

class ParsingService:
    """
    PDF文档解析服务类
    
    该类提供多种解析策略来提取和构建PDF文档内容，包括：
    - 全文提取
    - 逐页解析
    - 基于标题的分段
    - 文本和表格混合解析
    - 芯片物理设计文件解析（Netlist、LEF、LIB）
    """

    def parse_document(self, text: str, method: str, metadata: dict, page_map: list = None) -> dict:
        """
        使用指定方法解析文档

        参数:
            text (str): 文档的文本内容
            method (str): 解析方法
            metadata (dict): 文档元数据
            page_map (list): 包含每页内容和元数据的字典列表

        返回:
            dict: 解析后的文档数据
        """
        try:
            if not page_map:
                raise ValueError("Page map is required for parsing.")
            
            file_type = metadata.get("file_type", "").lower()
            
            if file_type == "pdf":
                return self.parse_pdf(text, method, metadata, page_map)
            elif file_type == "netlist":
                return self.parse_netlist(text, method, metadata, page_map)
            elif file_type == "lef":
                return self.parse_lef(text, method, metadata, page_map)
            elif file_type == "lib":
                return self.parse_lib(text, method, metadata, page_map)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
                
        except Exception as e:
            logger.error(f"Error in parse_document: {str(e)}")
            raise
            
    def parse_netlist(self, text: str, method: str, metadata: dict, page_map: list) -> dict:
        """
        解析 Netlist 文件
        
        参数:
            text (str): Netlist 文件内容
            method (str): 解析方法
            metadata (dict): 文档元数据
            page_map (list): 页面映射
            
        返回:
            dict: 解析后的文档数据
        """
        try:
            parsed_content = []
            
            if method == "all_text":
                parsed_content = self._parse_all_text(page_map)
            elif method == "by_modules":
                parsed_content = self._parse_by_modules(page_map)
            elif method == "by_ports":
                parsed_content = self._parse_by_ports(page_map)
            elif method == "by_instances":
                parsed_content = self._parse_by_instances(page_map)
            elif method == "by_pins":
                parsed_content = self._parse_by_pins_netlist(page_map)
            elif method == "by_nets":
                parsed_content = self._parse_by_nets(page_map)
            else:
                raise ValueError(f"Unsupported parsing method for netlist: {method}")
                
            document_data = {
                "metadata": {
                    "filename": metadata.get("filename", ""),
                    "total_modules": len(parsed_content),
                    "parsing_method": method,
                    "timestamp": datetime.now().isoformat()
                },
                "content": parsed_content
            }
            
            return document_data
            
        except Exception as e:
            logger.error(f"Error in parse_netlist: {str(e)}")
            raise
            
    def parse_lef(self, text: str, method: str, metadata: dict, page_map: list) -> dict:
        """
        解析 LEF 文件
        
        参数:
            text (str): LEF 文件内容
            method (str): 解析方法
            metadata (dict): 文档元数据
            page_map (list): 页面映射
            
        返回:
            dict: 解析后的文档数据
        """
        try:
            parsed_content = []
            
            if method == "all_text":
                parsed_content = self._parse_all_text(page_map)
            elif method == "by_layers":
                parsed_content = self._parse_by_layers(page_map)
            elif method == "by_macros":
                parsed_content = self._parse_by_macros(page_map)
            else:
                raise ValueError(f"Unsupported parsing method for LEF: {method}")
                
            document_data = {
                "metadata": {
                    "filename": metadata.get("filename", ""),
                    "total_sections": len(parsed_content),
                    "parsing_method": method,
                    "timestamp": datetime.now().isoformat()
                },
                "content": parsed_content
            }
            
            return document_data
            
        except Exception as e:
            logger.error(f"Error in parse_lef: {str(e)}")
            raise
            
    def parse_lib(self, text: str, method: str, metadata: dict, page_map: list) -> dict:
        """
        解析 LIB 文件
        
        参数:
            text (str): LIB 文件内容
            method (str): 解析方法
            metadata (dict): 文档元数据
            page_map (list): 页面映射
            
        返回:
            dict: 解析后的文档数据
        """
        try:
            parsed_content = []
            
            if method == "all_text":
                parsed_content = self._parse_all_text(page_map)
            elif method == "by_cells":
                parsed_content = self._parse_by_cells(page_map)
            elif method == "by_pins":
                parsed_content = self._parse_by_pins(page_map)
            else:
                raise ValueError(f"Unsupported parsing method for LIB: {method}")
                
            document_data = {
                "metadata": {
                    "filename": metadata.get("filename", ""),
                    "total_cells": len(parsed_content),
                    "parsing_method": method,
                    "timestamp": datetime.now().isoformat()
                },
                "content": parsed_content
            }
            
            return document_data
            
        except Exception as e:
            logger.error(f"Error in parse_lib: {str(e)}")
            raise
            
    def parse_pdf(self, text: str, method: str, metadata: dict, page_map: list = None) -> dict:
        """
        使用指定方法解析PDF文档

        参数:
            text (str): PDF文档的文本内容
            method (str): 解析方法 ('all_text', 'by_pages', 'by_titles', 或 'text_and_tables')
            metadata (dict): 文档元数据，包括文件名和其他属性
            page_map (list): 包含每页内容和元数据的字典列表

        返回:
            dict: 解析后的文档数据，包括元数据和结构化内容

        异常:
            ValueError: 当page_map为空或指定了不支持的解析方法时抛出
        """
        try:
            if not page_map:
                raise ValueError("Page map is required for parsing.")
            
            parsed_content = []
            total_pages = len(page_map)
            
            if method == "all_text":
                parsed_content = self._parse_all_text(page_map)
            elif method == "by_pages":
                parsed_content = self._parse_by_pages(page_map)
            elif method == "by_titles":
                parsed_content = self._parse_by_titles(page_map)
            elif method == "text_and_tables":
                parsed_content = self._parse_text_and_tables(page_map)
            else:
                raise ValueError(f"Unsupported parsing method: {method}")
                
            # Create document-level metadata
            document_data = {
                "metadata": {
                    "filename": metadata.get("filename", ""),
                    "total_pages": total_pages,
                    "parsing_method": method,
                    "timestamp": datetime.now().isoformat()
                },
                "content": parsed_content
            }
            
            return document_data
            
        except Exception as e:
            logger.error(f"Error in parse_pdf: {str(e)}")
            raise

    def _parse_all_text(self, page_map: list) -> list:
        """
        将文档中的所有文本内容提取为连续流

        参数:
            page_map (list): 包含每页内容的字典列表

        返回:
            list: 包含带页码的文本内容的字典列表
        """
        return [{
            "type": "Text",
            "content": page["text"],
            "page": page["page"]
        } for page in page_map]

    def _parse_by_pages(self, page_map: list) -> list:
        """
        逐页解析文档，保持页面边界

        参数:
            page_map (list): 包含每页内容的字典列表

        返回:
            list: 包含带页码的分页内容的字典列表
        """
        parsed_content = []
        for page in page_map:
            parsed_content.append({
                "type": "Page",
                "page": page["page"],
                "content": page["text"]
            })
        return parsed_content

    def _parse_by_titles(self, page_map: list) -> list:
        """
        通过识别标题来解析文档并将内容组织成章节

        使用简单的启发式方法识别标题：
        长度小于60个字符且全部大写的行被视为章节标题

        参数:
            page_map (list): 包含每页内容的字典列表

        返回:
            list: 包含带标题和页码的分章节内容的字典列表
        """
        parsed_content = []
        current_title = None
        current_content = []

        for page in page_map:
            lines = page["text"].split('\n')
            for line in lines:
                # Simple heuristic: consider lines with less than 60 chars and all caps as titles
                if len(line.strip()) < 60 and line.isupper():
                    if current_title:
                        parsed_content.append({
                            "type": "section",
                            "title": current_title,
                            "content": '\n'.join(current_content),
                            "page": page["page"]
                        })
                    current_title = line.strip()
                    current_content = []
                else:
                    current_content.append(line)

        # Add the last section
        if current_title:
            parsed_content.append({
                "type": "section",
                "title": current_title,
                "content": '\n'.join(current_content),
                "page": page["page"]
            })

        return parsed_content

    def _parse_text_and_tables(self, page_map: list) -> list:
        """
        通过分离文本和表格内容来解析文档

        使用基本的表格检测启发式方法（存在'|'或制表符）
        来识别潜在的表格内容

        参数:
            page_map (list): 包含每页内容的字典列表

        返回:
            list: 包含分离的文本和表格内容（带页码）的字典列表
        """
        parsed_content = []
        for page in page_map:
            # Extract tables using tabula-py or similar library
            # For this example, we'll just simulate table detection
            content = page["text"]
            if '|' in content or '\t' in content:
                parsed_content.append({
                    "type": "table",
                    "content": content,
                    "page": page["page"]
                })
            else:
                parsed_content.append({
                    "type": "text",
                    "content": content,
                    "page": page["page"]
                })
        return parsed_content

    def _parse_by_modules(self, page_map: list) -> list:
        """按模块解析 Netlist"""
        return [{
            "type": "module",
            "name": page["metadata"]["name"],
            "content": page["text"],
            "page": page["page"]
        } for page in page_map]
        
    def _parse_by_ports(self, page_map: list) -> list:
        """按端口解析 Netlist"""
        parsed_content = []
        for page in page_map:
            lines = page["text"].split('\n')
            for line in lines:
                line = line.strip()
                # 检查多种端口定义格式
                if (line.startswith('.PIN') or 
                    line.startswith('.PORT') or 
                    line.startswith('input') or 
                    line.startswith('output') or 
                    line.startswith('inout')):
                    # 提取端口名称
                    parts = line.split()
                    port_name = parts[1] if len(parts) > 1 else "Unknown"
                    # 提取端口类型
                    port_type = parts[0].replace('.', '').upper()
                    parsed_content.append({
                        "type": "port",
                        "name": port_name,
                        "content": line,
                        "page": page["page"],
                        "port_type": port_type
                    })
        return parsed_content
        
    def _parse_by_layers(self, page_map: list) -> list:
        """按层解析 LEF"""
        return [{
            "type": "layer",
            "name": page["metadata"]["name"],
            "content": page["text"],
            "page": page["page"]
        } for page in page_map if page["metadata"]["type"] == "layer"]
        
    def _parse_by_macros(self, page_map: list) -> list:
        """按宏单元解析 LEF"""
        return [{
            "type": "macro",
            "name": page["metadata"]["name"],
            "content": page["text"],
            "page": page["page"]
        } for page in page_map if page["metadata"]["type"] == "macro"]
        
    def _parse_by_cells(self, page_map: list) -> list:
        """按单元解析 LIB"""
        return [{
            "type": "cell",
            "name": page["metadata"]["name"],
            "content": page["text"],
            "page": page["page"]
        } for page in page_map]
        
    def _parse_by_pins(self, page_map: list) -> list:
        """按引脚解析 LIB"""
        parsed_content = []
        for page in page_map:
            lines = page["text"].split('\n')
            for line in lines:
                if line.strip().startswith('pin'):
                    parsed_content.append({
                        "type": "pin",
                        "name": line.split()[1] if len(line.split()) > 1 else "Unknown",
                        "content": line,
                        "page": page["page"]
                    })
        return parsed_content

    def _parse_by_instances(self, page_map: list) -> list:
        """按实例解析 Netlist"""
        parsed_content = []
        for page in page_map:
            lines = page["text"].split('\n')
            for line in lines:
                line = line.strip()
                # 检查实例定义格式
                if line.startswith('X') or line.startswith('U') or line.startswith('I'):
                    parts = line.split()
                    if len(parts) >= 2:
                        instance_name = parts[0]
                        instance_type = parts[-1]  # 最后一个部分通常是实例类型
                        # 提取引脚连接
                        pins = parts[1:-1] if len(parts) > 2 else []
                        parsed_content.append({
                            "type": "instance",
                            "name": instance_name,
                            "instance_type": instance_type,
                            "pins": pins,
                            "content": line,
                            "page": page["page"]
                        })
        return parsed_content

    def _parse_by_pins_netlist(self, page_map: list) -> list:
        """按引脚解析 Netlist"""
        parsed_content = []
        for page in page_map:
            lines = page["text"].split('\n')
            for line in lines:
                line = line.strip()
                # 检查引脚定义格式
                if (line.startswith('.PIN') or 
                    line.startswith('.PORT') or 
                    line.startswith('input') or 
                    line.startswith('output') or 
                    line.startswith('inout') or
                    line.startswith('pin')):
                    # 提取引脚名称和类型
                    parts = line.split()
                    pin_name = parts[1] if len(parts) > 1 else "Unknown"
                    pin_type = parts[0].replace('.', '').upper()
                    # 提取引脚连接信息
                    connections = []
                    if len(parts) > 2:
                        connections = parts[2:]
                    parsed_content.append({
                        "type": "pin",
                        "name": pin_name,
                        "pin_type": pin_type,
                        "connections": connections,
                        "content": line,
                        "page": page["page"]
                    })
        return parsed_content

    def _parse_by_nets(self, page_map: list) -> list:
        """按网络解析 Netlist"""
        parsed_content = []
        current_net = None
        current_connections = []
        
        for page in page_map:
            lines = page["text"].split('\n')
            for line in lines:
                line = line.strip()
                # 检查网络定义格式
                if line.startswith('net') or line.startswith('NET'):
                    # 如果已经有当前网络，保存它
                    if current_net:
                        parsed_content.append({
                            "type": "net",
                            "name": current_net,
                            "connections": current_connections,
                            "content": '\n'.join(current_connections),
                            "page": page["page"]
                        })
                    # 开始新的网络
                    parts = line.split()
                    current_net = parts[1] if len(parts) > 1 else "Unknown"
                    current_connections = [line]
                elif current_net and line:
                    # 继续当前网络
                    current_connections.append(line)
                    
        # 保存最后一个网络
        if current_net:
            parsed_content.append({
                "type": "net",
                "name": current_net,
                "connections": current_connections,
                "content": '\n'.join(current_connections),
                "page": page["page"]
            })
            
        return parsed_content 
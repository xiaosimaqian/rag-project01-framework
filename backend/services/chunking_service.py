from datetime import datetime
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Generator, Dict, List, Any, Optional
import tempfile
import re
import gc
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from dataclasses import dataclass
from tqdm import tqdm

logger = logging.getLogger(__name__)

@dataclass
class ChunkProgress:
    """分块进度跟踪"""
    total_modules: int
    processed_modules: int = 0
    total_chunks: int = 0
    processed_chunks: int = 0
    
    def update_module(self):
        self.processed_modules += 1
        
    def update_chunks(self, count: int):
        self.processed_chunks += count
        
    @property
    def module_progress(self) -> float:
        return (self.processed_modules / self.total_modules * 100) if self.total_modules > 0 else 0
        
    @property
    def chunk_progress(self) -> float:
        return (self.processed_chunks / self.total_chunks * 100) if self.total_chunks > 0 else 0

@dataclass
class ModuleInfo:
    """模块信息数据结构"""
    name: str
    ports: List[str]
    port_info: List[Dict[str, Any]]
    instance_info: List[Dict[str, Any]]
    declarations: List[str]
    relations: Optional[Dict[str, Any]] = None

class ChunkingService:
    """
    文本分块服务，提供多种文本分块策略
    
    该服务支持以下分块方法：
    - by_pages: 按页面分块，每页作为一个块
    - fixed_size: 按固定大小分块
    - by_paragraphs: 按段落分块
    - by_sentences: 按句子分块
    - by_semicolons: 按分号分块(Verilog文件)
    - by_module: 对大型Verilog模块进行结构化分块
    - by_hdl_struct: 使用HDL解析库进行结构化分块
    """
    
    def __init__(self):
        """初始化分块服务"""
        self.progress = None
        self.max_workers = min(32, (os.cpu_count() or 1) + 4)
        
    def _stream_parse_file(self, file_path: str) -> Generator[ModuleInfo, None, None]:
        """流式解析文件，逐模块处理"""
        try:
            from pyverilog.vparser.parser import parse
            ast, _ = parse([file_path])
            
            # 计算总模块数
            total_modules = sum(1 for desc in ast.description.definitions 
                              if desc.__class__.__name__ == 'ModuleDef')
            self.progress = ChunkProgress(total_modules=total_modules)
            
            for desc in ast.description.definitions:
                if desc.__class__.__name__ == 'ModuleDef':
                    module_info = self._process_module(desc)
                    self.progress.update_module()
                    yield module_info
                    # 主动进行垃圾回收
                    gc.collect()
                    
        except Exception as e:
            logger.error(f"解析文件时出错: {str(e)}")
            raise
            
    def _process_module(self, module_def) -> ModuleInfo:
        """处理单个模块"""
        try:
            module_name = str(module_def.name)
            ports = [str(p.name) for p in module_def.portlist.ports]
            
            # 使用生成器处理模块项
            port_info = []
            instance_info = []
            declarations = []
            
            for item in module_def.items:
                if item.__class__.__name__ == 'Decl':
                    port_info.extend(self._process_declaration(item))
                elif item.__class__.__name__ == 'InstanceList':
                    instance_info.extend(self._process_instance(item))
                else:
                    declarations.append(str(item))
                    
            # 创建模块信息对象
            module_info = ModuleInfo(
                name=module_name,
                ports=ports,
                port_info=port_info,
                instance_info=instance_info,
                declarations=declarations
            )
            
            # 分析模块关系
            module_info.relations = self._analyze_module_relations(module_info)
            
            return module_info
            
        except Exception as e:
            logger.error(f"处理模块时出错: {str(e)}")
            raise
            
    def _process_declaration(self, decl_item) -> List[Dict[str, Any]]:
        """处理声明项"""
        port_info = []
        for decl in decl_item.list:
            if hasattr(decl, 'name'):
                info = {
                    "name": str(decl.name),
                    "type": decl.__class__.__name__,
                    "width": str(decl.width) if hasattr(decl, 'width') else None,
                    "dimensions": str(decl.dimensions) if hasattr(decl, 'dimensions') else None
                }
                port_info.append(info)
        return port_info
        
    def _process_instance(self, instance_item) -> List[Dict[str, Any]]:
        """处理实例化项"""
        instance_info = []
        for inst in instance_item.instances:
            port_mapping = {}
            for c in inst.portlist:
                if c.portname and c.argname:
                    port_mapping[str(c.portname)] = {
                        "connected_to": str(c.argname),
                        "direction": self._get_port_direction(str(c.portname))
                    }
            
            instance_info.append({
                "name": str(inst.name),
                "module": str(instance_item.module),
                "connections": port_mapping
            })
        return instance_info
        
    def _get_port_direction(self, port_name: str) -> str:
        """获取端口方向"""
        port_name = port_name.lower()
        if "input" in port_name:
            return "input"
        elif "output" in port_name:
            return "output"
        return "inout"
        
    def _analyze_module_relations(self, module_info: ModuleInfo) -> Dict[str, Any]:
        """分析模块间的关系"""
        relations = {
            "port_connections": {},
            "instance_connections": {},
            "hierarchy": {}
        }
        
        # 分析端口连接
        for port in module_info.port_info:
            port_name = port['name']
            relations['port_connections'][port_name] = {
                "type": port['type'],
                "connected_instances": []
            }
        
        # 分析实例连接
        for inst in module_info.instance_info:
            inst_name = inst['name']
            relations['instance_connections'][inst_name] = {
                "module": inst['module'],
                "ports": inst['connections']
            }
            
            # 更新端口连接信息
            for port_name, conn in inst['connections'].items():
                if conn['connected_to'] in relations['port_connections']:
                    relations['port_connections'][conn['connected_to']]['connected_instances'].append(inst_name)
        
        return relations
        
    def _create_chunks(self, module_info: ModuleInfo, chunk_size: int) -> List[Dict[str, Any]]:
        """创建分块"""
        chunks = []
        
        # 1. 模块头部
        header_chunk = {
            "text": f"module {module_info.name} (\n    " + 
                   ",\n    ".join(module_info.ports) + "\n);\n",
            "hdl_struct": {
                "module": module_info.name,
                "type": "module_header",
                "ports": module_info.port_info
            }
        }
        chunks.append(header_chunk)
        
        # 2. 端口声明
        port_chunks = self._create_port_chunks(module_info, chunk_size)
        chunks.extend(port_chunks)
        
        # 3. 实例化
        instance_chunks = self._create_instance_chunks(module_info, chunk_size)
        chunks.extend(instance_chunks)
        
        # 4. 其他声明
        if module_info.declarations:
            decl_chunks = self._create_declaration_chunks(module_info, chunk_size)
            chunks.extend(decl_chunks)
            
        # 更新进度
        self.progress.update_chunks(len(chunks))
        
        return chunks
        
    def _create_port_chunks(self, module_info: ModuleInfo, chunk_size: int) -> List[Dict[str, Any]]:
        """创建端口声明分块，包含进度跟踪"""
        chunks = []
        total_ports = len(module_info.port_info)
        
        for i in range(0, total_ports, chunk_size):
            port_chunk = module_info.port_info[i:i + chunk_size]
            progress = (i + len(port_chunk)) / total_ports * 100
            
            try:
                chunk_text = "// Port declarations\n"
                for port in port_chunk:
                    decl_text = f"{port['type']} {port['name']}"
                    if port['width']:
                        decl_text += f" [{port['width']}]"
                    if port['dimensions']:
                        decl_text += f" [{port['dimensions']}]"
                    decl_text += ";\n"
                    chunk_text += decl_text
                    
                chunks.extend(self.split_large_chunk({
                    "text": chunk_text,
                    "hdl_struct": {
                        "module": module_info.name,
                        "type": "module_ports",
                        "ports": port_chunk,
                        "progress": progress
                    }
                }, max_size=5000))
                
            except Exception as e:
                logger.error(f"创建端口分块时出错: {str(e)}")
                # 添加错误恢复
                chunks.append({
                    "text": f"// Error in port declarations {i} to {i + len(port_chunk)}",
                    "hdl_struct": {
                        "module": module_info.name,
                        "type": "error",
                        "error": str(e),
                        "progress": progress
                    }
                })
                
        return chunks
        
    def _create_instance_chunks(self, module_info: ModuleInfo, chunk_size: int) -> List[Dict[str, Any]]:
        """创建实例化分块"""
        chunks = []
        total_instances = len(module_info.instance_info)
        
        for i in range(0, total_instances, chunk_size):
            instance_chunk = module_info.instance_info[i:i + chunk_size]
            progress = (i + len(instance_chunk)) / total_instances * 100
            
            try:
                # 添加实例化描述信息
                chunk_text = f"// Module: {module_info.name}\n"
                chunk_text += f"// Instance Group {i//chunk_size + 1}/{(total_instances + chunk_size - 1)//chunk_size}\n"
                chunk_text += f"// Total Instances: {total_instances}\n"
                chunk_text += f"// Instances in this group: {len(instance_chunk)}\n"
                chunk_text += f"// Instance range: {i+1}-{i+len(instance_chunk)}\n"
                chunk_text += "// Module instantiations\n"
                
                for inst in instance_chunk:
                    chunk_text += f"{inst['module']} {inst['name']} (\n"
                    for port_name, conn in inst['connections'].items():
                        chunk_text += f"    .{port_name}({conn['connected_to']}),\n"
                    chunk_text = chunk_text.rstrip(",\n") + "\n);\n"
                    
                chunks.extend(self.split_large_chunk({
                    "text": chunk_text,
                    "hdl_struct": {
                        "module": module_info.name,
                        "type": "module_instances",
                        "instances": instance_chunk,
                        "progress": progress,
                        "relations": module_info.relations,
                        "group_index": i//chunk_size + 1,
                        "total_groups": (total_instances + chunk_size - 1)//chunk_size,
                        "instance_range": f"{i+1}-{i+len(instance_chunk)}"
                    }
                }, max_size=5000))
                
            except Exception as e:
                logger.error(f"创建实例分块时出错: {str(e)}")
                chunks.append({
                    "text": f"// Error in instance declarations {i} to {i + len(instance_chunk)}",
                    "hdl_struct": {
                        "module": module_info.name,
                        "type": "error",
                        "error": str(e),
                        "progress": progress
                    }
                })
                
        return chunks
        
    def _create_declaration_chunks(self, module_info: ModuleInfo, chunk_size: int) -> List[Dict[str, Any]]:
        """创建其他声明分块"""
        chunks = []
        total_decls = len(module_info.declarations)
        
        for i in range(0, total_decls, chunk_size):
            decl_chunk = module_info.declarations[i:i + chunk_size]
            progress = (i + len(decl_chunk)) / total_decls * 100
            
            try:
                chunk_text = "// Other declarations\n"
                chunk_text += "\n".join(decl_chunk)
                
                chunks.extend(self.split_large_chunk({
                    "text": chunk_text,
                    "hdl_struct": {
                        "module": module_info.name,
                        "type": "other_declarations",
                        "declarations": decl_chunk,
                        "progress": progress
                    }
                }, max_size=5000))
                
            except Exception as e:
                logger.error(f"创建声明分块时出错: {str(e)}")
                chunks.append({
                    "text": f"// Error in declarations {i} to {i + len(decl_chunk)}",
                    "hdl_struct": {
                        "module": module_info.name,
                        "type": "error",
                        "error": str(e),
                        "progress": progress
                    }
                })
                
        return chunks
        
    def _hdl_struct_chunks(self, text: str) -> list:
        """使用简化的解析策略，将HDL代码分块"""
        chunks = []
        global_chunk_index = 1  # 添加全局分块索引计数器

        try:
            # 添加调试日志
            logger.info(f"输入文本长度: {len(text)}")
            logger.debug(f"输入文本前200个字符: {text[:200]}")
            
            # 预处理文本
            # 1. 移除多余的空白字符
            text = re.sub(r'\s+', ' ', text)
            # 2. 确保每个语句后都有分号
            text = re.sub(r'([^;])\n', r'\1;\n', text)
            # 3. 规范化换行符
            text = text.replace('\r\n', '\n').replace('\r', '\n')
            
            # 1. 使用更简单的正则表达式预分块
            # 首先匹配module关键字和模块名
            module_pattern = r'module\s+(\w+)'
            module_matches = list(re.finditer(module_pattern, text))
            
            if not module_matches:
                logger.warning("未找到任何模块声明")
                return [{
                    "text": text,
                    "hdl_struct": {
                        "type": "raw",
                        "error": "No module declarations found"
                    }
                }]
            
            logger.info(f"找到 {len(module_matches)} 个模块声明")
            
            # 2. 串行处理模块，以确保正确的索引顺序
            for i, match in enumerate(module_matches):
                module_name = match.group(1)
                start_pos = match.start()
                
                # 添加调试日志
                logger.info(f"找到模块: {module_name}, 起始位置: {start_pos}")
                
                # 提取完整的模块文本
                if i < len(module_matches) - 1:
                    end_pos = module_matches[i + 1].start()
                else:
                    end_pos = len(text)
                
                module_text = text[start_pos:end_pos]
                
                # 添加调试日志
                logger.info(f"提取的模块文本长度: {len(module_text)}")
                logger.debug(f"模块文本前200个字符: {module_text[:200]}")
                
                # 预处理模块文本
                # 1. 移除注释
                module_text = re.sub(r'//.*?\n', '\n', module_text)
                # 2. 规范化空白字符
                module_text = re.sub(r'\s+', ' ', module_text)
                # 3. 确保每个语句后都有分号
                module_text = re.sub(r'([^;])\n', r'\1;\n', module_text)
                
                # 添加调试日志
                logger.debug(f"预处理后的模块文本前200个字符: {module_text[:200]}")
                
                # 处理单个模块并获取其分块
                module_chunks = self._process_single_module(module_text, module_name, global_chunk_index)
                if module_chunks:
                    chunks.extend(module_chunks)
                    # 更新全局索引为最后一个分块的索引加1
                    global_chunk_index = module_chunks[-1]["hdl_struct"]["chunk_index"] + 1
                    logger.info(f"成功处理模块 {module_name}，生成了 {len(module_chunks)} 个分块，当前全局索引: {global_chunk_index}")
                else:
                    logger.warning(f"模块 {module_name} 处理未生成任何分块")
            
            # 3. 处理剩余的代码（非模块部分）
            # 移除所有已处理的模块文本
            remaining_text = text
            for i, match in enumerate(module_matches):
                start_pos = match.start()
                if i < len(module_matches) - 1:
                    end_pos = module_matches[i + 1].start()
                else:
                    end_pos = len(text)
                remaining_text = remaining_text.replace(text[start_pos:end_pos], "")
            
            if remaining_text.strip():
                # 检查剩余文本是否包含有效的Verilog代码
                if re.search(r'(input|output|inout|wire|reg|module|endmodule)', remaining_text):
                    logger.warning("发现未处理的Verilog代码，可能包含不完整的模块")
                chunks.append({
                    "text": remaining_text.strip(),
                    "hdl_struct": {
                        "type": "other",
                        "content_type": "non_module_code",
                        "chunk_index": global_chunk_index
                    }
                })
            
            # 添加调试日志
            logger.info(f"总共生成了 {len(chunks)} 个分块")
            
            return chunks
            
        except Exception as e:
            logger.error(f"HDL结构分块时出错: {str(e)}")
            return [{
                "text": text,
                "hdl_struct": {
                    "type": "raw",
                    "error": str(e)
                }
            }]

    def _process_single_module(self, module_text: str, module_name: str, start_chunk_index: int, parent_module: str = None) -> list:
        """
        优化版：避免重复分块，统计更准确，支持分块最大长度
        """
        chunks = []
        MAX_CHUNK_SIZE = 4000  # 设置适合embedding model的大小限制
        current_chunk_index = start_chunk_index  # 使用传入的起始索引

        try:
            # 1. 提取模块声明和端口列表
            # 使用更健壮的正则表达式来匹配模块声明
            module_decl_pattern = r'module\s+(\w+)\s*\((.*?)\)\s*;'
            module_decl_match = re.search(module_decl_pattern, module_text, re.DOTALL)
            
            if module_decl_match:
                module_name = module_decl_match.group(1)
                port_list = module_decl_match.group(2)
                
                # 处理端口列表，移除多余的空白字符
                ports = []
                for port in port_list.split(','):
                    port = port.strip()
                    if port:
                        ports.append(port)
                
                # 将端口列表分成多个较小的组
                port_groups = []
                current_group = []
                current_length = 0
                
                for port in ports:
                    port_length = len(port) + 2  # 加上逗号和空格的长度
                    if current_length + port_length > 100:  # 每行最多100个字符
                        port_groups.append(current_group)
                        current_group = [port]
                        current_length = port_length
                    else:
                        current_group.append(port)
                        current_length += port_length
                
                if current_group:
                    port_groups.append(current_group)
                
                # 创建格式化的模块声明
                module_decl = f"module {module_name} (\n"
                for group in port_groups:
                    module_decl += "    " + ", ".join(group) + ",\n"
                module_decl = module_decl.rstrip(",\n") + "\n);"
                
                # 如果模块声明太大，分成多个chunk
                if len(module_decl) > MAX_CHUNK_SIZE:
                    # 将模块声明分成多个部分
                    header = f"// Module: {module_name}\nmodule {module_name} (\n"
                    chunks.append({
                        "text": header,
                        "hdl_struct": {
                            "module": module_name,
                            "type": "module_header_start",
                            "parent_module": parent_module,
                            "chunk_index": current_chunk_index,
                            "total_ports": len(ports),
                            "port_groups": len(port_groups)
                        }
                    })
                    current_chunk_index += 1
                    
                    # 为每个端口组创建一个chunk
                    for i, group in enumerate(port_groups):
                        # 添加上下文信息
                        context = f"// Module: {module_name}\n// Port Group {i+1}/{len(port_groups)}\n// Total Ports: {len(ports)}\n"
                        port_text = context + "    " + ", ".join(group) + ("," if i < len(port_groups) - 1 else "")
                        chunks.append({
                            "text": port_text,
                            "hdl_struct": {
                                "module": module_name,
                                "type": "module_ports",
                                "port_group": i + 1,
                                "total_groups": len(port_groups),
                                "parent_module": parent_module,
                                "chunk_index": current_chunk_index,
                                "total_ports": len(ports),
                                "ports_in_group": len(group),
                                "port_names": group
                            }
                        })
                        current_chunk_index += 1
                    
                    # 添加结束部分
                    chunks.append({
                        "text": ");",
                        "hdl_struct": {
                            "module": module_name,
                            "type": "module_header_end",
                            "parent_module": parent_module,
                            "chunk_index": current_chunk_index,
                            "total_ports": len(ports)
                        }
                    })
                    current_chunk_index += 1
                else:
                    # 如果模块声明不大，保持为一个chunk
                    chunks.append({
                        "text": f"// Module: {module_name}\n{module_decl}",
                        "hdl_struct": {
                            "module": module_name,
                            "type": "module_header",
                            "ports": ports,
                            "parent_module": parent_module,
                            "chunk_index": current_chunk_index,
                            "total_ports": len(ports)
                        }
                    })
                    current_chunk_index += 1

            # 2. 提取端口声明和wire声明
            port_pattern = r'(input|output|inout)\s+(?:\[([^\]]+)\])?\s*([\w, ]+)\s*;'
            wire_pattern = r'wire\s+(?:\[([^\]]+)\])?\s*([\w, ]+)\s*;'
            port_info = []
            wire_info = []
            port_lines = []
            wire_lines = []

            for match in re.finditer(port_pattern, module_text):
                port_type = match.group(1)
                width = match.group(2)
                names = match.group(3)
                for name in [n.strip() for n in names.split(',') if n.strip()]:
                    port_info.append({
                        "name": name,
                        "type": port_type,
                        "width": width,
                        "direction": port_type,
                        "connected_nets": [],
                        "connected_instances": []
                    })
                port_lines.append(match.group(0))

            for match in re.finditer(wire_pattern, module_text):
                width = match.group(1)
                names = match.group(2)
                for name in [n.strip() for n in names.split(',') if n.strip()]:
                    wire_info.append({
                        "name": name,
                        "width": width,
                        "connections": [],
                        "connected_ports": [],
                        "connected_instances": []
                    })
                wire_lines.append(match.group(0))

            # 将声明分成多个较小的chunk
            if port_lines or wire_lines:
                all_lines = port_lines + wire_lines
                current_chunk = []
                current_length = 0
                
                for line in all_lines:
                    line_length = len(line) + 1  # 加上换行符
                    if current_length + line_length > MAX_CHUNK_SIZE:
                        # 创建当前chunk
                        if current_chunk:
                            decl_text = f"// Module: {module_name}\n" + "\n".join(current_chunk)
                            chunks.append({
                                "text": decl_text,
                                "hdl_struct": {
                                    "module": module_name,
                                    "type": "module_declarations",
                                    "parent_module": parent_module,
                                    "chunk_index": current_chunk_index
                                }
                            })
                            current_chunk_index += 1
                        current_chunk = [line]
                        current_length = line_length
                    else:
                        current_chunk.append(line)
                        current_length += line_length
                
                # 处理最后一个chunk
                if current_chunk:
                    decl_text = f"// Module: {module_name}\n" + "\n".join(current_chunk)
                    chunks.append({
                        "text": decl_text,
                        "hdl_struct": {
                            "module": module_name,
                            "type": "module_declarations",
                            "parent_module": parent_module,
                            "chunk_index": current_chunk_index
                        }
                    })
                    current_chunk_index += 1

            # 3. 提取实例化
            instance_pattern = r'(\w+)\s+(\w+)\s*\(([^;]*?)\)\s*;'
            instance_info = []
            instance_lines = []
            net_connections = {}

            for match in re.finditer(instance_pattern, module_text):
                module_type = match.group(1)
                instance_name = match.group(2)
                if instance_name == module_name:
                    continue
                connections = match.group(3)
                conn_pattern = r'\.(\w+)\s*\(\s*([^,\)]+)\s*\)'
                port_mapping = {}
                for conn in re.finditer(conn_pattern, connections):
                    port = conn.group(1)
                    net = conn.group(2).strip()
                    port_mapping[port] = net
                    if net not in net_connections:
                        net_connections[net] = []
                    net_connections[net].append({"instance": instance_name, "port": port})
                instance_info.append({
                    "name": instance_name,
                    "module": module_type,
                    "connections": port_mapping,
                    "parent_module": module_name
                })
                instance_lines.append(match.group(0))

            # 将实例化分成多个较小的chunk
            if instance_lines:
                current_chunk = []
                current_length = 0
                chunk_index = 0
                total_chunks = (len(instance_lines) + 50 - 1) // 50  # 每50个实例一个chunk
                
                for line in instance_lines:
                    line_length = len(line) + 1  # 加上换行符
                    if current_length + line_length > MAX_CHUNK_SIZE or len(current_chunk) >= 50:
                        # 创建当前chunk
                        if current_chunk:
                            chunk_index += 1
                            inst_text = f"// Module: {module_name}\n"
                            inst_text += f"// Instance Group {chunk_index}/{total_chunks}\n"
                            inst_text += f"// Total Instances: {len(instance_lines)}\n"
                            inst_text += f"// Instances in this group: {len(current_chunk)}\n"
                            inst_text += f"// Instance range: {chunk_index*50-49}-{min(chunk_index*50, len(instance_lines))}\n"
                            inst_text += "// Module instantiations\n"
                            inst_text += "\n".join(current_chunk)
                            chunks.append({
                                "text": inst_text,
                                "hdl_struct": {
                                    "module": module_name,
                                    "type": "module_instances",
                                    "parent_module": parent_module,
                                    "chunk_index": current_chunk_index,
                                    "group_index": chunk_index,
                                    "total_groups": total_chunks,
                                    "instance_range": f"{chunk_index*50-49}-{min(chunk_index*50, len(instance_lines))}"
                                }
                            })
                            current_chunk_index += 1
                        current_chunk = [line]
                        current_length = line_length
                    else:
                        current_chunk.append(line)
                        current_length += line_length
                
                # 处理最后一个chunk
                if current_chunk:
                    chunk_index += 1
                    inst_text = f"// Module: {module_name}\n"
                    inst_text += f"// Instance Group {chunk_index}/{total_chunks}\n"
                    inst_text += f"// Total Instances: {len(instance_lines)}\n"
                    inst_text += f"// Instances in this group: {len(current_chunk)}\n"
                    inst_text += f"// Instance range: {chunk_index*50-49}-{min(chunk_index*50, len(instance_lines))}\n"
                    inst_text += "// Module instantiations\n"
                    inst_text += "\n".join(current_chunk)
                    chunks.append({
                        "text": inst_text,
                        "hdl_struct": {
                            "module": module_name,
                            "type": "module_instances",
                            "parent_module": parent_module,
                            "chunk_index": current_chunk_index,
                            "group_index": chunk_index,
                            "total_groups": total_chunks,
                            "instance_range": f"{chunk_index*50-49}-{min(chunk_index*50, len(instance_lines))}"
                        }
                    })
                    current_chunk_index += 1

            # 4. 其他声明
            remaining_text = module_text
            if module_decl_match:
                remaining_text = remaining_text[len(module_decl_match.group(0)):]
            for line in port_lines + wire_lines + instance_lines:
                remaining_text = remaining_text.replace(line, "")
            other_lines = [line.strip() for line in remaining_text.split('\n') if line.strip()]
            if other_lines:
                # 移除endmodule（如果存在）
                other_lines = [line for line in other_lines if line != "endmodule"]
                
                # 将其他声明分成多个较小的chunk
                current_chunk = []
                current_length = 0
                chunk_index = 0
                total_chunks = (len(other_lines) + 50 - 1) // 50  # 每50行一个chunk
                
                for line in other_lines:
                    line_length = len(line) + 1  # 加上换行符
                    if current_length + line_length > MAX_CHUNK_SIZE or len(current_chunk) >= 50:
                        # 创建当前chunk
                        if current_chunk:
                            chunk_index += 1
                            other_text = f"// Module: {module_name}\n"
                            other_text += f"// Other Declarations Group {chunk_index}/{total_chunks}\n"
                            other_text += f"// Total Lines: {len(other_lines)}\n"
                            other_text += f"// Lines in this group: {len(current_chunk)}\n"
                            other_text += f"// Line range: {chunk_index*50-49}-{min(chunk_index*50, len(other_lines))}\n"
                            other_text += "\n".join(current_chunk)
                            chunks.append({
                                "text": other_text,
                                "hdl_struct": {
                                    "module": module_name,
                                    "type": "module_other",
                                    "parent_module": parent_module,
                                    "chunk_index": current_chunk_index,
                                    "group_index": chunk_index,
                                    "total_groups": total_chunks,
                                    "line_range": f"{chunk_index*50-49}-{min(chunk_index*50, len(other_lines))}"
                                }
                            })
                            current_chunk_index += 1
                        current_chunk = [line]
                        current_length = line_length
                    else:
                        current_chunk.append(line)
                        current_length += line_length
                
                # 处理最后一个chunk
                if current_chunk:
                    chunk_index += 1
                    other_text = f"// Module: {module_name}\n"
                    other_text += f"// Other Declarations Group {chunk_index}/{total_chunks}\n"
                    other_text += f"// Total Lines: {len(other_lines)}\n"
                    other_text += f"// Lines in this group: {len(current_chunk)}\n"
                    other_text += f"// Line range: {chunk_index*50-49}-{min(chunk_index*50, len(other_lines))}\n"
                    other_text += "\n".join(current_chunk)
                    chunks.append({
                        "text": other_text,
                        "hdl_struct": {
                            "module": module_name,
                            "type": "module_other",
                            "parent_module": parent_module,
                            "chunk_index": current_chunk_index,
                            "group_index": chunk_index,
                            "total_groups": total_chunks,
                            "line_range": f"{chunk_index*50-49}-{min(chunk_index*50, len(other_lines))}"
                        }
                    })
                    current_chunk_index += 1

            # 5. 统计信息
            port_conn_count = {p["name"]: 0 for p in port_info}
            wire_conn_count = {w["name"]: 0 for w in wire_info}
            for net, conns in net_connections.items():
                if net in port_conn_count:
                    port_conn_count[net] += len(conns)
                if net in wire_conn_count:
                    wire_conn_count[net] += len(conns)

            stats = {
                "total_ports": len(port_info),
                "total_wires": len(wire_info),
                "total_instances": len(instance_info),
                "total_nets": len(net_connections),
                "module_name": module_name,
                "parent_module": parent_module,
                "port_details": {p["name"]: {"type": p["type"], "width": p["width"], "connections": port_conn_count[p["name"]]} for p in port_info},
                "wire_details": {w["name"]: {"width": w["width"], "connections": wire_conn_count[w["name"]]} for w in wire_info},
                "instance_details": {i["name"]: {"module": i["module"], "connections": len(i["connections"])} for i in instance_info}
            }

            # 计算模块的起始和结束索引
            module_start_index = start_chunk_index  # 使用传入的起始索引
            module_end_index = current_chunk_index + 3  # 当前分块索引 + 3个统计分块

            # 将统计信息分成多个较小的chunk
            # 添加统计信息chunk
            chunks.append({
                "text": f"// Module: {module_name}\n// Module Statistics:\n// Module Name: {stats['module_name']}\n" + 
                       (f"// Parent Module: {stats['parent_module']}\n" if stats['parent_module'] else "") +
                       f"// Total Ports: {stats['total_ports']}\n" +
                       f"// Total Wires: {stats['total_wires']}\n" +
                       f"// Total Instances: {stats['total_instances']}\n" +
                       f"// Total Nets: {stats['total_nets']}\n" +
                       f"// Total Chunks: {module_end_index - module_start_index + 1}\n" +
                       f"// Module Chunks Range: {module_start_index}-{module_end_index}\n",
                "hdl_struct": {
                    "module": module_name,
                    "type": "module_stats_header",
                    "parent_module": parent_module,
                    "total_chunks": module_end_index - module_start_index + 1,
                    "module_start_index": module_start_index,
                    "module_end_index": module_end_index,
                    "chunk_index": current_chunk_index
                }
            })
            current_chunk_index += 1

            # 添加端口详情
            port_details_text = f"// Module: {module_name}\n// Port Details:\n"
            for port_name, details in stats["port_details"].items():
                port_details_text += f"//   {port_name}: {details['type']} {details['width'] or ''} ({details['connections']} connections)\n"
            chunks.append({
                "text": port_details_text,
                "hdl_struct": {
                    "module": module_name,
                    "type": "module_stats_ports",
                    "parent_module": parent_module,
                    "total_chunks": module_end_index - module_start_index + 1,
                    "module_start_index": module_start_index,
                    "module_end_index": module_end_index,
                    "chunk_index": current_chunk_index
                }
            })
            current_chunk_index += 1

            # 添加wire详情
            wire_details_text = f"// Module: {module_name}\n// Wire Details:\n"
            for wire_name, details in stats["wire_details"].items():
                wire_details_text += f"//   {wire_name}: {details['width'] or ''} ({details['connections']} connections)\n"
            chunks.append({
                "text": wire_details_text,
                "hdl_struct": {
                    "module": module_name,
                    "type": "module_stats_wires",
                    "parent_module": parent_module,
                    "total_chunks": module_end_index - module_start_index + 1,
                    "module_start_index": module_start_index,
                    "module_end_index": module_end_index,
                    "chunk_index": current_chunk_index
                }
            })
            current_chunk_index += 1

            # 添加实例详情
            instance_details_text = f"// Module: {module_name}\n// Instance Details:\n"
            instance_lines = []
            for inst_name, details in stats["instance_details"].items():
                instance_lines.append(f"//   {inst_name}: {details['module']} ({details['connections']} connections)")
            
            # 将实例详情分成多个较小的chunk
            current_chunk = []
            current_length = 0
            chunk_header = f"// Module: {module_name}\n// Instance Details:\n"
            
            for line in instance_lines:
                line_length = len(line) + 1  # 加上换行符
                if current_length + line_length > MAX_CHUNK_SIZE - len(chunk_header):
                    # 创建当前chunk
                    if current_chunk:
                        chunk_text = chunk_header + "\n".join(current_chunk)
                        chunks.append({
                            "text": chunk_text,
                            "hdl_struct": {
                                "module": module_name,
                                "type": "module_stats_instances",
                                "parent_module": parent_module,
                                "total_chunks": module_end_index - module_start_index + 1,
                                "module_start_index": module_start_index,
                                "module_end_index": module_end_index,
                                "chunk_index": current_chunk_index,
                                "is_final_chunk": False
                            }
                        })
                    current_chunk = [line]
                    current_length = line_length
                else:
                    current_chunk.append(line)
                    current_length += line_length
            
            # 处理最后一个chunk
            if current_chunk:
                chunk_text = chunk_header + "\n".join(current_chunk)
                chunks.append({
                    "text": chunk_text + "\nendmodule",
                    "hdl_struct": {
                        "module": module_name,
                        "type": "module_stats_instances",
                        "parent_module": parent_module,
                        "total_chunks": module_end_index - module_start_index + 1,
                        "module_start_index": module_start_index,
                        "module_end_index": module_end_index,
                        "chunk_index": current_chunk_index,
                        "is_final_chunk": True
                    }
                })
                current_chunk_index += 1

            # 更新每个分块的索引
            for i, chunk in enumerate(chunks):
                if "hdl_struct" in chunk:
                    chunk["hdl_struct"]["chunk_index"] = module_start_index + i
                    chunk["hdl_struct"]["module_start_index"] = module_start_index
                    chunk["hdl_struct"]["module_end_index"] = module_end_index

            logger.info(f"模块 {module_name} 处理完成，生成了 {len(chunks)} 个分块，索引范围: {module_start_index}-{module_end_index}")
            return chunks

        except Exception as e:
            logger.error(f"处理模块 {module_name} 时出错: {str(e)}")
            return [{
                "text": f"// Module: {module_name}\n{module_text}",
                "hdl_struct": {
                    "module": module_name,
                    "type": "module_raw",
                    "error": str(e),
                    "parent_module": parent_module,
                    "chunk_index": start_chunk_index
                }
            }]
    
    def chunk_text(self, text: str, method: str, metadata: dict, page_map: list = None, chunk_size: int = 1000, max_chunk_size: int = 2048) -> dict:
        """
        将文本按指定方法分块
        新增max_chunk_size参数，递归细分时可自定义最大分块长度
        
        Args:
            text: 原始文本内容
            method: 分块方法，支持 'by_pages', 'fixed_size', 'by_paragraphs', 'by_sentences', 'by_semicolons', 'by_module', 'by_hdl_struct'
            metadata: 文档元数据
            page_map: 页面映射列表，每个元素包含页码和页面文本
            chunk_size: 固定大小分块时的块大小
            
        Returns:
            包含分块结果的文档数据结构
        
        Raises:
            ValueError: 当分块方法不支持或页面映射为空时
        """
        try:
            if not page_map:
                raise ValueError("Page map is required for chunking.")
            
            logger.info(f"开始分块处理，方法: {method}, 页数: {len(page_map)}")
            
            # 确保元数据是可序列化的
            def make_serializable(obj):
                if isinstance(obj, (str, int, float, bool, type(None))):
                    return obj
                elif isinstance(obj, dict):
                    return {k: make_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [make_serializable(item) for item in obj]
                else:
                    return str(obj)
            
            # 处理输入元数据
            metadata = make_serializable(metadata)
            
            chunks = []
            total_pages = len(page_map)
            
            # 记录开始时间
            start_time = datetime.now()
            
            if method == "by_hdl_struct":
                logger.info("使用 HDL 结构化分块方法")
                for page_data in page_map:
                    try:
                        page_chunks = self._hdl_struct_chunks(page_data['text'])
                        logger.info(f"页面 {page_data['page']} 分块完成，获得 {len(page_chunks)} 个块")
                        
                        for chunk in page_chunks:
                            chunk_metadata = {
                                **metadata,
                                "file_name": metadata.get("file_name", metadata.get("filename", "")),
                                "chunk_id": len(chunks) + 1,
                                "page_number": page_data['page'],
                                "page_range": str(page_data['page']),
                                "word_count": len(chunk["text"].split()),
                                "text": chunk["text"],
                                "chunk_type": "hdl_struct",
                                "chunk_index": len(chunks) + 1,
                                "chunk_size": len(chunk["text"]),
                                "chunk_start": 0,
                                "chunk_end": len(chunk["text"]),
                                "hdl_struct": chunk.get("hdl_struct", {}),
                                "loading_method": metadata.get("loading_method", "unknown") if metadata else "unknown"
                            }
                            
                            # 验证分块内容
                            if not self._validate_chunk(chunk_metadata):
                                logger.warning(f"分块 {chunk_metadata['chunk_id']} 验证失败，跳过")
                                continue
                                
                            chunks.append({
                                "text": chunk["text"],
                                "metadata": chunk_metadata
                            })
                            
                    except Exception as e:
                        logger.error(f"处理页面 {page_data['page']} 时出错: {str(e)}")
                        continue
                        
            elif method == "by_pages":
                # 直接使用 page_map 中的每页作为一个 chunk
                for page_data in page_map:
                    # 合并文档级元数据和块级元数据
                    chunk_metadata = {
                        **metadata,
                        "file_name": metadata.get("file_name", metadata.get("filename", "")),
                        "chunk_id": len(chunks) + 1,
                        "page_number": int(page_data['page']),
                        "page_range": str(page_data['page']),
                        "word_count": len(page_data['text'].split()),
                        "text": page_data['text'],
                        "chunk_type": "netlist",
                        "chunk_index": len(chunks) + 1,
                        "chunk_size": len(page_data['text']),
                        "chunk_start": 0,
                        "chunk_end": len(page_data['text']),
                        "loading_method": metadata.get("loading_method", "unknown") if metadata else "unknown"
                    }
                    chunks.append({
                        "text": page_data['text'],
                        "metadata": chunk_metadata
                    })
            
            elif method == "fixed_size":
                # 对每页内容进行固定大小分块
                for page_data in page_map:
                    page_chunks = self._fixed_size_chunks(page_data['text'], chunk_size)
                    for idx, chunk in enumerate(page_chunks, 1):
                        # 合并文档级元数据和块级元数据
                        chunk_metadata = {
                            **metadata,
                            "file_name": metadata.get("file_name", metadata.get("filename", "")),
                            "chunk_id": len(chunks) + 1,
                            "page_number": page_data['page'],
                            "page_range": str(page_data['page']),
                            "word_count": len(chunk["text"].split()),
                            "text": chunk["text"],
                            "chunk_type": "netlist",
                            "chunk_index": len(chunks) + 1,
                            "chunk_size": len(chunk["text"]),
                            "chunk_start": text.find(chunk["text"]),
                            "chunk_end": text.find(chunk["text"]) + len(chunk["text"]),
                            "loading_method": metadata.get("loading_method", "unknown") if metadata else "unknown"
                        }
                        chunks.append({
                            "text": chunk["text"],
                            "metadata": chunk_metadata
                        })
            
            elif method in ["by_paragraphs", "by_sentences"]:
                # 对每页内容进行段落或句子分块
                splitter_method = self._paragraph_chunks if method == "by_paragraphs" else self._sentence_chunks
                for page_data in page_map:
                    page_chunks = splitter_method(page_data['text'])
                    for chunk in page_chunks:
                        # 合并文档级元数据和块级元数据
                        chunk_metadata = {
                            **metadata,
                            "file_name": metadata.get("file_name", metadata.get("filename", "")),
                            "chunk_id": len(chunks) + 1,
                            "page_number": page_data['page'],
                            "page_range": str(page_data['page']),
                            "word_count": len(chunk["text"].split()),
                            "text": chunk["text"],
                            "chunk_type": "netlist",
                            "chunk_index": len(chunks) + 1,
                            "chunk_size": len(chunk["text"]),
                            "chunk_start": text.find(chunk["text"]),
                            "chunk_end": text.find(chunk["text"]) + len(chunk["text"]),
                            "loading_method": metadata.get("loading_method", "unknown") if metadata else "unknown"
                        }
                        chunks.append({
                            "text": chunk["text"],
                            "metadata": chunk_metadata
                        })
            
            elif method == "by_semicolons":
                # 对Verilog文件按分号分块
                for page_data in page_map:
                    page_chunks = self._verilog_chunks(page_data['text'], metadata)
                    current_module = None
                    for chunk in page_chunks:
                        import re
                        # 检查并更新当前模块名
                        if "module" in chunk["text"]:
                            m = re.search(r'module\s+(\w+)', chunk["text"])
                            if m:
                                current_module = m.group(1)
                                # 单独生成声明chunk
                                decl_line = chunk["text"].split('\n')[0]
                                decl_metadata = {
                                    **metadata,
                                    "file_name": metadata.get("file_name", metadata.get("filename", "")),
                                    "chunk_id": len(chunks) + 1,
                                    "page_number": page_data['page'],
                                    "page_range": str(page_data['page']),
                                    "word_count": len(decl_line.split()),
                                    "text": decl_line,
                                    "chunk_type": "module_decl",
                                    "chunk_index": len(chunks) + 1,
                                    "chunk_size": len(decl_line),
                                    "chunk_start": text.find(decl_line),
                                    "chunk_end": text.find(decl_line) + len(decl_line),
                                    "module_name": current_module,
                                    "is_module_decl": True,
                                    "loading_method": metadata.get("loading_method", "unknown") if metadata else "unknown"
                                }
                                chunks.append({
                                    "text": decl_line,
                                    "metadata": decl_metadata
                                })
                        # 提取首行注释
                        comment = None
                        lines = chunk["text"].splitlines()
                        for line in lines:
                            if line.strip().startswith("//"):
                                comment = line.strip()[2:].strip()
                                break
                        # 拼接上下文
                        enhanced_text = self._add_context_to_chunk(
                            chunk["text"],
                            file_name=metadata.get("file_name", ""),
                            module_name=current_module,
                            comment=comment
                        )
                        chunk_metadata = {
                            **metadata,
                            "file_name": metadata.get("file_name", metadata.get("filename", "")),
                            "chunk_id": len(chunks) + 1,
                            "page_number": page_data['page'],
                            "page_range": str(page_data['page']),
                            "word_count": len(chunk["text"].split()),
                            "text": chunk["text"],
                            "chunk_type": "netlist",
                            "chunk_index": len(chunks) + 1,
                            "chunk_size": len(chunk["text"]),
                            "chunk_start": text.find(chunk["text"]),
                            "chunk_end": text.find(chunk["text"]) + len(chunk["text"]),
                            "module_name": current_module,
                            "loading_method": metadata.get("loading_method", "unknown") if metadata else "unknown"
                        }
                        chunks.append({
                            "text": enhanced_text,
                            "metadata": chunk_metadata
                        })
            elif method == "by_module":
                # 对大型Verilog模块进行结构化分块
                for page_data in page_map:
                    page_chunks = self._verilog_module_chunks(page_data['text'])
                    current_module = None
                    for chunk in page_chunks:
                        import re
                        if "module" in chunk["text"]:
                            m = re.search(r'module\s+(\w+)', chunk["text"])
                            if m:
                                current_module = m.group(1)
                                decl_line = chunk["text"].split('\n')[0]
                                decl_metadata = {
                                    **metadata,
                                    "file_name": metadata.get("file_name", metadata.get("filename", "")),
                                    "chunk_id": len(chunks) + 1,
                                    "page_number": page_data['page'],
                                    "page_range": str(page_data['page']),
                                    "word_count": len(decl_line.split()),
                                    "text": decl_line,
                                    "chunk_type": "module_decl",
                                    "chunk_index": len(chunks) + 1,
                                    "chunk_size": len(decl_line),
                                    "chunk_start": text.find(decl_line),
                                    "chunk_end": text.find(decl_line) + len(decl_line),
                                    "module_name": current_module,
                                    "is_module_decl": True,
                                    "loading_method": metadata.get("loading_method", "unknown") if metadata else "unknown"
                                }
                                # 判断声明是否超长，超长则细分
                                if len(decl_line) > max_chunk_size:
                                    sub_chunks = self._split_large_chunk(decl_line, decl_metadata, max_tokens=max_chunk_size)
                                    chunks.extend(sub_chunks)
                                else:
                                    chunks.append({
                                        "text": decl_line,
                                        "metadata": decl_metadata
                                    })
                        comment = None
                        lines = chunk["text"].splitlines()
                        for line in lines:
                            if line.strip().startswith("//"):
                                comment = line.strip()[2:].strip()
                                break
                        enhanced_text = self._add_context_to_chunk(
                            chunk["text"],
                            file_name=metadata.get("file_name", ""),
                            module_name=current_module,
                            comment=comment
                        )
                        chunk_metadata = {
                            **metadata,
                            "file_name": metadata.get("file_name", metadata.get("filename", "")),
                            "chunk_id": len(chunks) + 1,
                            "page_number": page_data['page'],
                            "page_range": str(page_data['page']),
                            "word_count": len(chunk["text"].split()),
                            "text": chunk["text"],
                            "chunk_type": "netlist_module",
                            "chunk_index": len(chunks) + 1,
                            "chunk_size": len(chunk["text"]),
                            "chunk_start": text.find(chunk["text"]),
                            "chunk_end": text.find(chunk["text"]) + len(chunk["text"]),
                            "module_name": current_module,
                            "loading_method": metadata.get("loading_method", "unknown") if metadata else "unknown"
                        }
                        # 判断结构块是否超长，超长则细分
                        if len(chunk["text"]) > max_chunk_size:
                            sub_chunks = self._split_large_chunk(chunk["text"], chunk_metadata, max_tokens=max_chunk_size)
                            chunks.extend(sub_chunks)
                        else:
                            chunks.append({
                                "text": enhanced_text,
                                "metadata": chunk_metadata
                            })
            else:
                raise ValueError(f"Unsupported chunking method: {method}")

            # 记录结束时间和统计信息
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            # 创建标准化的文档数据结构
            document_data = {
                "filename": metadata.get("file_name", ""),
                "total_chunks": len(chunks),
                "total_pages": total_pages,
                "loading_method": metadata.get("loading_method", "unknown") if metadata else "unknown",
                "chunking_method": method,
                "timestamp": datetime.now().isoformat(),
                "processing_time": processing_time,
                "chunks": chunks
            }
            
            # 验证最终结果
            if not self._validate_document(document_data):
                raise ValueError("文档验证失败")
                
            # 保存分块结果
            try:
                output_dir = "backend/02-parsed-docs/chunks"
                os.makedirs(output_dir, exist_ok=True)
                
                output_file = os.path.join(output_dir, f"{metadata.get('file_name', 'output')}_chunks.json")
                with open(output_file, 'w') as f:
                    json.dump(document_data, f, indent=2)
                    
                logger.info(f"分块结果已保存到: {output_file}")
                logger.info(f"分块统计: 总块数={len(chunks)}, 处理时间={processing_time}秒")
                
            except Exception as e:
                logger.error(f"保存分块结果时出错: {str(e)}")
                raise
            
            return document_data
            
        except Exception as e:
            logger.error(f"分块处理失败: {str(e)}")
            raise

    def _fixed_size_chunks(self, text: str, chunk_size: int) -> list[dict]:
        """
        将文本按固定大小分块
        
        Args:
            text: 要分块的文本
            chunk_size: 每块的最大字符数
            
        Returns:
            分块后的文本列表
        """
        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + (1 if current_length > 0 else 0)
            if current_length + word_length > chunk_size and current_chunk:
                chunks.append({"text": " ".join(current_chunk)})
                current_chunk = []
                current_length = 0
            current_chunk.append(word)
            current_length += word_length
            
        if current_chunk:
            chunks.append({"text": " ".join(current_chunk)})
            
        return chunks

    def _paragraph_chunks(self, text: str) -> list[dict]:
        """
        将文本按段落分块
        
        Args:
            text: 要分块的文本
            
        Returns:
            分块后的段落列表
        """
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        return [{"text": para} for para in paragraphs]

    def _sentence_chunks(self, text: str) -> list[dict]:
        """
        将文本按句子分块
        
        Args:
            text: 要分块的文本
            
        Returns:
            分块后的句子列表
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=[".", "!", "?", "\n", " "]
        )
        texts = splitter.split_text(text)
        return [{"text": t} for t in texts]

    def _verilog_chunks(self, text: str, metadata: dict = None) -> list[dict]:
        """
        将Verilog文本按分号分块
        
        Args:
            text: 要分块的Verilog文本
            metadata: 文档元数据
        
        Returns:
            分块后的文本列表
        """
        chunks = []
        current_module = None
        module_info = {
            'ports': {},        # 存储模块端口定义
            'instances': {},    # 存储实例及其端口连接
            'nets': set()       # 存储所有网络
        }
        
        # 按分号分割语句
        statements = [s.strip() + ';' for s in text.split(';') if s.strip()]
        
        for stmt in statements:
            # 跳过注释
            if stmt.startswith('//'):
                continue
            
            # 1. 检测模块定义
            if 'module' in stmt and '(' in stmt:
                module_name = stmt.split('(')[0].split()[1]
                current_module = module_name
                module_info['ports'][module_name] = []
                module_info['instances'][module_name] = []
                
            # 2. 检测端口定义
            elif any(stmt.startswith(port) for port in ['input ', 'output ', 'inout ']):
                if current_module:
                    port_type = stmt.split()[0]
                    # 处理多端口声明的情况
                    port_names = stmt[len(port_type):].strip().rstrip(';').split(',')
                    for port_name in port_names:
                        port_name = port_name.strip()
                        module_info['ports'][current_module].append({
                            'name': port_name,
                            'type': port_type
                        })
                    
            # 3. 检测实例化
            elif ' u_' in stmt and '(' in stmt:
                inst_name = stmt.split('(')[0].strip()
                module_type = stmt.split(' u_')[0].strip()
                
                # 解析端口连接
                port_connections = {}
                if '(' in stmt and ')' in stmt:
                    connections = stmt[stmt.find('(')+1:stmt.rfind(')')].split(',')
                    for conn in connections:
                        if '.' in conn:
                            port, net = conn.split('(')
                            port = port.strip().lstrip('.')
                            net = net.strip().rstrip(')')
                            port_connections[port] = net
                            module_info['nets'].add(net)
                
                if current_module:
                    module_info['instances'][current_module].append({
                        'name': inst_name,
                        'type': module_type,
                        'connections': port_connections
                    })
                
            # 4. 添加到chunks时包含完整的关系信息
            chunk_metadata = {
                'chunk_type': 'netlist',
                'module_name': current_module,
                'module_info': {
                    'ports': module_info['ports'].get(current_module, []),
                    'instances': module_info['instances'].get(current_module, []),
                    'nets': list(module_info['nets'])
                },
                'loading_method': metadata.get("loading_method", "unknown") if metadata else "unknown"
            }
            chunks.append({
                'text': stmt,
                'metadata': chunk_metadata
            })
        
        return chunks

    def _verilog_module_chunks(self, text: str, max_statements: int = 50) -> list[dict]:
        """
        将大型Verilog模块进一步分块
        
        Args:
            text: 要分块的Verilog文本
            max_statements: 每个块中最大语句数量
            
        Returns:
            分块后的文本列表
        """
        chunks = []
        # 首先按模块分块
        module_chunks = self._verilog_chunks(text)
        
        for module_chunk in module_chunks:
            module_text = module_chunk["text"]
            
            # 检查是否是模块定义
            if 'module' in module_text and 'endmodule' in module_text:
                # 提取模块声明部分
                module_declaration = module_text.split(';')[0] + ';'
                
                # 提取模块内容（不包括endmodule）
                module_content = module_text[len(module_declaration):-len('endmodule;')]
                
                # 按分号分割语句
                statements = [s.strip() + ';' for s in module_content.split(';') if s.strip()]
                
                # 如果语句数量超过阈值，进行进一步分块
                if len(statements) > max_statements:
                    # 按端口声明、参数声明、内部信号声明、实例化等进行分组
                    port_declarations = []
                    param_declarations = []
                    signal_declarations = []
                    instantiations = []
                    other_statements = []
                    
                    for stmt in statements:
                        stmt_lower = stmt.lower()
                        if 'input' in stmt_lower or 'output' in stmt_lower or 'inout' in stmt_lower:
                            port_declarations.append(stmt)
                        elif 'parameter' in stmt_lower:
                            param_declarations.append(stmt)
                        elif any(keyword in stmt_lower for keyword in ['wire', 'reg', 'logic']):
                            signal_declarations.append(stmt)
                        elif any(keyword in stmt_lower for keyword in ['module', 'primitive']):
                            instantiations.append(stmt)
                        else:
                            other_statements.append(stmt)
                    
                    # 创建分块，确保最后一个块包含endmodule
                    if port_declarations:
                        chunks.append({"text": module_declaration + '\n'.join(port_declarations)})
                    if param_declarations:
                        chunks.append({"text": '\n'.join(param_declarations)})
                    if signal_declarations:
                        chunks.append({"text": '\n'.join(signal_declarations)})
                    
                    # 对实例化语句进行分组，每组最多max_statements个
                    if instantiations:
                        for i in range(0, len(instantiations), max_statements):
                            group = instantiations[i:i + max_statements]
                            chunks.append({"text": '\n'.join(group)})
                    
                    # 处理其他语句
                    if other_statements:
                        for i in range(0, len(other_statements), max_statements):
                            group = other_statements[i:i + max_statements]
                            # 如果是最后一组，添加endmodule
                            if i + max_statements >= len(other_statements):
                                chunks.append({"text": '\n'.join(group) + '\nendmodule;'})
                            else:
                                chunks.append({"text": '\n'.join(group)})
                    else:
                        # 如果没有其他语句，将endmodule添加到最后一个非空块
                        if chunks:
                            last_chunk = chunks[-1]
                            last_chunk["text"] += '\nendmodule;'
                        else:
                            # 如果没有任何语句，创建一个只包含endmodule的块
                            chunks.append({"text": 'endmodule;'})
                else:
                    # 如果语句数量在阈值内，保持原样
                    chunks.append(module_chunk)
            else:
                # 非模块定义的内容保持原样
                chunks.append(module_chunk)
        
        return chunks

    def _add_context_to_chunk(self, chunk_text, file_name=None, module_name=None, comment=None):
        context_lines = []
        if file_name:
            context_lines.append(f"// file: {file_name}")
        if module_name:
            context_lines.append(f"// module: {module_name}")
        if comment:
            context_lines.append(f"// comment: {comment}")
        return '\n'.join(context_lines + [chunk_text])

    def _split_large_chunk(self, chunk, max_size=5000):
        """
        将超大分块按max_size递归细分，保留原hdl_struct信息
        """
        text = chunk["text"]
        hdl_struct = chunk["hdl_struct"]
        if len(text) <= max_size:
            return [chunk]
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + max_size, len(text))
            sub_text = text[start:end]
            sub_chunk = {
                "text": sub_text,
                "hdl_struct": hdl_struct.copy()
            }
            chunks.append(sub_chunk)
            start = end
        return chunks

    def _validate_chunk(self, chunk_metadata: dict) -> bool:
        """验证单个分块的有效性"""
        try:
            # 检查必要字段
            required_fields = ['chunk_id', 'text', 'chunk_type', 'chunk_size']
            for field in required_fields:
                if field not in chunk_metadata:
                    logger.warning(f"分块缺少必要字段: {field}")
                    return False
                    
            # 检查文本内容
            if not chunk_metadata['text'].strip():
                logger.warning(f"分块 {chunk_metadata['chunk_id']} 内容为空")
                return False
                
            # 检查分块大小
            if chunk_metadata['chunk_size'] > 50000:  # 增加大小限制到50KB
                logger.warning(f"分块 {chunk_metadata['chunk_id']} 大小超出限制: {chunk_metadata['chunk_size']} 字节")
                # 不再跳过大分块，而是记录警告
                return True
                
            # 检查 HDL 结构信息
            if chunk_metadata['chunk_type'] == 'hdl_struct':
                hdl_struct = chunk_metadata.get('hdl_struct', {})
                if not hdl_struct:
                    logger.warning(f"分块 {chunk_metadata['chunk_id']} 缺少 HDL 结构信息")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"验证分块时出错: {str(e)}")
            return False
            
    def _validate_document(self, document_data: dict) -> bool:
        """验证整个文档的有效性"""
        try:
            # 检查文档结构
            if not isinstance(document_data, dict):
                logger.error("文档数据格式错误")
                return False
                
            # 检查必要字段
            required_fields = ['filename', 'total_chunks', 'chunks']
            for field in required_fields:
                if field not in document_data:
                    logger.error(f"文档缺少必要字段: {field}")
                    return False
                    
            # 检查分块数量
            if document_data['total_chunks'] != len(document_data['chunks']):
                logger.error("分块数量不匹配")
                return False
                
            # 检查分块内容
            for chunk in document_data['chunks']:
                if not self._validate_chunk(chunk['metadata']):
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"验证文档时出错: {str(e)}")
            return False

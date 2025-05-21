from pypdf import PdfReader
from unstructured.partition.pdf import partition_pdf
import pdfplumber
import fitz  # PyMuPDF
import logging
import os
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)
"""
PDF文档加载服务类
    这个服务类提供了多种PDF文档加载方法，支持不同的加载策略和分块选项。
    主要功能：
    1. 支持多种PDF解析库：
        - PyMuPDF (fitz): 适合快速处理大量PDF文件，性能最佳
        - PyPDF: 适合简单的PDF文本提取，依赖较少
        - pdfplumber: 适合需要处理表格或需要文本位置信息的场景
        - unstructured: 适合需要更好的文档结构识别和灵活分块策略的场景
    
    2. 文档加载特性：
        - 保持页码信息
        - 支持文本分块
        - 提供元数据存储
        - 支持不同的加载策略（使用unstructured时）
 """
class LoadingService:
    """
    PDF文档加载服务类，提供多种PDF文档加载和处理方法。
    
    属性:
        total_pages (int): 当前加载PDF文档的总页数
        current_page_map (list): 存储当前文档的页面映射信息，每个元素包含页面文本和页码
    """
    
    def __init__(self):
        self.total_pages = 0
        self.current_page_map = []
    
    def load_document(self, file_path: str, loading_method: str = "pdf", **kwargs) -> str:
        """
        加载文档内容
        
        参数:
            file_path: 文件路径
            loading_method: 加载方法
            **kwargs: 其他参数
            
        返回:
            文档内容
        """
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.pdf':
                return self.load_pdf(file_path, loading_method, **kwargs)
            elif file_ext in ['.v', '.sp', '.spice']:
                return self.load_netlist(file_path, loading_method)
            elif file_ext == '.lef':
                return self.load_lef(file_path)
            elif file_ext == '.lib':
                return self.load_lib(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
                
        except Exception as e:
            logger.error(f"Error loading document: {str(e)}")
            raise
    
    def load_pdf(self, file_path: str, method: str, strategy: str = None, chunking_strategy: str = None, chunking_options: dict = None) -> str:
        """
        加载PDF文档的主方法，支持多种加载策略。

        参数:
            file_path (str): PDF文件路径
            method (str): 加载方法，支持 'pymupdf', 'pypdf', 'pdfplumber', 'unstructured'
            strategy (str, optional): 使用unstructured方法时的策略，可选 'fast', 'hi_res', 'ocr_only'
            chunking_strategy (str, optional): 文本分块策略，可选 'basic', 'by_title'
            chunking_options (dict, optional): 分块选项配置

        返回:
            str: 提取的文本内容
        """
        try:
            if method == "pymupdf":
                return self._load_with_pymupdf(file_path)
            elif method == "pypdf":
                return self._load_with_pypdf(file_path)
            elif method == "pdfplumber":
                return self._load_with_pdfplumber(file_path)
            elif method == "unstructured":
                return self._load_with_unstructured(
                    file_path, 
                    strategy=strategy,
                    chunking_strategy=chunking_strategy,
                    chunking_options=chunking_options
                )
            else:
                raise ValueError(f"Unsupported loading method: {method}")
        except Exception as e:
            logger.error(f"Error loading PDF with {method}: {str(e)}")
            raise
    
    def get_total_pages(self) -> int:
        """
        获取当前加载文档的总页数。

        返回:
            int: 文档总页数
        """
        return max(page_data['page'] for page_data in self.current_page_map) if self.current_page_map else 0
    
    def get_page_map(self) -> list:
        """
        获取当前文档的页面映射信息。

        返回:
            list: 包含每页文本内容和页码的列表
        """
        return self.current_page_map
    
    def _load_with_pymupdf(self, file_path: str) -> str:
        """
        使用PyMuPDF库加载PDF文档。
        适合快速处理大量PDF文件，性能最佳。

        参数:
            file_path (str): PDF文件路径

        返回:
            str: 提取的文本内容
        """
        text_blocks = []
        try:
            with fitz.open(file_path) as doc:
                self.total_pages = len(doc)
                for page_num, page in enumerate(doc, 1):
                    text = page.get_text("text")
                    if text.strip():
                        text_blocks.append({
                            "text": text.strip(),
                            "page": page_num
                        })
            self.current_page_map = text_blocks
            return "\n".join(block["text"] for block in text_blocks)
        except Exception as e:
            logger.error(f"PyMuPDF error: {str(e)}")
            raise
    
    def _load_with_pypdf(self, file_path: str) -> str:
        """
        使用PyPDF库加载PDF文档。
        适合简单的PDF文本提取，依赖较少。

        参数:
            file_path (str): PDF文件路径

        返回:
            str: 提取的文本内容
        """
        try:
            text_blocks = []
            with open(file_path, "rb") as file:
                pdf = PdfReader(file)
                self.total_pages = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text_blocks.append({
                            "text": page_text.strip(),
                            "page": page_num
                        })
            self.current_page_map = text_blocks
            return "\n".join(block["text"] for block in text_blocks)
        except Exception as e:
            logger.error(f"PyPDF error: {str(e)}")
            raise
    
    def _load_with_unstructured(self, file_path: str, strategy: str = "fast", chunking_strategy: str = "basic", chunking_options: dict = None) -> str:
        """
        使用unstructured库加载PDF文档。
        适合需要更好的文档结构识别和灵活分块策略的场景。

        参数:
            file_path (str): PDF文件路径
            strategy (str): 加载策略，默认'fast'
            chunking_strategy (str): 分块策略，默认'basic'
            chunking_options (dict): 分块选项配置

        返回:
            str: 提取的文本内容
        """
        try:
            strategy_params = {
                "fast": {"strategy": "fast"},
                "hi_res": {"strategy": "hi_res"},
                "ocr_only": {"strategy": "ocr_only"}
            }            
         
            # Prepare chunking parameters based on strategy
            chunking_params = {}
            if chunking_strategy == "basic":
                chunking_params = {
                    "max_characters": chunking_options.get("maxCharacters", 4000),
                    "new_after_n_chars": chunking_options.get("newAfterNChars", 3000),
                    "combine_text_under_n_chars": chunking_options.get("combineTextUnderNChars", 2000),
                    "overlap": chunking_options.get("overlap", 200),
                    "overlap_all": chunking_options.get("overlapAll", False)
                }
            elif chunking_strategy == "by_title":
                chunking_params = {
                    "chunking_strategy": "by_title",
                    "combine_text_under_n_chars": chunking_options.get("combineTextUnderNChars", 2000),
                    "multipage_sections": chunking_options.get("multiPageSections", False)
                }
            
            # Combine strategy parameters with chunking parameters
            params = {**strategy_params.get(strategy, {"strategy": "fast"}), **chunking_params}
            
            elements = partition_pdf(file_path, **params)
            
            # Add debug logging
            for elem in elements:
                logger.debug(f"Element type: {type(elem)}")
                logger.debug(f"Element content: {str(elem)}")
                logger.debug(f"Element dir: {dir(elem)}")
            
            text_blocks = []
            pages = set()
            
            for elem in elements:
                metadata = elem.metadata.__dict__
                page_number = metadata.get('page_number')
                
                if page_number is not None:
                    pages.add(page_number)
                    
                    # Convert element to a serializable format
                    cleaned_metadata = {}
                    for key, value in metadata.items():
                        if key == '_known_field_names':
                            continue
                        
                        try:
                            # Try JSON serialization to test if value is serializable
                            json.dumps({key: value})
                            cleaned_metadata[key] = value
                        except (TypeError, OverflowError):
                            # If not serializable, convert to string
                            cleaned_metadata[key] = str(value)
                    
                    # Add additional element information
                    cleaned_metadata['element_type'] = elem.__class__.__name__
                    cleaned_metadata['id'] = str(getattr(elem, 'id', None))
                    cleaned_metadata['category'] = str(getattr(elem, 'category', None))
                    
                    text_blocks.append({
                        "text": str(elem),
                        "page": page_number,
                        "metadata": cleaned_metadata
                    })
            
            self.total_pages = max(pages) if pages else 0
            self.current_page_map = text_blocks
            return "\n".join(block["text"] for block in text_blocks)
            
        except Exception as e:
            logger.error(f"Unstructured error: {str(e)}")
            raise
    
    def _load_with_pdfplumber(self, file_path: str) -> str:
        """
        使用pdfplumber库加载PDF文档。
        适合需要处理表格或需要文本位置信息的场景。

        参数:
            file_path (str): PDF文件路径

        返回:
            str: 提取的文本内容
        """
        text_blocks = []
        try:
            with pdfplumber.open(file_path) as pdf:
                self.total_pages = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text_blocks.append({
                            "text": page_text.strip(),
                            "page": page_num
                        })
            self.current_page_map = text_blocks
            return "\n".join(block["text"] for block in text_blocks)
        except Exception as e:
            logger.error(f"pdfplumber error: {str(e)}")
            raise
    
    def save_document(self, filename: str, chunks: list, metadata: dict, loading_method: str, strategy: str = None, chunking_strategy: str = None) -> str:
        """
        保存处理后的文档数据。

        参数:
            filename (str): 原文件名
            chunks (list): 文档分块列表
            metadata (dict): 文档元数据
            loading_method (str): 使用的加载方法
            strategy (str, optional): 使用的加载策略
            chunking_strategy (str, optional): 使用的分块策略

        返回:
            str: 保存的文件路径
        """
        try:
            # 构建文档数据结构
            document_data = {
                "metadata": {
                    "filename": str(filename),
                    "file_type": os.path.splitext(filename)[1].lower()[1:],
                    "loading_method": str(loading_method),
                    "parsing_method": str(loading_method),
                    "strategy": str(strategy) if strategy else None,
                    "chunking_strategy": str(chunking_strategy) if chunking_strategy else None,
                    "timestamp": datetime.now().isoformat(),
                    "total_pages": int(metadata.get("total_pages", 1)),
                    "total_chunks": int(len(chunks))
                },
                "content": chunks
            }
            
            # 使用与 main.py 一致的路径处理方式
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            parsed_files_dir = os.path.join(base_dir, "01-loaded-docs")
            os.makedirs(parsed_files_dir, exist_ok=True)
            
            # 使用原始文件名（不含扩展名）作为保存文件名
            base_name = os.path.splitext(filename)[0]
            filepath = os.path.join(parsed_files_dir, f"{base_name}.json")
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(document_data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"文档已保存到: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"保存文档时出错: {str(e)}")
            raise

    def load_netlist(self, file_path: str, loading_method: str = "verilog") -> str:
        """
        加载 Netlist 文件
        
        参数:
            file_path: 文件路径
            loading_method: 加载方法，支持 "verilog" 或 "spice"
            
        返回:
            Netlist 内容
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 根据不同的加载方法处理内容
            if loading_method == "verilog":
                # Verilog 文件处理
                modules = []
                current_module = []
                
                for line in content.split('\n'):
                    line = line.strip()
                    if not line or line.startswith('//'):  # 跳过空行和注释
                        continue
                        
                    if line.startswith('module'):
                        if current_module:
                            modules.append('\n'.join(current_module))
                        current_module = [line]
                    else:
                        current_module.append(line)
                        
                if current_module:
                    modules.append('\n'.join(current_module))
                    
                # 创建页面映射
                self.current_page_map = [
                    {
                        'page': i + 1,
                        'text': module,
                        'metadata': {
                            'type': 'module',
                            'name': module.split()[1] if len(module.split()) > 1 else f'Module_{i+1}'
                        }
                    }
                    for i, module in enumerate(modules)
                ]
                
            elif loading_method == "spice":
                # SPICE 文件处理
                subcircuits = []
                current_subcircuit = []
                
                for line in content.split('\n'):
                    line = line.strip()
                    if not line or line.startswith('*'):  # 跳过空行和注释
                        continue
                        
                    if line.startswith('.SUBCKT'):
                        if current_subcircuit:
                            subcircuits.append('\n'.join(current_subcircuit))
                        current_subcircuit = [line]
                    else:
                        current_subcircuit.append(line)
                        
                if current_subcircuit:
                    subcircuits.append('\n'.join(current_subcircuit))
                    
                # 创建页面映射
                self.current_page_map = [
                    {
                        'page': i + 1,
                        'text': subcircuit,
                        'metadata': {
                            'type': 'subcircuit',
                            'name': subcircuit.split()[1] if len(subcircuit.split()) > 1 else f'Subcircuit_{i+1}'
                        }
                    }
                    for i, subcircuit in enumerate(subcircuits)
                ]
            else:
                raise ValueError(f"Unsupported netlist loading method: {loading_method}")
                
            self.total_pages = len(self.current_page_map)
            return content
            
        except Exception as e:
            logger.error(f"Error loading netlist: {str(e)}")
            raise
            
    def load_lef(self, file_path: str) -> str:
        """
        加载 LEF 文件
        
        参数:
            file_path: 文件路径
            
        返回:
            LEF 内容
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 将 LEF 内容按层和单元分块
            sections = []
            current_section = []
            in_section = False
            
            for line in content.split('\n'):
                line = line.strip()
                if not line or line.startswith('#'):  # 跳过空行和注释
                    continue
                    
                if line.startswith('LAYER') or line.startswith('MACRO'):
                    if current_section:
                        sections.append('\n'.join(current_section))
                    current_section = [line]
                    in_section = True
                elif line.startswith('END'):
                    if in_section:
                        current_section.append(line)
                        sections.append('\n'.join(current_section))
                        current_section = []
                        in_section = False
                elif in_section:
                    current_section.append(line)
                    
            if current_section:
                sections.append('\n'.join(current_section))
                
            # 创建页面映射
            self.current_page_map = [
                {
                    'page': i + 1,
                    'text': section,
                    'metadata': {
                        'type': 'section',
                        'name': section.split()[1] if len(section.split()) > 1 else f'Section_{i+1}'
                    }
                }
                for i, section in enumerate(sections)
            ]
            
            self.total_pages = len(self.current_page_map)
            return content
            
        except Exception as e:
            logger.error(f"Error loading LEF: {str(e)}")
            raise
            
    def load_lib(self, file_path: str) -> str:
        """
        加载 LIB 文件
        
        参数:
            file_path: 文件路径
            
        返回:
            LIB 内容
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 将 LIB 内容按单元分块
            cells = []
            current_cell = []
            in_cell = False
            
            for line in content.split('\n'):
                line = line.strip()
                if not line or line.startswith('#'):  # 跳过空行和注释
                    continue
                    
                if line.startswith('cell'):
                    if current_cell:
                        cells.append('\n'.join(current_cell))
                    current_cell = [line]
                    in_cell = True
                elif line.startswith('end'):
                    if in_cell:
                        current_cell.append(line)
                        cells.append('\n'.join(current_cell))
                        current_cell = []
                        in_cell = False
                elif in_cell:
                    current_cell.append(line)
                    
            if current_cell:
                cells.append('\n'.join(current_cell))
                
            # 创建页面映射
            self.current_page_map = [
                {
                    'page': i + 1,
                    'text': cell,
                    'metadata': {
                        'type': 'cell',
                        'name': cell.split()[1] if len(cell.split()) > 1 else f'Cell_{i+1}'
                    }
                }
                for i, cell in enumerate(cells)
            ]
            
            self.total_pages = len(self.current_page_map)
            return content
            
        except Exception as e:
            logger.error(f"Error loading LIB: {str(e)}")
            raise

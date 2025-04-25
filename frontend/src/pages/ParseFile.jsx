import React, { useState, useEffect } from 'react';
import RandomImage from '../components/RandomImage';
import { apiBaseUrl } from '../config/config';

const ParseFile = () => {
  const [file, setFile] = useState(null);
  const [loadingMethod, setLoadingMethod] = useState('pymupdf');
  const [parsingOption, setParsingOption] = useState('all_text');
  const [parsedContent, setParsedContent] = useState(null);
  const [status, setStatus] = useState('');
  const [docName, setDocName] = useState('');
  const [isProcessed, setIsProcessed] = useState(false);
  const [fileType, setFileType] = useState('pdf');

  // 初始化加载工具选项
  const [loadingTools, setLoadingTools] = useState([]);
  const [parsingOptions, setParsingOptions] = useState([]);

  // 当文件类型改变时更新加载工具和解析选项
  useEffect(() => {
    const tools = getLoadingTools();
    const options = getParsingOptions();
    setLoadingTools(tools);
    setParsingOptions(options);
    
    // 如果当前选择的加载工具不在新选项中，则选择第一个可用的工具
    if (tools.length > 0 && !tools.find(t => t.value === loadingMethod)) {
      setLoadingMethod(tools[0].value);
    }
    
    // 如果当前选择的解析选项不在新选项中，则选择第一个可用的选项
    if (options.length > 0 && !options.find(o => o.value === parsingOption)) {
      setParsingOption(options[0].value);
    }
  }, [fileType]);

  const getLoadingTools = () => {
    switch (fileType) {
      case 'pdf':
        return [
          { value: 'pymupdf', label: 'PyMuPDF' },
          { value: 'pypdf', label: 'PyPDF' },
          { value: 'unstructured', label: 'Unstructured' },
          { value: 'pdfplumber', label: 'PDF Plumber' }
        ];
      case 'netlist':
        return [
          { value: 'verilog', label: 'Verilog Parser' },
          { value: 'spice', label: 'SPICE Parser' }
        ];
      case 'lef':
        return [
          { value: 'lef', label: 'LEF Parser' }
        ];
      case 'lib':
        return [
          { value: 'lib', label: 'LIB Parser' }
        ];
      default:
        return [];
    }
  };

  const getParsingOptions = () => {
    switch (fileType) {
      case 'pdf':
        return [
          { value: 'all_text', label: '全部文本' },
          { value: 'by_pages', label: '按页面' },
          { value: 'by_titles', label: '按标题' },
          { value: 'text_and_tables', label: '文本和表格' }
        ];
      case 'netlist':
        return [
          { value: 'all_text', label: '全部文本' },
          { value: 'by_modules', label: '按模块' },
          { value: 'by_ports', label: '按端口' },
          { value: 'by_instances', label: '按实例' },
          { value: 'by_pins', label: '按引脚' },
          { value: 'by_nets', label: '按网络' }
        ];
      case 'lef':
        return [
          { value: 'all_text', label: '全部文本' },
          { value: 'by_layers', label: '按层' },
          { value: 'by_macros', label: '按宏单元' }
        ];
      case 'lib':
        return [
          { value: 'all_text', label: '全部文本' },
          { value: 'by_cells', label: '按单元' },
          { value: 'by_pins', label: '按引脚' }
        ];
      default:
        return [];
    }
  };

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setFile(file);
      const baseName = file.name.replace(/\.[^/.]+$/, '');
      setDocName(baseName);
      
      // 根据文件扩展名设置文件类型
      const ext = file.name.split('.').pop().toLowerCase();
      if (ext === 'pdf') {
        setFileType('pdf');
        setLoadingMethod('pymupdf');
      } else if (['v', 'sp', 'spice'].includes(ext)) {
        setFileType('netlist');
        setLoadingMethod('verilog');
      } else if (ext === 'lef') {
        setFileType('lef');
        setLoadingMethod('lef');
      } else if (ext === 'lib') {
        setFileType('lib');
        setLoadingMethod('lib');
      }
    }
  };

  const handleProcess = async () => {
    if (!file || !loadingMethod || !parsingOption) {
      setStatus('请选择所有必需的选项');
      return;
    }

    setStatus('处理中...');
    setParsedContent(null);
    setIsProcessed(false);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('loading_method', loadingMethod);
      formData.append('parsing_option', parsingOption);
      formData.append('file_type', fileType);

      const response = await fetch(`${apiBaseUrl}/parse`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setParsedContent(data.parsed_content);
      setStatus('处理完成！');
      setIsProcessed(true);
    } catch (error) {
      console.error('Error:', error);
      setStatus(`错误: ${error.message}`);
      setIsProcessed(false);
    }
  };

  const renderContent = (item) => {
    switch (item.type) {
      case 'instance':
        return (
          <div className="p-4 border rounded-lg bg-white shadow-sm">
            <h3 className="text-lg font-semibold mb-2">实例: {item.name}</h3>
            <p className="text-sm text-gray-600">类型: {item.instance_type}</p>
            <p className="text-sm text-gray-600">引脚: {item.pins.join(', ')}</p>
            <p className="text-sm text-gray-500 mt-2">原始内容: {item.content}</p>
          </div>
        );
      case 'pin':
        return (
          <div className="p-4 border rounded-lg bg-white shadow-sm">
            <h3 className="text-lg font-semibold mb-2">引脚: {item.name}</h3>
            <p className="text-sm text-gray-600">类型: {item.pin_type}</p>
            <p className="text-sm text-gray-600">连接: {item.connections.join(', ')}</p>
            <p className="text-sm text-gray-500 mt-2">原始内容: {item.content}</p>
          </div>
        );
      case 'net':
        return (
          <div className="p-4 border rounded-lg bg-white shadow-sm">
            <h3 className="text-lg font-semibold mb-2">网络: {item.name}</h3>
            <p className="text-sm text-gray-600">连接数: {item.connections.length}</p>
            <p className="text-sm text-gray-500 mt-2">原始内容: {item.content}</p>
          </div>
        );
      default:
        return (
          <div className="p-4 border rounded-lg bg-white shadow-sm">
            <h3 className="text-lg font-semibold mb-2">{item.type}</h3>
            <p className="text-sm text-gray-500">{item.content}</p>
          </div>
        );
    }
  };

  return (
    <div className="p-6">
      <h2 className="text-2xl font-bold mb-6">解析文件</h2>
      
      <div className="grid grid-cols-12 gap-6">
        {/* 左侧面板 (3/12) */}
        <div className="col-span-3 space-y-4">
          <div className="p-4 border rounded-lg bg-white shadow-sm">
            <div>
              <label className="block text-sm font-medium mb-1">上传文件</label>
              <input
                type="file"
                accept=".pdf,.v,.sp,.spice,.lef,.lib"
                onChange={handleFileSelect}
                className="block w-full border rounded px-3 py-2"
                required
              />
            </div>

            <div className="mt-4">
              <label className="block text-sm font-medium mb-1">加载工具</label>
              <select
                value={loadingMethod}
                onChange={(e) => setLoadingMethod(e.target.value)}
                className="block w-full p-2 border rounded"
                disabled={isProcessed}
              >
                {loadingTools.map(tool => (
                  <option key={tool.value} value={tool.value}>
                    {tool.label}
                  </option>
                ))}
              </select>
            </div>

            <div className="mt-4">
              <label className="block text-sm font-medium mb-1">解析选项</label>
              <select
                value={parsingOption}
                onChange={(e) => setParsingOption(e.target.value)}
                className="block w-full p-2 border rounded"
                disabled={isProcessed}
              >
                {parsingOptions.map(option => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>

            <div className="mt-4">
              <button 
                onClick={handleProcess}
                className={`w-full px-4 py-2 text-white rounded ${
                  isProcessed ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-500 hover:bg-blue-600'
                }`}
                disabled={!file || isProcessed}
              >
                {isProcessed ? '处理完成' : '处理文件'}
              </button>
            </div>

            {status && (
              <div className="mt-4 p-2 text-sm rounded">
                <p className={isProcessed ? 'text-green-600' : 'text-blue-600'}>
                  {status}
                </p>
              </div>
            )}
          </div>
        </div>

        {/* 右侧面板 (9/12) */}
        <div className="col-span-9 border rounded-lg bg-white shadow-sm">
          {parsedContent ? (
            <div className="p-4">
              <h3 className="text-xl font-semibold mb-4">解析结果</h3>
              <div className="mb-4 p-3 border rounded bg-gray-100">
                <h4 className="font-medium mb-2">文档信息</h4>
                <div className="text-sm text-gray-600">
                  <p>总页数: {parsedContent.metadata?.total_pages}</p>
                  <p>解析方法: {parsedContent.metadata?.parsing_method}</p>
                  <p>时间戳: {parsedContent.metadata?.timestamp && new Date(parsedContent.metadata.timestamp).toLocaleString()}</p>
                </div>
              </div>
              <div className="space-y-3 max-h-[calc(100vh-300px)] overflow-y-auto">
                {parsedContent.content.map((item, idx) => (
                  <div key={idx}>
                    {renderContent(item)}
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <RandomImage message="上传并解析文件以查看结果" />
          )}
        </div>
      </div>
    </div>
  );
};

export default ParseFile; 
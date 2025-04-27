import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import RandomImage from '../components/RandomImage';
import { apiBaseUrl } from '../config/config';
import { message } from 'antd';

// 添加文件类型定义
const SUPPORTED_FILE_TYPES = {
  // 文本文件
  text: ['.txt', '.md', '.json', '.csv', '.log'],
  // 芯片设计文件
  chip: ['.v', '.sv', '.vh', '.svh', '.lef', '.lib', '.def', '.sdc', '.spef', '.gds', '.gdsii'],
  // 文档文件
  document: ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx'],
  // 配置文件
  config: ['.yaml', '.yml', '.xml', '.ini', '.conf']
};

// 添加状态常量
const FILE_STATUS = {
  UPLOADING: 'uploading',
  SUCCESS: 'success',
  ERROR: 'error',
  ACTIVE: 'active'
};

// 添加存储工具函数
const storage = {
  setItem: (key, value) => {
    try {
      // 限制存储大小
      const maxSize = 4 * 1024 * 1024; // 4MB
      const valueStr = JSON.stringify(value);
      if (valueStr.length > maxSize) {
        console.warn(`Storage quota exceeded for key: ${key}`);
        return false;
      }
      localStorage.setItem(key, valueStr);
      return true;
    } catch (error) {
      console.error('Storage error:', error);
      return false;
    }
  },
  
  getItem: (key) => {
    try {
      const value = localStorage.getItem(key);
      return value ? JSON.parse(value) : null;
    } catch (error) {
      console.error('Storage error:', error);
      return null;
    }
  },
  
  removeItem: (key) => {
    try {
      localStorage.removeItem(key);
      return true;
    } catch (error) {
      console.error('Storage error:', error);
      return false;
    }
  }
};

const Generation = () => {
  const location = useLocation();
  
  // 状态变量
  const [provider, setProvider] = useState('');
  const [modelName, setModelName] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [models, setModels] = useState({});
  const [isGenerating, setIsGenerating] = useState(false);
  const [response, setResponse] = useState('');
  const [status, setStatus] = useState('');
  const [query, setQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [selectedFile, setSelectedFile] = useState('');
  const [searchFiles, setSearchFiles] = useState([]);
  const [showReasoning, setShowReasoning] = useState(true);
  const [generationMode, setGenerationMode] = useState('search');
  const [collections, setCollections] = useState([]);
  const [selectedCollection, setSelectedCollection] = useState('');
  const [showRequestCommand, setShowRequestCommand] = useState(false);
  const [selectedContextFiles, setSelectedContextFiles] = useState([]);
  const [contextContents, setContextContents] = useState([]);
  
  // 修改状态初始化
  const [uploadedFiles, setUploadedFiles] = useState(() => {
    // 只从 localStorage 恢复小文件的状态信息
    const savedFiles = storage.getItem('uploadedFiles');
    if (savedFiles) {
      return savedFiles.filter(file => file.size < 2 * 1024 * 1024); // 只恢复小于2MB的文件
    }
    return [];
  });

  const [additionalContext, setAdditionalContext] = useState(() => {
    // 只从 localStorage 恢复小文件的内容
    const savedContext = storage.getItem('additionalContext');
    return savedContext || '';
  });

  // 添加文件列表状态
  const [fileList, setFileList] = useState([]);
  const [showFileManager, setShowFileManager] = useState(false);

  // 添加获取文件列表的函数
  const fetchFileList = async () => {
    try {
      console.log('获取文件列表...');
      const response = await fetch(`${apiBaseUrl}/files`);
      if (!response.ok) {
        throw new Error('获取文件列表失败');
      }
      const data = await response.json();
      console.log('文件列表:', data);
      setFileList(data.files || []);
    } catch (error) {
      console.error('获取文件列表错误:', error);
      setStatus(`获取文件列表失败: ${error.message}`);
    }
  };

  // 在组件加载时获取文件列表
  useEffect(() => {
    fetchFileList();
  }, []);

  // 在文件上传成功后刷新文件列表
  const handleFileUpload = async (event) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    for (const file of files) {
      try {
        const formData = new FormData();
        formData.append('file', file);

        console.log('开始上传文件:', {
          name: file.name,
          size: file.size,
          type: file.type
        });

        const response = await fetch(`${apiBaseUrl}/upload-context`, {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`上传失败: ${response.status}`);
        }

        const data = await response.json();
        console.log('上传成功:', data);
        
        // 刷新文件列表
        await fetchFileList();
        
        setStatus('文件上传成功');
      } catch (error) {
        console.error('文件上传错误:', error);
        setStatus(`文件上传失败: ${error.message}`);
      }
    }
  };

  // 修改文件删除函数
  const removeFile = async (fileId) => {
    try {
      console.log('开始删除文件:', fileId);
      
      // 先从列表中移除
      setUploadedFiles(prev => prev.filter(file => file.id !== fileId));
      
      const response = await fetch(`${apiBaseUrl}/files/${fileId}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('删除失败响应:', {
          status: response.status,
          statusText: response.statusText,
          errorText: errorText
        });
        
        let errorMessage;
        try {
          const errorData = JSON.parse(errorText);
          errorMessage = errorData.detail || `删除失败: ${response.status} ${response.statusText}`;
        } catch (e) {
          errorMessage = `删除失败: ${response.status} ${response.statusText}`;
        }
        
        // 如果删除失败，恢复文件到列表
        setUploadedFiles(prev => [...prev, {
          id: fileId,
          name: fileId,
          size: 0,
          status: FILE_STATUS.ERROR,
          error: errorMessage,
          uploadTime: new Date().toISOString()
        }]);
        
        throw new Error(errorMessage);
      }

      message.success('文件删除成功');
      
      // 刷新文件列表
      await fetchFileList();
    } catch (error) {
      console.error('删除文件错误:', error);
      message.error(error.message || '删除文件失败');
    }
  };

  // 修改清理函数
  const clearContext = () => {
    setUploadedFiles([]);
    setAdditionalContext('');
    storage.removeItem('uploadedFiles');
    storage.removeItem('additionalContext');
  };

  // 在组件卸载时清理
  useEffect(() => {
    return () => {
      // 可以选择是否在组件卸载时清理状态
      // clearContext();
    };
  }, []);

  // 加载可用模型列表、搜索结果文件列表和集合列表
  useEffect(() => {
    const fetchData = async () => {
      try {
        // 获取模型列表
        const modelsResponse = await fetch(`${apiBaseUrl}/generation/models`);
        const modelsData = await modelsResponse.json();
        setModels(modelsData.models);

        // 获取搜索结果文件列表
        const filesResponse = await fetch(`${apiBaseUrl}/search-results`);
        const filesData = await filesResponse.json();
        setSearchFiles(filesData.files);

        // 获取集合列表
        const collectionsResponse = await fetch(`${apiBaseUrl}/collections`);
        const collectionsData = await collectionsResponse.json();
        setCollections(Array.isArray(collectionsData.collections) ? collectionsData.collections : []);
      } catch (error) {
        console.error('Error fetching data:', error);
        setStatus('获取数据失败');
      }
    };

    fetchData();
  }, []);

  // 加载选中的搜索结果文件内容
  useEffect(() => {
    const loadSearchResults = async () => {
      if (!selectedFile) {
        setQuery('');
        setSearchResults([]);
        return;
      }

      try {
        const response = await fetch(`${apiBaseUrl}/search-results/${selectedFile}`);
        const data = await response.json();
        setQuery(data.query);
        setSearchResults(data.results);
      } catch (error) {
        console.error('Error loading search results:', error);
        setStatus('加载搜索结果失败');
      }
    };

    loadSearchResults();
  }, [selectedFile]);

  // 如果从搜索页面跳转过来，获取搜索结果
  useEffect(() => {
    if (location.state) {
      const { query: searchQuery, results } = location.state;
      if (searchQuery) setQuery(searchQuery);
      if (results) setSearchResults(results);
    }
  }, [location]);

  // 修改文件选择处理函数
  const handleFileSelection = async (fileId) => {
    try {
      console.log('开始获取文件内容:', fileId);
      const response = await fetch(`${apiBaseUrl}/files/${fileId}`);
      if (!response.ok) {
        const errorText = await response.text();
        console.error('获取文件内容失败:', {
          status: response.status,
          statusText: response.statusText,
          errorText: errorText
        });
        throw new Error(`获取文件内容失败: ${response.status} ${response.statusText}`);
      }
      
      const data = await response.json();
      console.log('获取到的文件内容:', {
        fileId: data.file_id,
        name: data.name,
        size: data.size,
        status: data.status,
        encoding: data.encoding
      });
      
      if (data.status === 'empty') {
        throw new Error(`文件 ${data.name} 内容为空`);
      }
      
      if (!data.content) {
        throw new Error(`文件 ${data.name} 内容读取失败`);
      }
      
      // 更新选中的文件和内容
      setSelectedContextFiles(prev => [...prev, fileId]);
      setContextContents(prev => [...prev, data.content]);
      
      console.log('更新后的contextContents:', [...contextContents, data.content]);
    } catch (error) {
      console.error(`读取文件 ${fileId} 失败:`, error);
      setStatus(`读取文件失败: ${error.message}`);
      // 如果读取失败，取消选择
      setSelectedContextFiles(prev => prev.filter(id => id !== fileId));
    }
  };

  // 修改文件取消选择处理函数
  const handleFileDeselection = (fileId) => {
    const index = selectedContextFiles.indexOf(fileId);
    if (index !== -1) {
      setSelectedContextFiles(prev => prev.filter(id => id !== fileId));
      setContextContents(prev => prev.filter((_, i) => i !== index));
    }
  };

  // 修改生成函数
  const handleGenerate = async () => {
    if (!provider || !modelName) {
      setStatus('请选择生成模型');
      return;
    }

    if (!query) {
      setStatus('请输入问题');
      return;
    }

    if (generationMode === 'search' && searchResults.length === 0) {
      setStatus('请确保有搜索结果');
      return;
    }

    if (generationMode === 'collection' && !selectedCollection) {
      setStatus('请选择集合');
      return;
    }

    setIsGenerating(true);
    setStatus('');
    try {
      const requestBody = {
        provider,
        model_name: modelName,
        query,
        api_key: apiKey || undefined,
        show_reasoning: showReasoning,
        context_file_ids: selectedContextFiles,
        context_contents: contextContents
      };

      // 根据生成方式添加不同的参数
      if (generationMode === 'search') {
        requestBody.search_results = searchResults;
      } else {
        requestBody.collection_name = selectedCollection;
      }

      // 添加详细的请求日志
      console.log('Generation Mode:', generationMode);
      console.log('Selected Collection:', selectedCollection);
      console.log('Request Body:', JSON.stringify(requestBody, null, 2));

      const response = await fetch(`${apiBaseUrl}/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Error Response:', {
          status: response.status,
          statusText: response.statusText,
          errorText: errorText
        });
        
        let errorMessage;
        try {
          const errorData = JSON.parse(errorText);
          errorMessage = errorData.detail || `生成失败: ${response.status} ${response.statusText}`;
        } catch (e) {
          errorMessage = `生成失败: ${response.status} ${response.statusText}`;
        }
        
        throw new Error(errorMessage);
      }

      const data = await response.json();
      console.log('Success Response:', data);
      
      if (data.data && data.data.response) {
        setResponse(data.data.response);
        setStatus('生成完成！');
      } else {
        console.error('Unexpected response format:', data);
        setStatus('生成完成，但返回格式不正确');
      }
    } catch (error) {
      console.error('Generation error:', error);
      setStatus(error.message || '生成失败');
    } finally {
      setIsGenerating(false);
    }
  };

  // 在文件管理对话框打开时刷新文件列表
  useEffect(() => {
    if (showFileManager) {
      fetchFileList();
    }
  }, [showFileManager]);

  return (
    <div className="p-6">
      <h2 className="text-2xl font-bold mb-6">Generation</h2>
      
      <div className="grid grid-cols-12 gap-6">
        {/* Left Panel - Generation Controls */}
        <div className="col-span-4 space-y-4">
          <div className="p-4 border rounded-lg bg-white shadow-sm">
            <div className="space-y-4">
              {/* 生成方式选择 */}
              <div>
                <label className="block text-sm font-medium mb-1">生成方式</label>
                <div className="flex space-x-4">
                  <label className="inline-flex items-center">
                    <input
                      type="radio"
                      value="search"
                      checked={generationMode === 'search'}
                      onChange={(e) => setGenerationMode(e.target.value)}
                      className="form-radio"
                    />
                    <span className="ml-2">从搜索结果生成</span>
                  </label>
                  <label className="inline-flex items-center">
                    <input
                      type="radio"
                      value="collection"
                      checked={generationMode === 'collection'}
                      onChange={(e) => setGenerationMode(e.target.value)}
                      className="form-radio"
                    />
                    <span className="ml-2">直接从集合生成</span>
                  </label>
                </div>
              </div>

              {/* 搜索结果文件选择（仅当选择从搜索结果生成时显示） */}
              {generationMode === 'search' && (
                <div>
                  <label className="block text-sm font-medium mb-1">Search Results File</label>
                  <select
                    value={selectedFile}
                    onChange={(e) => setSelectedFile(e.target.value)}
                    className="block w-full p-2 border rounded"
                  >
                    <option value="">Select search results file...</option>
                    {searchFiles.map(file => (
                      <option key={file.id} value={file.id}>
                        {file.name}
                      </option>
                    ))}
                  </select>
                </div>
              )}

              {/* 集合选择（仅当选择直接从集合生成时显示） */}
              {generationMode === 'collection' && (
                <div>
                  <label className="block text-sm font-medium mb-1">选择集合</label>
                  <select
                    value={selectedCollection}
                    onChange={(e) => setSelectedCollection(e.target.value)}
                    className="block w-full p-2 border rounded"
                  >
                    <option value="">Select collection...</option>
                    {collections.map(collection => (
                      <option key={collection.id || collection} value={collection.id || collection}>
                        {collection.name || collection}
                      </option>
                    ))}
                  </select>
                </div>
              )}

              <div>
                <label className="block text-sm font-medium mb-1">Question</label>
                <textarea
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Enter your question..."
                  className="block w-full p-2 border rounded h-32 resize-none"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">Provider</label>
                <select
                  value={provider}
                  onChange={(e) => setProvider(e.target.value)}
                  className="block w-full p-2 border rounded"
                >
                  <option value="">Select provider...</option>
                  {Object.keys(models).map(p => (
                    <option key={p} value={p}>{p}</option>
                  ))}
                </select>
              </div>

              {provider && (
                <div>
                  <label className="block text-sm font-medium mb-1">Model</label>
                  <select
                    value={modelName}
                    onChange={(e) => setModelName(e.target.value)}
                    className="block w-full p-2 border rounded"
                  >
                    <option value="">Select model...</option>
                    {Object.entries(models[provider] || {}).map(([id, name]) => (
                      <option key={id} value={id}>
                        {id === 'deepseek-v3' ? 'DeepSeek V3' :
                         id === 'deepseek-r1' ? 'DeepSeek R1' :
                         name}
                      </option>
                    ))}
                  </select>
                </div>
              )}

              {(provider === 'openai' || provider === 'deepseek') && (
                <div>
                  <label className="block text-sm font-medium mb-1">API Key</label>
                  <input
                    type="password"
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                    placeholder="Enter your API key..."
                    className="block w-full p-2 border rounded"
                  />
                </div>
              )}

              {provider === 'deepseek' && modelName === 'deepseek-r1' && (
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="showReasoning"
                    checked={showReasoning}
                    onChange={(e) => setShowReasoning(e.target.checked)}
                    className="rounded border-gray-300 text-green-500 focus:ring-green-500"
                  />
                  <label htmlFor="showReasoning" className="text-sm font-medium">
                    显示思维链过程
                  </label>
                </div>
              )}

              {/* 上下文文件选择 */}
              <div>
                <label className="block text-sm font-medium mb-1">选择上下文文件</label>
                <div className="max-h-[200px] overflow-y-auto border rounded p-2">
                  {fileList.map(file => (
                    <div key={file.file_id} className="flex items-center space-x-2 py-1">
                      <input
                        type="checkbox"
                        id={`file-${file.file_id}`}
                        checked={selectedContextFiles.includes(file.file_id)}
                        onChange={async (e) => {
                          if (e.target.checked) {
                            await handleFileSelection(file.file_id);
                          } else {
                            handleFileDeselection(file.file_id);
                          }
                        }}
                        className="rounded border-gray-300 text-green-500 focus:ring-green-500"
                      />
                      <label htmlFor={`file-${file.file_id}`} className="text-sm">
                        {file.name}
                      </label>
                    </div>
                  ))}
                </div>
              </div>

              {/* 显示请求命令的开关 */}
              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="showRequestCommand"
                  checked={showRequestCommand}
                  onChange={(e) => setShowRequestCommand(e.target.checked)}
                  className="rounded border-gray-300 text-green-500 focus:ring-green-500"
                />
                <label htmlFor="showRequestCommand" className="text-sm font-medium">
                  显示发送给LLM的请求命令
                </label>
              </div>

              {/* 补充上下文区域 */}
              <div className="mb-6">
                {/* 文件上传区域 */}
                <div className="bg-white rounded-lg shadow-sm border border-gray-100">
                  {/* 标题栏 */}
                  <div className="px-6 py-4 border-b border-gray-100 bg-gray-50">
                    <h3 className="text-lg font-medium text-gray-800">补充上下文</h3>
                    <p className="mt-1 text-sm text-gray-500">上传文件以提供额外的上下文信息</p>
                  </div>

                  {/* 上传按钮和文件格式说明 */}
                  <div className="px-6 py-4 bg-white">
                    <div className="flex items-center justify-between">
                      <div className="flex-1">
                        <input
                          type="file"
                          multiple
                          onChange={handleFileUpload}
                          className="hidden"
                          id="context-files"
                          accept={Object.values(SUPPORTED_FILE_TYPES).flat().join(',')}
                        />
                        <label
                          htmlFor="context-files"
                          className="inline-flex items-center px-4 py-2 border border-indigo-200 shadow-sm text-sm font-medium rounded-md text-indigo-700 bg-indigo-50 hover:bg-indigo-100 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-colors duration-200"
                        >
                          <svg className="w-5 h-5 mr-2 text-indigo-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4" />
                          </svg>
                          选择文件
                        </label>
                        <span className="ml-3 text-sm text-gray-500">
                          支持多种文件格式
                        </span>
                      </div>
                    </div>

                    {/* 文件格式说明 */}
                    <div className="mt-4 grid grid-cols-2 gap-4">
                      <div className="text-sm p-3 bg-gray-50 rounded-md border border-gray-100">
                        <p className="font-medium text-gray-800">文本文件</p>
                        <p className="text-gray-500">.txt, .md, .json, .csv, .log</p>
                      </div>
                      <div className="text-sm p-3 bg-gray-50 rounded-md border border-gray-100">
                        <p className="font-medium text-gray-800">芯片设计</p>
                        <p className="text-gray-500">.v, .sv, .lef, .lib, .def, .sdc, .spef, .gds</p>
                      </div>
                      <div className="text-sm p-3 bg-gray-50 rounded-md border border-gray-100">
                        <p className="font-medium text-gray-800">文档文件</p>
                        <p className="text-gray-500">.pdf, .doc, .docx, .xls, .xlsx</p>
                      </div>
                      <div className="text-sm p-3 bg-gray-50 rounded-md border border-gray-100">
                        <p className="font-medium text-gray-800">配置文件</p>
                        <p className="text-gray-500">.yaml, .yml, .xml, .ini, .conf</p>
                      </div>
                    </div>
                  </div>

                  {/* 已上传文件列表 */}
                  {uploadedFiles.length > 0 && (
                    <div className="px-6 py-4 border-t border-gray-100 bg-gray-50">
                      <h4 className="text-sm font-medium text-gray-800 mb-3">已上传文件</h4>
                      <div className="space-y-2">
                        {uploadedFiles.map((file, index) => {
                          console.log('渲染文件:', file);
                          return (
                            <div 
                              key={`${file.id}-${index}`}
                              className={`flex items-center justify-between py-2 px-3 bg-white rounded-md border ${
                                file.status === FILE_STATUS.ERROR ? 'border-red-200' :
                                file.status === FILE_STATUS.SUCCESS ? 'border-green-200' :
                                'border-gray-100'
                              } shadow-sm transition-all duration-200`}
                            >
                              <div className="flex items-center min-w-0">
                                <div className="flex-shrink-0">
                                  {file.status === FILE_STATUS.UPLOADING && (
                                    <div className="w-5 h-5">
                                      <svg className="animate-spin h-5 w-5 text-indigo-500" fill="none" viewBox="0 0 24 24">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                                      </svg>
                                    </div>
                                  )}
                                  {file.status === FILE_STATUS.SUCCESS && (
                                    <svg className="h-5 w-5 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                                    </svg>
                                  )}
                                  {file.status === FILE_STATUS.ERROR && (
                                    <svg className="h-5 w-5 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                                    </svg>
                                  )}
                                </div>
                                <div className="ml-3 min-w-0">
                                  <p className="text-sm font-medium text-gray-800 truncate">{file.name}</p>
                                  <p className="text-xs text-gray-500">
                                    {Math.round(file.size / 1024)} KB • {new Date(file.uploadTime).toLocaleString()}
                                  </p>
                                  {file.status === FILE_STATUS.ERROR && (
                                    <p className="text-xs text-red-500 mt-1">{file.error}</p>
                                  )}
                                  {file.status === FILE_STATUS.UPLOADING && (
                                    <p className="text-xs text-indigo-500 mt-1">正在处理...</p>
                                  )}
                                  {file.status === FILE_STATUS.SUCCESS && (
                                    <p className="text-xs text-green-500 mt-1">处理成功</p>
                                  )}
                                </div>
                              </div>
                              <button
                                onClick={() => removeFile(file.id)}
                                className="ml-4 flex-shrink-0 text-gray-400 hover:text-red-500 transition-colors duration-200"
                              >
                                <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                                </svg>
                              </button>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  {/* 上下文预览 */}
                  {additionalContext && (
                    <div className="px-6 py-4 border-t border-gray-100 bg-white">
                      <h4 className="text-sm font-medium text-gray-800 mb-3">上下文预览</h4>
                      <div className="bg-gray-50 rounded-md p-4 max-h-[200px] overflow-y-auto border border-gray-100">
                        <pre className="text-sm text-gray-700 whitespace-pre-wrap font-mono">{additionalContext}</pre>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* 显示请求命令 */}
              {showRequestCommand && (
                <div className="mt-4 p-4 bg-gray-50 rounded-lg">
                  <h4 className="text-sm font-medium mb-2">发送给LLM的请求命令：</h4>
                  <pre className="text-xs bg-gray-100 p-2 rounded overflow-x-auto">
                    {JSON.stringify({
                      provider,
                      model_name: modelName,
                      query,
                      show_reasoning: showReasoning,
                      context_file_ids: selectedContextFiles,
                      context_contents: contextContents.filter(content => content !== null),  // 过滤掉null值
                      ...(generationMode === 'search' ? { search_results: searchResults } : { collection_name: selectedCollection })
                    }, null, 2)}
                  </pre>
                </div>
              )}

              <button
                onClick={handleGenerate}
                disabled={isGenerating}
                className="w-full px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600 disabled:bg-green-300"
              >
                {isGenerating ? 'Generating...' : 'Generate'}
              </button>

              {status && (
                <div className={`p-4 rounded-lg ${
                  status.includes('失败') ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'
                }`}>
                  {status}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right Panel - Context and Response */}
        <div className="col-span-8">
          {isGenerating ? (
            <div className="flex flex-col items-center justify-center h-[400px] bg-white rounded-lg shadow-sm">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-green-500 mb-4"></div>
              <p className="text-gray-600">正在生成回答...</p>
            </div>
          ) : generationMode === 'search' ? (
            selectedFile ? (
              <>
                {/* Search Results Context */}
                <div className="mb-6 p-4 border rounded-lg bg-white shadow-sm hover:shadow-md transition-shadow duration-200">
                  <h3 className="text-xl font-semibold mb-4 text-gray-800">Search Context</h3>
                  <div className="space-y-4 max-h-[300px] overflow-y-auto custom-scrollbar">
                    {searchResults.map((result, idx) => (
                      <div key={idx} className="p-4 border rounded bg-gray-50 hover:bg-gray-100 transition-colors duration-200">
                        <div className="flex justify-between items-start mb-2">
                          <span className="font-medium text-sm text-gray-500">
                            相似度: {(result.score * 100).toFixed(1)}%
                          </span>
                          <div className="text-sm text-gray-500">
                            <div>文档: {result.metadata?.document_name || 'N/A'}</div>
                            <div>页码: {result.metadata?.page_number || 'N/A'}</div>
                            <div>块号: {result.metadata?.chunk_id || 'N/A'}</div>
                          </div>
                        </div>
                        <p className="text-sm whitespace-pre-wrap text-gray-700">{result.content}</p>
                      </div>
                    ))}
                    {searchResults.length === 0 && (
                      <div className="text-gray-500 text-center py-4">
                        No search results available. Please perform a search first.
                      </div>
                    )}
                  </div>
                </div>

                {/* Generated Response */}
                {response && (
                  <div className="p-4 border rounded-lg bg-white shadow-sm hover:shadow-md transition-shadow duration-200">
                    <h3 className="text-xl font-semibold mb-4 text-gray-800">Generated Response</h3>
                    <div className="p-4 border rounded bg-gray-50">
                      <p className="whitespace-pre-wrap text-gray-700 leading-relaxed">{response}</p>
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div className="flex flex-col items-center justify-center h-[400px] bg-white rounded-lg shadow-sm">
                <RandomImage message="Select a search results file to start generation" />
              </div>
            )
          ) : (
            <>
              {/* Collection Generation Response */}
              {response ? (
                <div className="p-4 border rounded-lg bg-white shadow-sm hover:shadow-md transition-shadow duration-200">
                  <h3 className="text-xl font-semibold mb-4 text-gray-800">Generated Response</h3>
                  <div className="p-4 border rounded bg-gray-50">
                    <p className="whitespace-pre-wrap text-gray-700 leading-relaxed">{response}</p>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center h-[400px] bg-white rounded-lg shadow-sm">
                  <RandomImage message="Enter your question and select a collection to start generation" />
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {/* 添加自定义滚动条样式 */}
      <style jsx>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 6px;
          height: 6px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: #f8fafc;
          border-radius: 3px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: #cbd5e1;
          border-radius: 3px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: #94a3b8;
        }
        .custom-scrollbar::-webkit-scrollbar-corner {
          background: #f8fafc;
        }
      `}</style>

      {/* 文件管理按钮 */}
      <button
        onClick={() => setShowFileManager(!showFileManager)}
        className="inline-flex items-center px-4 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
      >
        <svg className="w-5 h-5 mr-2 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
        </svg>
        文件管理
      </button>

      {/* 文件管理对话框 */}
      {showFileManager && (
        <div className="fixed inset-0 bg-gray-500 bg-opacity-75 flex items-center justify-center">
          <div className="bg-white rounded-lg p-6 max-w-4xl w-full max-h-[80vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-medium text-gray-900">文件管理</h3>
              <div className="flex space-x-2">
                <button
                  onClick={fetchFileList}
                  className="px-3 py-1 text-sm text-blue-600 hover:text-blue-800"
                >
                  刷新列表
                </button>
                <button
                  onClick={() => setShowFileManager(false)}
                  className="text-gray-400 hover:text-gray-500"
                >
                  <svg className="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            </div>
            
            <div className="space-y-4">
              {fileList.map(file => (
                <div
                  key={file.file_id}
                  className="flex items-center justify-between p-4 bg-gray-50 rounded-lg"
                >
                  <div className="flex items-center space-x-4">
                    <div className="flex-shrink-0">
                      <svg className="h-8 w-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                    </div>
                    <div>
                      <p className="text-sm font-medium text-gray-900">{file.name}</p>
                      <p className="text-xs text-gray-500">
                        {Math.round(file.size / 1024)} KB • {new Date(file.upload_time).toLocaleString()}
                      </p>
                      <p className="text-xs text-gray-500">
                        使用次数: {file.used_count} • 状态: {file.status}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => removeFile(file.file_id)}
                      className="text-red-600 hover:text-red-800"
                    >
                      <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                      </svg>
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Generation; 
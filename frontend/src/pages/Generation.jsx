import React, { useState, useEffect, useMemo } from 'react';
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
  const [fileList, setFileList] = useState([]);
  const [showFileManager, setShowFileManager] = useState(false);
  const [chunkedFiles, setChunkedFiles] = useState([]);
  const [selectedChunkedFiles, setSelectedChunkedFiles] = useState([]);
  const [showChunkedFileManager, setShowChunkedFileManager] = useState(false);
  const [additionalContext, setAdditionalContext] = useState('');
  
  // 添加新的状态变量
  const [isLoadingFiles, setIsLoadingFiles] = useState(false);
  const [isLoadingChunkedFiles, setIsLoadingChunkedFiles] = useState(false);
  const [fileLoadError, setFileLoadError] = useState(null);
  const [chunkedFileLoadError, setChunkedFileLoadError] = useState(null);
  const [lastFileUpdate, setLastFileUpdate] = useState(null);
  const [lastChunkedFileUpdate, setLastChunkedFileUpdate] = useState(null);

  // 使用 useMemo 优化分块文件列表
  const uniqueChunkedFiles = useMemo(() => {
    return chunkedFiles.reduce((acc, file) => {
      if (acc.some(f => f.id === file.id)) {
        return acc;
      }
      return [...acc, file];
    }, []).sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
  }, [chunkedFiles]);

  // 添加本地存储相关状态
  const [uploadedFiles, setUploadedFiles] = useState(() => {
    try {
      // 只从 localStorage 恢复小文件的状态信息
      const savedFiles = storage.getItem('uploadedFiles');
      if (savedFiles) {
        return savedFiles.filter(file => file.size < 2 * 1024 * 1024); // 只恢复小于2MB的文件
      }
    } catch (error) {
      console.error('Error loading files from storage:', error);
    }
    return [];
  });

  // 添加缓存时间常量
  const CACHE_DURATION = 5 * 60 * 1000; // 5分钟缓存

  // 修改文件列表获取函数
  const fetchFileList = async (force = false) => {
    // 如果数据已缓存且未过期，直接返回
    if (!force && lastFileUpdate && (Date.now() - lastFileUpdate) < CACHE_DURATION && fileList.length > 0) {
      return;
    }

    try {
      setIsLoadingFiles(true);
      setFileLoadError(null);
      console.log('获取文件列表...');
      const response = await fetch(`${apiBaseUrl}/files`);
      if (!response.ok) {
        throw new Error('获取文件列表失败');
      }
      const data = await response.json();
      console.log('文件列表:', data);
      setFileList(data.files || []);
      setLastFileUpdate(Date.now());
    } catch (error) {
      console.error('获取文件列表错误:', error);
      setFileLoadError(error.message);
      // 如果是网络错误，3秒后自动重试
      if (error.name === 'TypeError') {
        setTimeout(() => fetchFileList(true), 3000);
      }
    } finally {
      setIsLoadingFiles(false);
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

  // 修改分块文件列表获取函数
  const fetchChunkedFiles = async (force = false) => {
    // 如果数据已缓存且未过期，直接返回
    if (!force && lastChunkedFileUpdate && (Date.now() - lastChunkedFileUpdate) < CACHE_DURATION && chunkedFiles.length > 0) {
      return;
    }

    try {
      setIsLoadingChunkedFiles(true);
      setChunkedFileLoadError(null);
      console.log('获取分块文件列表...');
      const response = await fetch(`${apiBaseUrl}/chunked-files`);
      if (!response.ok) {
        throw new Error('获取分块文件列表失败');
      }
      const data = await response.json();
      console.log('分块文件列表:', data);
      setChunkedFiles(data.files || []);
      setLastChunkedFileUpdate(Date.now());
    } catch (error) {
      console.error('获取分块文件列表错误:', error);
      setChunkedFileLoadError(error.message);
      // 如果是网络错误，3秒后自动重试
      if (error.name === 'TypeError') {
        setTimeout(() => fetchChunkedFiles(true), 3000);
      }
    } finally {
      setIsLoadingChunkedFiles(false);
    }
  };
  
  // 在组件加载时获取分块文件列表
  useEffect(() => {
    fetchChunkedFiles();
  }, []);
  
  // 优化分块文件选择处理函数
  const handleChunkedFileSelection = async (fileId) => {
    try {
      // 检查是否已经选择
      if (selectedChunkedFiles.includes(fileId)) {
        return;
      }
      
      // 立即更新UI状态，提供即时反馈
      setSelectedChunkedFiles(prev => [...prev, fileId]);
      
      // 异步获取文件内容
      const response = await fetch(`${apiBaseUrl}/chunked-files/${fileId}`);
      if (!response.ok) {
        // 如果获取失败，回滚选择状态
        setSelectedChunkedFiles(prev => prev.filter(id => id !== fileId));
        throw new Error('获取分块文件内容失败');
      }
      
      const data = await response.json();
      setStatus(`已选择分块文件: ${data.document_name}`);
    } catch (error) {
      console.error('选择分块文件错误:', error);
      setStatus(`选择分块文件失败: ${error.message}`);
    }
  };

  // 优化分块文件取消选择处理函数
  const handleChunkedFileDeselection = (fileId) => {
    setSelectedChunkedFiles(prev => prev.filter(id => id !== fileId));
  };

  // 修改组件加载时的文件获取逻辑
  useEffect(() => {
    // 只在显示文件管理器时获取数据
    if (showFileManager) {
      fetchFileList();
    }
    if (showChunkedFileManager) {
      fetchChunkedFiles();
    }
  }, [showFileManager, showChunkedFileManager]);

  // 添加手动刷新按钮的处理函数
  const handleRefreshFiles = () => {
    fetchFileList(true);
  };

  const handleRefreshChunkedFiles = () => {
    fetchChunkedFiles(true);
  };

  // 修改文件管理器渲染函数
  const renderFileManager = () => {
    if (!showFileManager) return null;

    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg p-6 w-3/4 max-h-[80vh] overflow-y-auto">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold">选择文件</h3>
            <div className="flex space-x-2">
              <button
                onClick={handleRefreshFiles}
                className="text-sm bg-blue-100 px-3 py-1 rounded hover:bg-blue-200"
              >
                刷新
              </button>
              <button
                onClick={() => setShowFileManager(false)}
                className="text-gray-500 hover:text-gray-700"
              >
                ✕
              </button>
            </div>
          </div>
          
          {/* 加载状态和错误提示 */}
          {isLoadingFiles && (
            <div className="flex items-center justify-center py-4">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500"></div>
              <span className="ml-2 text-gray-600">加载文件列表中...</span>
            </div>
          )}
          
          {fileLoadError && (
            <div className="bg-red-50 text-red-600 p-3 rounded mb-4 flex items-center justify-between">
              <span>{fileLoadError}</span>
              <button
                onClick={() => fetchFileList(true)}
                className="text-sm bg-red-100 px-3 py-1 rounded hover:bg-red-200"
              >
                重试
              </button>
            </div>
          )}
          
          {/* 文件列表 */}
          <div className="space-y-2">
            {fileList.map((file) => (
              <div
                key={file.id}
                className="flex items-center justify-between p-2 hover:bg-gray-50 rounded"
              >
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={selectedContextFiles.includes(file.id)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        handleFileSelection(file.id);
                      } else {
                        handleFileDeselection(file.id);
                      }
                    }}
                    className="form-checkbox h-4 w-4 text-blue-600"
                  />
                  <span className="text-sm">{file.filename}</span>
                </div>
                <div className="text-xs text-gray-500">
                  {new Date(file.upload_time).toLocaleString()}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  };

  // 修改分块文件管理器渲染函数
  const renderChunkedFileManager = () => {
    if (!showChunkedFileManager) return null;

    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg p-6 w-3/4 max-h-[80vh] overflow-y-auto">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold">选择分块文件</h3>
            <div className="flex space-x-2">
              <button
                onClick={handleRefreshChunkedFiles}
                className="text-sm bg-blue-100 px-3 py-1 rounded hover:bg-blue-200"
              >
                刷新
              </button>
              <button
                onClick={() => setShowChunkedFileManager(false)}
                className="text-gray-500 hover:text-gray-700"
              >
                ✕
              </button>
            </div>
          </div>

          {/* 加载状态和错误提示 */}
          {isLoadingChunkedFiles && (
            <div className="flex items-center justify-center py-4">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-green-500"></div>
              <span className="ml-2 text-gray-600">加载分块文件列表中...</span>
            </div>
          )}
          
          {chunkedFileLoadError && (
            <div className="bg-red-50 text-red-600 p-3 rounded mb-4 flex items-center justify-between">
              <span>{chunkedFileLoadError}</span>
              <button
                onClick={() => fetchChunkedFiles(true)}
                className="text-sm bg-red-100 px-3 py-1 rounded hover:bg-red-200"
              >
                重试
              </button>
            </div>
          )}
          
          {/* 分块文件列表 */}
          <div className="space-y-2">
            {uniqueChunkedFiles.map((file) => (
              <div
                key={file.id}
                className="flex items-center justify-between p-2 hover:bg-gray-50 rounded"
              >
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={selectedChunkedFiles.includes(file.id)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        handleChunkedFileSelection(file.id);
                      } else {
                        handleChunkedFileDeselection(file.id);
                      }
                    }}
                    className="form-checkbox h-4 w-4 text-green-600"
                  />
                  <div>
                    <span className="text-sm">{file.name}</span>
                    <div className="text-xs text-gray-500">
                      {file.total_chunks} 块 • {file.total_pages} 页 • {file.chunking_method}
                    </div>
                  </div>
                </div>
                <div className="text-xs text-gray-500">
                  {new Date(file.timestamp).toLocaleString()}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  };

  // 添加文件选择区域渲染函数
  const renderFileSelection = () => {
    return (
      <div className="mb-4">
        <div className="flex items-center justify-between mb-2">
          <label className="block text-sm font-medium text-gray-700">补充文件</label>
          <div className="flex space-x-2">
            <label className="px-3 py-1 text-sm bg-blue-500 text-white rounded hover:bg-blue-600 cursor-pointer">
              添加新文件
              <input
                type="file"
                multiple
                onChange={handleFileUpload}
                className="hidden"
                accept={Object.values(SUPPORTED_FILE_TYPES).flat().join(',')}
              />
            </label>
            <button
              onClick={() => setShowChunkedFileManager(true)}
              className="px-3 py-1 text-sm bg-green-500 text-white rounded hover:bg-green-600"
            >
              添加分块文件
            </button>
          </div>
        </div>
        
        {/* 显示已选择的文件 */}
        {(selectedContextFiles.length > 0 || selectedChunkedFiles.length > 0) && (
          <div className="mt-2 space-y-2 max-h-[300px] overflow-y-auto">
            {/* 显示已选择的新文件 */}
            {selectedContextFiles.map((fileId) => {
              const file = fileList.find(f => f.id === fileId);
              return file ? (
                <div key={fileId} className="flex items-center justify-between p-2 bg-gray-50 rounded hover:bg-gray-100 transition-colors duration-200">
                  <div className="flex items-center space-x-2 min-w-0 flex-1">
                    <div className="truncate">
                      <span className="text-sm font-medium text-gray-900 truncate">{file.filename}</span>
                      <span className="text-xs text-gray-500 ml-2">(新文件)</span>
                    </div>
                  </div>
                  <button
                    onClick={() => handleFileDeselection(fileId)}
                    className="ml-4 flex-shrink-0 text-red-500 hover:text-red-700"
                  >
                    移除
                  </button>
                </div>
              ) : null;
            })}
            
            {/* 显示已选择的分块文件 */}
            {selectedChunkedFiles.map((fileId) => {
              const file = chunkedFiles.find(f => f.id === fileId);
              return file ? (
                <div key={fileId} className="flex items-center justify-between p-2 bg-gray-50 rounded hover:bg-gray-100 transition-colors duration-200">
                  <div className="flex items-center space-x-2 min-w-0 flex-1">
                    <div className="truncate">
                      <span className="text-sm font-medium text-gray-900 truncate">{file.name}</span>
                      <span className="text-xs text-gray-500 ml-2">(分块文件)</span>
                    </div>
                  </div>
                  <button
                    onClick={() => handleChunkedFileDeselection(fileId)}
                    className="ml-4 flex-shrink-0 text-red-500 hover:text-red-700"
                  >
                    移除
                  </button>
                </div>
              ) : null;
            })}
          </div>
        )}
      </div>
    );
  };

  // 修改生成函数
  const handleGenerate = async () => {
    try {
      setIsGenerating(true);
      setStatus('正在生成...');
      setResponse('');

      // 准备请求数据
      const requestData = {
        provider,
        model_name: modelName,
        query,
        api_key: apiKey,
        show_reasoning: showReasoning,
        ...(generationMode === 'search' && { search_results: searchResults }),
        ...(generationMode === 'collection' && { collection_name: selectedCollection }),
        ...(generationMode !== 'direct' && {
          context_file_ids: selectedContextFiles,
          chunked_file_ids: selectedChunkedFiles,
          context_contents: contextContents,
          additional_context: additionalContext
        })
      };
      
      console.log('生成请求数据:', requestData);
      
      // 使用 fetch 发送 POST 请求
      const response = await fetch(`${apiBaseUrl}/generate/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData)
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // 获取响应的 reader
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let responseText = '';

      // 读取流式响应
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        // 解码并处理数据
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              if (data.type === 'content') {
                responseText += data.content;
                setResponse(responseText);
              } else if (data.type === 'status') {
                setStatus(data.status);
              } else if (data.type === 'error') {
                throw new Error(data.error);
              } else if (data.type === 'done') {
                setIsGenerating(false);
                setStatus('生成完成');
              }
            } catch (error) {
              console.error('处理响应数据出错:', error);
              setStatus(`生成失败: ${error.message}`);
              setIsGenerating(false);
            }
          }
        }
      }
      
    } catch (error) {
      console.error('生成错误:', error);
      setStatus(`生成失败: ${error.message}`);
      setResponse('');
      setIsGenerating(false);
    }
  };

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
                  <label className="inline-flex items-center">
                    <input
                      type="radio"
                      value="direct"
                      checked={generationMode === 'direct'}
                      onChange={(e) => setGenerationMode(e.target.value)}
                      className="form-radio"
                    />
                    <span className="ml-2">直接生成（不使用上下文）</span>
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

              {/* 补充上下文区域（仅当选择从搜索结果或集合生成时显示） */}
              {generationMode !== 'direct' && (
                <div className="mb-4">
                  <div className="flex items-center justify-between mb-2">
                    <label className="block text-sm font-medium text-gray-700">补充上下文</label>
                  </div>
                  <textarea
                    value={additionalContext}
                    onChange={(e) => setAdditionalContext(e.target.value)}
                    className="w-full h-32 p-2 border rounded"
                    placeholder="输入补充的上下文信息..."
                  />
                </div>
              )}

              {/* 文件选择区域（仅当选择从搜索结果或集合生成时显示） */}
              {generationMode !== 'direct' && renderFileSelection()}

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
                      generation_mode: generationMode,
                      ...(generationMode === 'search' && { search_results: searchResults }),
                      ...(generationMode === 'collection' && { collection_name: selectedCollection }),
                      ...(generationMode !== 'direct' && {
                        context_file_ids: selectedContextFiles,
                        context_contents: contextContents.filter(content => content !== null),
                        additional_context: additionalContext
                      })
                    }, null, 2)}
                  </pre>
                </div>
              )}

              <button
                onClick={handleGenerate}
                disabled={isGenerating || !provider || !modelName}
                className="w-full px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600 disabled:bg-green-300"
              >
                {isGenerating ? 'Generating...' : 'Generate'}
              </button>
              {/* 未选择模型时的提示 */}
              {(!provider || !modelName) && (
                <div className="mt-2 p-2 bg-yellow-50 text-yellow-700 rounded text-sm flex items-center">
                  <svg className="h-4 w-4 mr-1 text-yellow-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12c0 4.97-4.03 9-9 9s-9-4.03-9-9 4.03-9 9-9 9 4.03 9 9z" /></svg>
                  请先选择模型提供商和模型
                </div>
              )}

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
          ) : response ? (
            <div className="p-4 border rounded-lg bg-white shadow-sm hover:shadow-md transition-shadow duration-200">
              <h3 className="text-xl font-semibold mb-4 text-gray-800">Generated Response</h3>
              <div className="p-4 border rounded bg-gray-50">
                <p className="whitespace-pre-wrap text-gray-700 leading-relaxed">{response}</p>
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center h-[400px] bg-white rounded-lg shadow-sm">
              <RandomImage message="Enter your question and click Generate to start" />
            </div>
          )}
        </div>
      </div>

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
      {renderFileManager()}
      
      {/* 分块文件管理对话框 */}
      {renderChunkedFileManager()}
    </div>
  );
};

export default Generation; 
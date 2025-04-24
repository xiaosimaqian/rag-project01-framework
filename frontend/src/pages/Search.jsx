// src/pages/Search.jsx
import React, { useState, useEffect } from 'react';
import RandomImage from '../components/RandomImage';
import { apiBaseUrl } from '../config/config';

const Search = () => {
  const [query, setQuery] = useState('');
  const [collection, setCollection] = useState('');
  const [results, setResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [topK, setTopK] = useState(3);
  const [threshold, setThreshold] = useState(0.7);
  const [collections, setCollections] = useState([]);
  const [selectedProvider, setSelectedProvider] = useState('milvus');
  const [wordCountThreshold, setWordCountThreshold] = useState(100);
  const [saveResults, setSaveResults] = useState(false);
  const [status, setStatus] = useState('');

  // 定义支持的向量数据库
  const providers = [
    { id: 'milvus', name: 'Milvus' },
    { id: 'chroma', name: 'ChromaDB' }
  ];

  // 加载collections
  useEffect(() => {
    const fetchCollections = async () => {
      try {
        setStatus('正在加载集合列表...');
        const response = await fetch(`${apiBaseUrl}/collections?provider=${selectedProvider}`);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setCollections(data.collections || []);
        setStatus('');
      } catch (error) {
        console.error('Error fetching collections:', error);
        setStatus(`加载集合列表失败: ${error.message}`);
        setCollections([]);
      }
    };

    fetchCollections();
  }, [selectedProvider]);

  const handleSearch = async () => {
    if (!query || !collection) {
      setStatus('请选择集合并输入搜索内容');
      return;
    }

    setIsSearching(true);
    setStatus('正在搜索...');
    try {
      const searchParams = {
        query: query,
        collection_name: collection,
        provider: selectedProvider,
        top_k: topK,
        threshold: threshold
      };
      
      console.log('发送搜索请求:', searchParams);

      const response = await fetch(`${apiBaseUrl}/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(searchParams),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log('搜索响应:', data);

      if (data.details && data.details.hits && data.details.hits.length > 0) {
        setResults(data.details.hits);
        setStatus('搜索完成！');
      } else {
        setResults([]);
        setStatus('未找到匹配的结果');
      }
    } catch (error) {
      console.error('搜索错误:', error);
      setStatus(`搜索出错: ${error.message}`);
      setResults([]);
    } finally {
      setIsSearching(false);
    }
  };

  const handleSaveResults = async () => {
    if (!results.length) {
      setStatus('没有可保存的搜索结果');
      return;
    }

    try {
      const saveParams = {
        query,
        collection_id: collection,
        results: results
      };

      console.log('发送保存请求:', saveParams);
      
      const response = await fetch(`${apiBaseUrl}/save-search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(saveParams),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setStatus(`结果已保存至: ${data.saved_filepath}`);
    } catch (error) {
      console.error('保存错误:', error);
      setStatus(`保存失败: ${error.message}`);
    }
  };

  return (
    <div className="p-6">
      <h2 className="text-2xl font-bold mb-6">相似度搜索</h2>
      
      <div className="grid grid-cols-12 gap-6">
        {/* Left Panel - Search Controls */}
        <div className="col-span-3 space-y-4">
          <div className="p-4 border rounded-lg bg-white shadow-sm">
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-1">搜索问题</label>
                <textarea
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="请输入您的搜索问题..."
                  className="block w-full p-2 border rounded h-32 resize-none"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">向量数据库</label>
                <select
                  value={selectedProvider}
                  onChange={(e) => setSelectedProvider(e.target.value)}
                  className="block w-full p-2 border rounded"
                >
                  {providers.map(provider => (
                    <option key={provider.id} value={provider.id}>
                      {provider.name}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">集合</label>
                <select
                  value={collection}
                  onChange={(e) => setCollection(e.target.value)}
                  className="block w-full p-2 border rounded"
                >
                  <option value="">选择集合...</option>
                  {collections.map(coll => (
                    <option key={coll.id} value={coll.id}>
                      {coll.name} ({coll.count} 个向量)
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">返回结果数量</label>
                <input
                  type="number"
                  value={topK}
                  onChange={(e) => setTopK(parseInt(e.target.value))}
                  min="1"
                  max="10"
                  className="block w-full p-2 border rounded"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">
                  相似度阈值: {threshold}
                </label>
                <input
                  type="range"
                  value={threshold}
                  onChange={(e) => setThreshold(parseFloat(e.target.value))}
                  min="0"
                  max="1"
                  step="0.1"
                  className="block w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">
                  最小词数: {wordCountThreshold}
                </label>
                <input
                  type="range"
                  value={wordCountThreshold}
                  onChange={(e) => setWordCountThreshold(parseInt(e.target.value))}
                  min="0"
                  max="500"
                  step="10"
                  className="block w-full"
                />
              </div>

              <div className="mt-4">
                <label className="flex items-center space-x-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={saveResults}
                    onChange={(e) => setSaveResults(e.target.checked)}
                    className="form-checkbox h-4 w-4 text-blue-600"
                  />
                  <span className="text-sm font-medium">保存搜索结果</span>
                </label>
              </div>

              <button 
                onClick={handleSearch}
                disabled={isSearching}
                className="w-full px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-blue-300"
              >
                {isSearching ? '搜索中...' : '开始搜索'}
              </button>
            </div>
          </div>

          {status && (
            <div className={`p-4 rounded-lg ${
              status.includes('错误') || status.includes('失败') 
                ? 'bg-red-100 text-red-700' 
                : 'bg-green-100 text-green-700'
            }`}>
              {status}
            </div>
          )}
        </div>

        {/* Right Panel - Results */}
        <div className="col-span-9 border rounded-lg bg-white shadow-sm">
          {results.length > 0 ? (
            <div className="p-4">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-xl font-semibold">搜索结果</h3>
                <button
                  onClick={handleSaveResults}
                  className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
                >
                  保存搜索结果
                </button>
              </div>
              <div className="space-y-4 max-h-[calc(100vh-200px)] overflow-y-auto">
                {results.map((result, idx) => (
                  <div key={idx} className="p-4 border rounded bg-gray-50">
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
                    <p className="text-sm whitespace-pre-wrap">{result.content || result.text}</p>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <RandomImage message="搜索结果将在这里显示" />
          )}
        </div>
      </div>
    </div>
  );
};

export default Search;
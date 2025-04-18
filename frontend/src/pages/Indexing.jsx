// src/pages/Indexing.jsx
import React, { useState, useEffect } from 'react';
import RandomImage from '../components/RandomImage';
import { apiBaseUrl } from '../config/config';

const Indexing = () => {
  const [embeddingFile, setEmbeddingFile] = useState('');
  const [selectedProvider, setSelectedProvider] = useState('milvus');
  const [indexMode, setIndexMode] = useState('flat');
  const [status, setStatus] = useState('');
  const [embeddedFiles, setEmbeddedFiles] = useState([]);
  const [indexingResult, setIndexingResult] = useState(null);
  const [collections, setCollections] = useState([]);
  const [selectedCollection, setSelectedCollection] = useState('');
  const [collectionDetails, setCollectionDetails] = useState(null);
  const [providers, setProviders] = useState(['milvus', 'chroma']);
  const [indexAction, setIndexAction] = useState('create');

  const dbConfigs = {
    milvus: {
      modes: ['flat', 'ivf_flat', 'ivf_sq8', 'hnsw', 'autoindex']
    },
    chroma: {
      modes: ['hnsw']
    }
  };

  const fetchEmbeddedFiles = async () => {
    setStatus('Loading embedding files...');
    try {
      const response = await fetch(`${apiBaseUrl}/list-embedded`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      if (data.documents) {
        setEmbeddedFiles(data.documents.map(doc => ({
          ...doc,
          id: doc.name,
          displayName: doc.name || 'Unnamed File'
        })));
        setStatus('');
      } else {
        setEmbeddedFiles([]);
        setStatus('No embedding files found.');
      }
    } catch (error) {
      console.error('Error fetching embedded files:', error);
      setStatus('Error loading embedding files: ' + error.message);
      setEmbeddedFiles([]);
    }
  };

  const fetchCollections = async (provider) => {
    if (!provider) return;
    setStatus(`Loading collections for ${provider}...`);
    try {
      const response = await fetch(`${apiBaseUrl}/collections?provider=${provider}`);
      if (!response.ok) {
         throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setCollections(data.collections || []);
      setStatus('');
      if (selectedCollection && !data.collections.some(c => c.id === selectedCollection)) {
          setSelectedCollection('');
      }
    } catch (error) {
      console.error(`Error fetching collections for ${provider}:`, error);
      setStatus(`Error loading collections: ${error.message}`);
      setCollections([]);
    }
  };

  useEffect(() => {
    fetchEmbeddedFiles();
    fetchCollections(selectedProvider);
  }, []);

  useEffect(() => {
    if (dbConfigs[selectedProvider]) {
      setIndexMode(dbConfigs[selectedProvider].modes[0]);
    } else {
      setIndexMode('');
    }
    fetchCollections(selectedProvider);
    setSelectedCollection('');
    setIndexAction('create');
    setIndexingResult(null);
  }, [selectedProvider]);

  const handleIndex = async () => {
    if (!embeddingFile) {
      setStatus('请选择嵌入向量文件');
      return;
    }
    if (indexAction === 'append' && !selectedCollection) {
       setStatus('请选择要追加的目标集合');
       return;
    }

    setStatus('正在索引...');
    setIndexingResult(null);

    const payload = {
      fileId: embeddingFile,
      vectorDb: selectedProvider,
      indexMode: indexMode,
      action: indexAction,
      targetCollectionName: indexAction === 'append' ? selectedCollection : null
    };

    try {
      const response = await fetch(`${apiBaseUrl}/index`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || `索引失败，状态码: ${response.status}`);
      }

      setIndexingResult({
        database: data.details.database,
        collection_name: data.details.collection_name,
        index_mode: data.details.index_mode,
        action: data.details.action,
        total_vectors: data.details.total_vectors,
        total_entities: data.details.total_entities,
        processing_time: data.details.processing_time,
        index_size: data.details.index_size
      });
      
      setStatus(data.message);

      if (indexAction === 'create' || indexAction === 'append') {
        fetchCollections(selectedProvider);
      }
    } catch (error) {
      console.error('索引错误:', error);
      setStatus('索引过程中发生错误: ' + error.message);
      setIndexingResult(null);
    }
  };

  const handleDisplay = async (collectionName) => {
    if (!collectionName) return;
    setStatus(`正在获取集合 "${collectionName}" 的信息...`);
    setIndexingResult(null);

    try {
        const response = await fetch(`${apiBaseUrl}/collections/${selectedProvider}/${collectionName}`);
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `获取集合信息失败: ${response.status}`);
        }
        const data = await response.json();

        // 构造显示数据，使用默认值防止未定义错误
        const displayData = {
            database: selectedProvider,
            collection_name: data.name || collectionName,
            total_vectors: data.num_entities || 0,
            schema: data.schema || {},
            index_type: data.index_type || 'N/A',
            index_params: data.index_params || {},
            metric_type: data.metric_type || 'N/A',
            description: data.description || `集合 ${collectionName} 的信息`
        };
        
        setIndexingResult(displayData);
        setStatus(`集合信息已加载: ${displayData.collection_name}`);
    } catch (error) {
        console.error('获取集合信息错误:', error);
        setStatus(`获取集合信息失败: ${error.message}`);
        setIndexingResult(null);
    }
  };

  const handleDelete = async (collectionName) => {
    if (!collectionName) return;

    if (window.confirm(`Are you sure you want to delete collection "${collectionName}" from ${selectedProvider}? This cannot be undone.`)) {
      setStatus(`Deleting collection ${collectionName}...`);
      try {
        const response = await fetch(`${apiBaseUrl}/collections/${selectedProvider}/${collectionName}`, {
          method: 'DELETE',
        });

        if (!response.ok) {
           const errorData = await response.json();
           throw new Error(errorData.detail || `Failed to delete: ${response.status}`);
        }

        setStatus(`Collection "${collectionName}" deleted successfully.`);
        setSelectedCollection('');
        setIndexingResult(null);
        fetchCollections(selectedProvider);
      } catch (error) {
        console.error('Error deleting collection:', error);
        setStatus(`Error deleting collection: ${error.message}`);
      }
    }
  };

  return (
    <div className="p-6 bg-gray-50 min-h-screen">
      <h2 className="text-3xl font-bold mb-6 text-gray-800">Vector Database Indexing</h2>

      <div className="grid grid-cols-1 md:grid-cols-12 gap-6">
        <div className="md:col-span-4 lg:col-span-3">
          <div className="p-5 border rounded-lg bg-white shadow-lg space-y-5">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Embedding File</label>
              <select
                value={embeddingFile}
                onChange={(e) => setEmbeddingFile(e.target.value)}
                className="block w-full p-2 border border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500"
              >
                <option value="">-- Select Embedding File --</option>
                {embeddedFiles.length > 0 ? embeddedFiles.map(file => (
                  <option key={file.id} value={file.id}>
                    {file.displayName}
                  </option>
                )) : <option disabled>Loading or no files...</option>}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Vector Database</label>
              <select
                value={selectedProvider}
                onChange={(e) => setSelectedProvider(e.target.value)}
                className="block w-full p-2 border border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500"
              >
                {providers.map(p => <option key={p} value={p}>{p.charAt(0).toUpperCase() + p.slice(1)}</option>)}
              </select>
            </div>

             <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Index Mode</label>
              <select
                value={indexMode}
                onChange={(e) => setIndexMode(e.target.value)}
                className="block w-full p-2 border border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500"
                disabled={!dbConfigs[selectedProvider]?.modes.length}
              >
                {dbConfigs[selectedProvider]?.modes.length > 0 ? (
                    dbConfigs[selectedProvider].modes.map(mode => (
                    <option key={mode} value={mode}>
                        {mode.toUpperCase()}
                    </option>
                    ))
                ) : (
                    <option disabled>No modes available</option>
                )}
               </select>
            </div>

            <div className="space-y-2">
                 <label className="block text-sm font-medium text-gray-700 mb-1">Index Action</label>
                 <div className="flex items-center space-x-4">
                     <label className="flex items-center">
                         <input
                             type="radio"
                             name="indexAction"
                             value="create"
                             checked={indexAction === 'create'}
                             onChange={(e) => setIndexAction(e.target.value)}
                             className="focus:ring-indigo-500 h-4 w-4 text-indigo-600 border-gray-300"
                         />
                         <span className="ml-2 text-sm text-gray-700">Create New Collection</span>
                     </label>
                     <label className="flex items-center">
                         <input
                             type="radio"
                             name="indexAction"
                             value="append"
                             checked={indexAction === 'append'}
                             onChange={(e) => setIndexAction(e.target.value)}
                             className="focus:ring-indigo-500 h-4 w-4 text-indigo-600 border-gray-300"
                             disabled={collections.length === 0}
                         />
                         <span className={`ml-2 text-sm ${collections.length === 0 ? 'text-gray-400' : 'text-gray-700'}`}>Add to Existing</span>
                     </label>
                 </div>
            </div>

            {indexAction === 'append' && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Target Collection (Append Mode) <span className="text-red-500">*</span>
                </label>
                <select
                  value={selectedCollection}
                  onChange={(e) => setSelectedCollection(e.target.value)}
                  className="block w-full p-2 border border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500"
                  required
                >
                  <option value="">-- Select Collection to Append --</option>
                  {collections.map(coll => (
                    <option key={coll.id} value={coll.id}>
                      {coll.name} ({coll.count ?? 'N/A'} vectors)
                    </option>
                  ))}
                </select>
              </div>
            )}

            <button
              onClick={handleIndex}
              className="w-full px-4 py-2 bg-indigo-600 text-white rounded-md shadow-sm hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:bg-indigo-300"
              disabled={!embeddingFile || (indexAction === 'append' && !selectedCollection) || status.includes('Loading')}
            >
              {indexAction === 'create' ? 'Create & Index' : 'Append to Collection'}
            </button>

             <div className="pt-4 border-t border-gray-200 mt-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">Manage Existing Collections</label>
                <select
                    value={selectedCollection}
                    onChange={(e) => setSelectedCollection(e.target.value)}
                    className="block w-full p-2 border border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 mb-2"
                >
                    <option value="">-- Select Collection --</option>
                     {collections.length > 0 ? collections.map(coll => (
                        <option key={coll.id} value={coll.id}>
                            {coll.name} ({coll.count ?? 'N/A'} vectors)
                        </option>
                    )) : <option disabled>No collections found</option>}
                </select>
                <div className="flex space-x-2">
                    <button
                        onClick={() => handleDisplay(selectedCollection)}
                        disabled={!selectedCollection}
                        className="flex-1 px-4 py-2 bg-green-600 text-white rounded-md shadow-sm hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 disabled:bg-green-300"
                    >
                        Display Info
                    </button>
                    <button
                        onClick={() => handleDelete(selectedCollection)}
                        disabled={!selectedCollection}
                        className="flex-1 px-4 py-2 bg-red-600 text-white rounded-md shadow-sm hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 disabled:bg-red-300"
                    >
                        Delete
                    </button>
                </div>
            </div>

            {status && (
              <div className={`mt-4 p-3 rounded border ${status.includes('Error') ? 'bg-red-50 border-red-300 text-red-700' : 'bg-blue-50 border-blue-300 text-blue-700'}`}>
                <p className="text-sm">{status}</p>
              </div>
            )}
          </div>
        </div>

        <div className="md:col-span-8 lg:col-span-9 border rounded-lg bg-white shadow-lg min-h-[400px] flex items-center justify-center">
          {indexingResult ? (
            <div className="p-6 w-full">
                <h3 className="text-xl font-semibold mb-4 text-gray-800 border-b pb-2">
                    集合详细信息
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* 基本信息 */}
                    <div className="bg-white p-4 rounded-lg border border-gray-200 shadow-sm">
                        <h4 className="font-semibold text-lg mb-3 text-indigo-600">基本信息</h4>
                        <div className="space-y-2">
                            <div className="flex justify-between">
                                <span className="text-gray-600">数据库类型</span>
                                <span className="font-medium">{indexingResult.database || 'N/A'}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-gray-600">集合名称</span>
                                <span className="font-medium">{indexingResult.collection_name || 'N/A'}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-gray-600">向量总数</span>
                                <span className="font-medium">{indexingResult.total_vectors?.toLocaleString() || 0}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-gray-600">创建时间</span>
                                <span className="font-medium">
                                    {indexingResult.creation_time ? 
                                        new Date(indexingResult.creation_time).toLocaleString() : 'N/A'}
                                </span>
                            </div>
                        </div>
                    </div>

                    {/* 索引信息 */}
                    <div className="bg-white p-4 rounded-lg border border-gray-200 shadow-sm">
                        <h4 className="font-semibold text-lg mb-3 text-indigo-600">索引配置</h4>
                        <div className="space-y-2">
                            <div className="flex justify-between">
                                <span className="text-gray-600">索引类型</span>
                                <span className="font-medium">{indexingResult.index_type || 'N/A'}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-gray-600">度量类型</span>
                                <span className="font-medium">{indexingResult.metric_type || 'N/A'}</span>
                            </div>
                            {Object.keys(indexingResult.index_params || {}).length > 0 && (
                                <div className="mt-3">
                                    <span className="text-gray-600 block mb-1">索引参数</span>
                                    <div className="bg-gray-50 p-2 rounded text-sm font-mono overflow-x-auto">
                                        <pre className="whitespace-pre-wrap">
                                            {JSON.stringify(indexingResult.index_params, null, 2)}
                                        </pre>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Schema 信息 */}
                    {indexingResult.schema && indexingResult.schema.fields && (
                        <div className="col-span-1 md:col-span-2 bg-white p-4 rounded-lg border border-gray-200 shadow-sm">
                            <h4 className="font-semibold text-lg mb-3 text-indigo-600">集合结构 (Schema)</h4>
                            <div className="overflow-x-auto">
                                <table className="min-w-full divide-y divide-gray-200">
                                    <thead className="bg-gray-50">
                                        <tr>
                                            <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">字段名</th>
                                            <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">类型</th>
                                            <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">说明</th>
                                            <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">属性</th>
                                        </tr>
                                    </thead>
                                    <tbody className="bg-white divide-y divide-gray-200">
                                        {indexingResult.schema.fields.map((field, index) => (
                                            <tr key={index} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                                                <td className="px-4 py-2 text-sm font-medium text-gray-900">
                                                    {field.name}
                                                    {field.is_primary && (
                                                        <span className="ml-2 inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-yellow-100 text-yellow-800">
                                                            主键
                                                        </span>
                                                    )}
                                                </td>
                                                <td className="px-4 py-2 text-sm text-gray-500">
                                                    <div className="flex flex-col">
                                                        <span>{field.type_description}</span>
                                                        <span className="text-xs text-gray-400">({field.dtype})</span>
                                                    </div>
                                                </td>
                                                <td className="px-4 py-2 text-sm text-gray-500">
                                                    {field.description || '-'}
                                                </td>
                                                <td className="px-4 py-2 text-sm text-gray-500">
                                                    <div className="space-y-1">
                                                        {field.is_primary && (
                                                            <div className="text-xs">
                                                                <span className="font-medium">主键</span>
                                                            </div>
                                                        )}
                                                        {field.auto_id && (
                                                            <div className="text-xs">
                                                                <span className="font-medium">自动生成ID</span>
                                                            </div>
                                                        )}
                                                        {field.max_length && (
                                                            <div className="text-xs">
                                                                <span className="font-medium">最大长度：</span>
                                                                {field.max_length}
                                                            </div>
                                                        )}
                                                        {field.dimension && (
                                                            <div className="text-xs">
                                                                <span className="font-medium">向量维度：</span>
                                                                {field.dimension}
                                                            </div>
                                                        )}
                                                        {field.params && Object.keys(field.params).length > 0 && (
                                                            <details className="cursor-pointer text-xs">
                                                                <summary className="text-indigo-600 hover:text-indigo-700">
                                                                    更多参数
                                                                </summary>
                                                                <pre className="mt-2 text-xs bg-gray-50 p-2 rounded overflow-x-auto">
                                                                    {JSON.stringify(field.params, null, 2)}
                                                                </pre>
                                                            </details>
                                                        )}
                                                    </div>
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    )}
                </div>
            </div>
          ) : (
            <RandomImage message="集合信息将在这里显示" />
          )}
        </div>
      </div>
    </div>
  );
};

export default Indexing;
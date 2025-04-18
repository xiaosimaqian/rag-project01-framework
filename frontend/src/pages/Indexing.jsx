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

      setIndexingResult(data);
      
      const statusMessage = [
        `索引完成：${data.collection_name || 'N/A'}`,
        `模式：${data.index_mode || 'N/A'}`,
        `处理向量数：${data.total_vectors || 0}`,
        `处理时间：${data.processing_time ? data.processing_time.toFixed(2) + 's' : 'N/A'}`,
        `索引大小：${data.index_size || 'N/A'}`
      ].join(' | ');
      
      setStatus(statusMessage);

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
    setStatus(`Fetching details for ${collectionName}...`);
    setIndexingResult(null);

    try {
      const response = await fetch(`${apiBaseUrl}/collections/${selectedProvider}/${collectionName}`);
       if (!response.ok) {
         const errorData = await response.json();
         throw new Error(errorData.detail || `Failed to fetch details: ${response.status}`);
      }
      const data = await response.json();

      const displayData = {
        database: selectedProvider,
        collection_name: data.name,
        total_vectors: data.num_entities,
        schema: data.schema,
        indexes: data.indexes,
        description: data.description
      };
      setIndexingResult(displayData);
      setStatus(`Details for collection: ${data.name}`);
    } catch (error) {
      console.error('Error displaying collection:', error);
      setStatus(`Error displaying collection: ${error.message}`);
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
              <h3 className="text-xl font-semibold mb-4 text-gray-800">
                索引结果
              </h3>
              <div className="space-y-3 text-sm text-gray-700 bg-gray-50 p-4 rounded border border-gray-200 max-w-full overflow-x-auto">
                <p><strong>数据库：</strong> {indexingResult.database}</p>
                <p><strong>集合名称：</strong> {indexingResult.collection_name}</p>
                <p><strong>索引模式：</strong> {indexingResult.index_mode}</p>
                <p><strong>向量总数：</strong> {indexingResult.total_vectors}</p>
                <p><strong>处理时间：</strong> {indexingResult.processing_time ? `${indexingResult.processing_time.toFixed(2)}秒` : 'N/A'}</p>
                <p><strong>索引大小：</strong> {indexingResult.index_size}</p>
                {indexingResult.schema && (
                  <div>
                    <strong>集合结构：</strong>
                    <pre className="bg-gray-100 p-2 rounded mt-1 text-xs overflow-auto">
                      {typeof indexingResult.schema === 'string' 
                        ? indexingResult.schema 
                        : JSON.stringify(indexingResult.schema, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            </div>
          ) : (
            <RandomImage message="索引结果将在这里显示" />
          )}
        </div>
      </div>
    </div>
  );
};

export default Indexing;
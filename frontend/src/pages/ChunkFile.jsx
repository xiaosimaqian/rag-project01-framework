import React, { useState, useEffect, useMemo } from 'react';
import RandomImage from '../components/RandomImage';
import { apiBaseUrl } from '../config/config';

const ChunkFile = () => {
  const [loadedDocuments, setLoadedDocuments] = useState([]);
  const [selectedDoc, setSelectedDoc] = useState('');
  const [chunkingOption, setChunkingOption] = useState('by_pages');
  const [chunkSize, setChunkSize] = useState(1000);
  const [chunks, setChunks] = useState(null);
  const [status, setStatus] = useState('');
  const [activeTab, setActiveTab] = useState('chunks');
  const [processingStatus, setProcessingStatus] = useState('');
  const [chunkedDocuments, setChunkedDocuments] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  // 使用useMemo优化chunks的渲染
  const renderedChunks = useMemo(() => {
    if (!chunks || !Array.isArray(chunks.chunks)) return null;
    
    return chunks.chunks.map((chunk) => (
      <div key={chunk.metadata.chunk_id} className="p-3 border rounded bg-gray-50">
        <div className="font-medium text-sm text-gray-500 mb-1">
          Chunk {chunk.metadata.chunk_id}
        </div>
        <div className="text-xs text-gray-400 mb-2">
          Page(s): {chunk.metadata.page_range} | 
          Words: {chunk.metadata.word_count}
        </div>
        <div className="text-sm mt-2">
          <div className="text-gray-600">{chunk.content}</div>
        </div>
      </div>
    ));
  }, [chunks]);

  useEffect(() => {
    fetchLoadedDocuments();
  }, []);

  const fetchLoadedDocuments = async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/documents?type=all`);
      const data = await response.json();
      setLoadedDocuments(data.documents);

      const chunkedResponse = await fetch(`${apiBaseUrl}/documents?type=chunked`);
      if (!chunkedResponse.ok) {
        throw new Error(`HTTP error! status: ${chunkedResponse.status}`);
      }
      const chunkedData = await chunkedResponse.json();
      
      if (!chunkedData.documents || !Array.isArray(chunkedData.documents)) {
        console.error('Invalid chunked documents data:', chunkedData);
        return;
      }

      const chunkedDocsWithDetails = await Promise.all(
        chunkedData.documents.map(async (doc) => {
          try {
            const detailResponse = await fetch(`${apiBaseUrl}/documents/${doc.name}?type=chunked`);
            if (!detailResponse.ok) {
              console.error(`Error fetching details for ${doc.name}:`, detailResponse.status);
              return doc;
            }
            const detailData = await detailResponse.json();
            
            return {
              ...doc,
              total_pages: detailData.total_pages,
              total_chunks: detailData.total_chunks,
              chunking_method: detailData.chunking_method,
              timestamp: detailData.timestamp
            };
          } catch (error) {
            console.error(`Error processing document ${doc.name}:`, error);
            return doc;
          }
        })
      );
      
      setChunkedDocuments(chunkedDocsWithDetails);
    } catch (error) {
      console.error('Error fetching documents:', error);
      setProcessingStatus(`Error fetching documents: ${error.message}`);
    }
  };

  const handleChunk = async () => {
    if (!selectedDoc || !chunkingOption) {
      setProcessingStatus('请选择文档和分块选项');
      return;
    }

    setIsLoading(true);
    setProcessingStatus('处理中...');
    console.log('开始分块处理:', {
      doc_id: selectedDoc,
      chunking_option: chunkingOption,
      chunk_size: chunkSize,
      doc_type: 'parsed'
    });

    try {
      const response = await fetch(`${apiBaseUrl}/chunk`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          doc_id: selectedDoc,
          chunking_option: chunkingOption,
          chunk_size: chunkSize,
          doc_type: 'parsed'
        }),
      });

      console.log('服务器响应状态:', response.status);
      
      if (!response.ok) {
        const errorData = await response.json();
        console.error('服务器错误:', errorData);
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log('分块结果:', data);
      
      setChunks(data);
      setProcessingStatus('分块完成!');
      setActiveTab('chunks');
      
      // 刷新文档列表
      fetchLoadedDocuments();
    } catch (error) {
      console.error('分块处理错误:', error);
      setProcessingStatus(`错误: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteDocument = async (docName) => {
    try {
      const response = await fetch(`${apiBaseUrl}/documents/${docName}?type=chunked`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      setProcessingStatus('Document deleted successfully');
      fetchLoadedDocuments();
      if (selectedDoc === docName) {
        setSelectedDoc('');
        setChunks(null);
      }
    } catch (error) {
      console.error('Error deleting document:', error);
      setProcessingStatus(`Error deleting document: ${error.message}`);
    }
  };

  const handleViewDocument = async (docName) => {
    try {
      const response = await fetch(`${apiBaseUrl}/documents/${docName}?type=chunked`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setChunks(data);
      setActiveTab('chunks');
    } catch (error) {
      console.error('Error viewing document:', error);
      setProcessingStatus(`Error viewing document: ${error.message}`);
    }
  };

  const renderRightPanel = () => {
    return (
      <div className="p-4 w-full h-full flex flex-col">
        <div className="flex mb-4 border-b">
          <button
            className={`px-4 py-2 ${
              activeTab === 'chunks'
                ? 'border-b-2 border-blue-500 text-blue-600'
                : 'text-gray-600'
            }`}
            onClick={() => setActiveTab('chunks')}
          >
            Chunks Preview
          </button>
          <button
            className={`px-4 py-2 ml-4 ${
              activeTab === 'documents'
                ? 'border-b-2 border-blue-500 text-blue-600'
                : 'text-gray-600'
            }`}
            onClick={() => setActiveTab('documents')}
          >
            Document Management
          </button>
        </div>

        {activeTab === 'chunks' ? (
          chunks ? (
            <div className="w-full">
              <div className="mb-4 p-3 border rounded bg-gray-100">
                <h4 className="font-medium mb-2">Document Information</h4>
                <div className="text-sm text-gray-600">
                  <p>Filename: {chunks.filename}</p>
                  <p>Total Pages: {chunks.total_pages}</p>
                  <p>Total Chunks: {chunks.total_chunks}</p>
                  <p>Loading Method: {chunks.loading_method}</p>
                  <p>Chunking Method: {chunks.chunking_method}</p>
                  <p>Timestamp: {chunks.timestamp ? new Date(chunks.timestamp).toLocaleString() : 'N/A'}</p>
                </div>
              </div>
              <div className="space-y-3 max-h-[calc(100vh-300px)] overflow-y-auto">
                {renderedChunks}
              </div>
            </div>
          ) : (
            <RandomImage message="Select a document and create chunks to see the results here" />
          )
        ) : (
          <div className="flex flex-col w-full h-full">
            <h3 className="text-xl font-semibold mb-4">Document Management</h3>
            <div className="space-y-4 w-full">
              {chunkedDocuments.length > 0 ? (
                chunkedDocuments.map((doc) => (
                  <div key={doc.name} className="p-4 border rounded-lg bg-gray-50 w-full">
                    <div className="flex justify-between items-start w-full">
                      <div className="flex-grow">
                        <h4 className="font-medium text-lg">{doc.name}</h4>
                        <div className="text-sm text-gray-600 mt-1">
                          <p>Pages: {doc.total_pages || 'N/A'}</p>
                          <p>Chunks: {doc.total_chunks || 'N/A'}</p>
                          <p>Chunking Method: {doc.chunking_method || 'N/A'}</p>
                          <p>Processing Date: {doc.timestamp ? new Date(doc.timestamp).toLocaleString() : 'N/A'}</p>
                        </div>
                      </div>
                      <div className="flex space-x-2 ml-4">
                        <button
                          onClick={() => handleViewDocument(doc.name)}
                          className="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600"
                        >
                          View
                        </button>
                        <button
                          onClick={() => handleDeleteDocument(doc.name)}
                          className="px-3 py-1 bg-red-500 text-white rounded hover:bg-red-600"
                        >
                          Delete
                        </button>
                      </div>
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-center text-gray-500 py-8 w-full">
                  No chunked documents available
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="p-6">
      <h2 className="text-2xl font-bold mb-6">Chunk File</h2>
      
      <div className="grid grid-cols-12 gap-6">
        {/* Left Panel */}
        <div className="col-span-3 space-y-4">
          <div className="p-4 border rounded-lg bg-white shadow-sm">
            <div className="mb-4">
              <label className="block text-sm font-medium mb-1">Select Document</label>
              <select
                value={selectedDoc}
                onChange={(e) => setSelectedDoc(e.target.value)}
                className="block w-full p-2 border rounded"
                disabled={isLoading}
              >
                <option value="">Choose a document...</option>
                {loadedDocuments.map((doc) => (
                  <option key={doc.name} value={doc.name}>
                    {doc.name}
                  </option>
                ))}
              </select>
            </div>

            <div className="mb-4">
              <label className="block text-sm font-medium mb-1">Chunking Method</label>
              <select
                value={chunkingOption}
                onChange={(e) => setChunkingOption(e.target.value)}
                className="block w-full p-2 border rounded"
                disabled={isLoading}
              >
                <option value="by_pages">By Pages</option>
                <option value="fixed_size">Fixed Size</option>
                <option value="by_paragraphs">By Paragraphs</option>
                <option value="by_sentences">By Sentences</option>
                <option value="by_semicolons">By Semicolons (Verilog)</option>
              </select>
            </div>

            {chunkingOption === 'fixed_size' && (
              <div className="mb-4">
                <label className="block text-sm font-medium mb-1">Chunk Size</label>
                <input
                  type="number"
                  value={chunkSize}
                  onChange={(e) => setChunkSize(Number(e.target.value))}
                  className="block w-full p-2 border rounded"
                  min="100"
                  max="5000"
                  disabled={isLoading}
                />
              </div>
            )}

            {chunkingOption === 'by_semicolons' && (
              <div className="mb-4">
                <label className="block text-sm font-medium mb-1">Verilog File Options</label>
                <div className="text-sm text-gray-600">
                  <p>This option will split Verilog files by semicolons, preserving module definitions and statements.</p>
                </div>
              </div>
            )}

            <button 
              onClick={handleChunk}
              className={`w-full px-4 py-2 text-white rounded ${
                isLoading 
                  ? 'bg-gray-400 cursor-not-allowed' 
                  : 'bg-blue-500 hover:bg-blue-600'
              }`}
              disabled={!selectedDoc || isLoading}
            >
              {isLoading ? 'Processing...' : 'Create Chunks'}
            </button>
          </div>

          {processingStatus && (
            <div className={`p-4 rounded-lg ${
              processingStatus.includes('Error') ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'
            }`}>
              {processingStatus}
            </div>
          )}
        </div>

        {/* Right Panel */}
        <div className="col-span-9 border rounded-lg bg-white shadow-sm">
          {renderRightPanel()}
        </div>
      </div>
    </div>
  );
};

export default ChunkFile;
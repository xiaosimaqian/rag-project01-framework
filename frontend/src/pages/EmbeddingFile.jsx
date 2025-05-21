// src/pages/EmbeddingFile.jsx
import React, { useState, useEffect } from 'react';
import { Button, message, Select, Space, Table, Modal, Input, Tag, Tooltip } from 'antd';
import { DeleteOutlined, EyeOutlined, SearchOutlined, ReloadOutlined } from '@ant-design/icons';
import { apiBaseUrl } from '../config/config';

const EmbeddingFile = () => {
  // 状态管理
  const [selectedFile, setSelectedFile] = useState(null);
  const [embeddingModel, setEmbeddingModel] = useState('text-embedding-3-small');
  const [isProcessing, setIsProcessing] = useState(false);
  const [embeddedFiles, setEmbeddedFiles] = useState([]);
  const [chunkedFiles, setChunkedFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [searchText, setSearchText] = useState('');
  const [previewVisible, setPreviewVisible] = useState(false);
  const [previewFile, setPreviewFile] = useState(null);
  const [embeddingTaskId, setEmbeddingTaskId] = useState(null);
  const [isEmbeddingInProgress, setIsEmbeddingInProgress] = useState(false);
  const [cancellationRequested, setCancellationRequested] = useState(false);
  const [embeddingStatusMessage, setEmbeddingStatusMessage] = useState('');

  // 获取分块文件列表
  const fetchChunkedFiles = async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/chunked-files`);
      if (!response.ok) {
        throw new Error('获取分块文件列表失败');
      }
      const data = await response.json();
      setChunkedFiles(data.files || []);
    } catch (error) {
      message.error(error.message);
    }
  };

  // 获取已嵌入文件列表
  const fetchEmbeddedFiles = async () => {
    console.log('fetchEmbeddedFiles called');
    try {
      setLoading(true);
      const response = await fetch(`${apiBaseUrl}/list-embedded`);
      if (!response.ok) {
        throw new Error('获取已嵌入文件列表失败');
      }
      const data = await response.json();
      if (data.documents) {
        setEmbeddedFiles(data.documents.map(doc => ({
          id: doc.name,
          filename: doc.metadata.document_name || doc.name,
          model: doc.metadata.embedding_model,
          dimension: doc.metadata.vector_dimension,
          vector_count: doc.metadata.total_vectors,
          timestamp: doc.metadata.embedding_timestamp
        })));
      } else {
        setEmbeddedFiles([]);
      }
    } catch (error) {
      message.error(error.message);
    } finally {
      setLoading(false);
    }
  };

  // 初始加载和定期刷新文件列表
  useEffect(() => {
    let isMounted = true;
    let retryCount = 0;
    const MAX_RETRIES = 3;
    const RETRY_DELAY = 5000; // 5秒
    const POLLING_INTERVAL = 60000; // 1分钟

    const fetchData = async () => {
      try {
        if (isMounted) {
          await fetchChunkedFiles();
          await fetchEmbeddedFiles();
          retryCount = 0; // 重置重试计数
        }
      } catch (error) {
        console.error('Error fetching data:', error);
        if (retryCount < MAX_RETRIES) {
          retryCount++;
          setTimeout(fetchData, RETRY_DELAY);
        }
      }
    };

    fetchData();
    const timer = setInterval(fetchData, POLLING_INTERVAL);

    return () => {
      isMounted = false;
      clearInterval(timer);
    };
  }, []);

  // 处理文件嵌入
  const handleEmbed = async () => {
    console.log('handleEmbed called, selectedFile:', selectedFile, 'embeddingModel:', embeddingModel);
    if (!selectedFile) {
      message.error('请选择要嵌入的文件');
      return;
    }
    setIsProcessing(true);
    setIsEmbeddingInProgress(true);
    setCancellationRequested(false);
    setEmbeddingTaskId(null);
    setEmbeddingStatusMessage('开始嵌入处理...');
    try {
      const docName = selectedFile.endsWith('.json') ? selectedFile.slice(0, -5) : selectedFile;
      const response = await fetch(`${apiBaseUrl}/embed`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          docName: docName,
          docType: "chunked",
          embeddingConfig: {
            provider: "ollama",
            model: embeddingModel
          }
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || '文件嵌入失败');
      }

      const result = await response.json();
      if (result.status === 'success') {
        message.success('文件嵌入成功');
        setEmbeddingStatusMessage('文件嵌入成功');
      } else {
        setEmbeddingStatusMessage('嵌入请求已发送，请稍后查看结果。');
      }
    } catch (error) {
      console.error('Error:', error);
      message.error(error.message);
      setEmbeddingStatusMessage(`嵌入失败: ${error.message}`);
    } finally {
      setIsProcessing(false);
      setIsEmbeddingInProgress(false);
      setEmbeddingTaskId(null);
    }
  };

  const handleCancelEmbedding = async () => {
    if (!embeddingTaskId) {
      message.error('没有正在进行的嵌入任务可中止。');
      return;
    }
    if (cancellationRequested) {
      message.info('取消请求已发送。');
      return;
    }
    setEmbeddingStatusMessage(`正在请求中止任务 ${embeddingTaskId}...`);
    setCancellationRequested(true);
    try {
      const response = await fetch(`${apiBaseUrl}/embedding/cancel/${embeddingTaskId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      const result = await response.json();
      if (!response.ok) {
        throw new Error(result.detail || `中止嵌入任务失败 (状态 ${response.status})`);
      }
      message.success(result.message || `任务 ${embeddingTaskId} 的中止请求已成功发送。`);
      setEmbeddingStatusMessage(result.message || `任务 ${embeddingTaskId} 中止请求已发送。后端将在处理完当前块后停止。`);
      setIsEmbeddingInProgress(false);
      setEmbeddingTaskId(null);
    } catch (error) {
      message.error(`中止嵌入失败: ${error.message}`);
      setEmbeddingStatusMessage(`中止嵌入任务 ${embeddingTaskId} 失败: ${error.message}`);
      setCancellationRequested(false);
    }
  };

  // 预览嵌入内容
  const handlePreview = async (file) => {
    try {
      const response = await fetch(`${apiBaseUrl}/embedded-docs/${file.id}`);
      if (!response.ok) {
        throw new Error('获取嵌入内容失败');
      }
      const data = await response.json();
      
      // 检查数据格式
      if (!data || !data.embeddings || !Array.isArray(data.embeddings)) {
        throw new Error('无效的嵌入数据格式');
      }

      // 构建预览数据
      setPreviewFile({
        metadata: {
          filename: file.filename,
          model: file.model,
          dimension: file.dimension,
          vector_count: file.vector_count,
          timestamp: file.timestamp
        },
        vectors: data.embeddings.map((embedding, index) => ({
          dimension: embedding.dimension || file.dimension,
          content: (embedding.metadata && embedding.metadata.content) || embedding.content || embedding.text || `向量 ${index + 1}`
        }))
      });
      setPreviewVisible(true);
    } catch (error) {
      message.error(error.message);
    }
  };

  // 删除嵌入文件
  const handleDelete = (file) => {
    Modal.confirm({
      title: '确认删除',
      content: `确定要删除文件 "${file.filename}" 的嵌入结果吗？`,
      okText: '确认',
      okType: 'danger',
      cancelText: '取消',
      onOk: async () => {
        try {
          const response = await fetch(`${apiBaseUrl}/documents/${file.id}?type=embedded`, {
            method: 'DELETE',
          });
          if (!response.ok) {
            throw new Error('删除嵌入文件失败');
          }
          message.success('嵌入文件删除成功');
          fetchEmbeddedFiles();
        } catch (error) {
          message.error(error.message);
        }
      },
    });
  };

  // 表格列定义
  const columns = [
    {
      title: '文件名',
      dataIndex: 'filename',
      key: 'filename',
      render: (text) => <span className="font-medium">{text}</span>,
    },
    {
      title: '嵌入模型',
      dataIndex: 'model',
      key: 'model',
      render: (model) => <Tag color="blue">{model}</Tag>,
    },
    {
      title: '向量维度',
      dataIndex: 'dimension',
      key: 'dimension',
      render: (dim) => <Tag>{dim}</Tag>,
    },
    {
      title: '向量数量',
      dataIndex: 'vector_count',
      key: 'vector_count',
      render: (count) => <Tag color="green">{count}</Tag>,
    },
    {
      title: '处理时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      render: (timestamp) => new Date(timestamp).toLocaleString(),
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record) => (
        <Space>
          <Tooltip title="预览">
            <Button
              type="text"
              icon={<EyeOutlined />}
              onClick={() => handlePreview(record)}
            />
          </Tooltip>
          <Tooltip title="删除">
            <Button
              type="text"
              danger
              icon={<DeleteOutlined />}
              onClick={() => handleDelete(record)}
            />
          </Tooltip>
        </Space>
      ),
    },
  ];

  // 过滤文件列表
  const filteredFiles = embeddedFiles.filter((file) =>
    file.filename.toLowerCase().includes(searchText.toLowerCase())
  );

  return (
    <div className="p-6">
      <h2 className="text-2xl font-bold mb-6">文本嵌入</h2>

      <div className="grid grid-cols-12 gap-6">
        {/* 左侧面板 (3/12) */}
        <div className="col-span-3 space-y-4">
          <div className="p-4 border rounded-lg bg-white shadow-sm">
            <div className="space-y-4">
              <div>
                <label htmlFor="select-file-for-embedding" className="block text-sm font-medium mb-1">选择文件</label>
                <Select
                  id="select-file-for-embedding"
                  value={selectedFile}
                  onChange={setSelectedFile}
                  className="w-full"
                  placeholder="请选择要嵌入的文件"
                  loading={loading}
                >
                  {chunkedFiles.map((file) => (
                    <Select.Option key={file.id} value={file.id}>
                      {file.name}
                    </Select.Option>
                  ))}
                </Select>
              </div>

              <div>
                <label htmlFor="select-embedding-model" className="block text-sm font-medium mb-1">嵌入模型</label>
                <Select
                  id="select-embedding-model"
                  value={embeddingModel}
                  onChange={setEmbeddingModel}
                  className="w-full"
                >
                  <Select.Option value="text-embedding-3-small">text-embedding-3-small</Select.Option>
                  <Select.Option value="text-embedding-3-large">text-embedding-3-large</Select.Option>
                  <Select.Option value="text-embedding-ada-002">text-embedding-ada-002</Select.Option>
                  <Select.Option value="bge-m3:latest">bge-m3:latest (Ollama)</Select.Option>
                </Select>
              </div>

              <Button
                type="primary"
                onClick={handleEmbed}
                loading={isEmbeddingInProgress && !cancellationRequested}
                disabled={!selectedFile || isEmbeddingInProgress}
                block
              >
                {isEmbeddingInProgress ? '处理中...' : '开始嵌入'}
              </Button>
              <Button
                type="default"
                danger
                onClick={handleCancelEmbedding}
                disabled={!isEmbeddingInProgress || !embeddingTaskId || cancellationRequested}
                block
              >
                中止嵌入
              </Button>
              {embeddingStatusMessage && (
                <div style={{ marginTop: 10 }}>
                  <span>{embeddingStatusMessage}</span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* 右侧面板 (9/12) */}
        <div className="col-span-9 space-y-4">
          {/* 已嵌入文件列表 */}
          <div className="border rounded-lg bg-white shadow-sm">
            <div className="p-4 border-b">
              <div className="flex justify-between items-center">
                <h3 className="text-lg font-semibold">已嵌入文件</h3>
                <Space>
                  <Input
                    id="embedding-file-search-input"
                    placeholder="搜索文件..."
                    prefix={<SearchOutlined />}
                    value={searchText}
                    onChange={(e) => setSearchText(e.target.value)}
                    style={{ width: 200 }}
                  />
                  <Button
                    icon={<ReloadOutlined />}
                    onClick={fetchEmbeddedFiles}
                    loading={loading}
                  />
                </Space>
              </div>
            </div>
            <Table
              columns={columns}
              dataSource={filteredFiles}
              rowKey={(record) => record.id || record.filename || Math.random().toString(36).substr(2, 9)}
              loading={loading}
              pagination={{
                defaultPageSize: 10,
                showSizeChanger: true,
                showTotal: (total) => `共 ${total} 个文件`,
              }}
            />
          </div>
        </div>
      </div>

      {/* 嵌入预览模态框 */}
      <Modal
        title="嵌入预览"
        open={previewVisible}
        onCancel={() => setPreviewVisible(false)}
        width={800}
        footer={null}
      >
        {previewFile && (
          <div className="max-h-[calc(100vh-200px)] overflow-y-auto">
            <div className="mb-4 p-3 border rounded bg-gray-100">
              <h4 className="font-medium mb-2">文件信息</h4>
              <div className="text-sm text-gray-600">
                <p>文件名: {previewFile.metadata?.filename}</p>
                <p>嵌入模型: {previewFile.metadata?.model}</p>
                <p>向量维度: {previewFile.metadata?.dimension}</p>
                <p>向量数量: {previewFile.metadata?.vector_count}</p>
                <p>处理时间: {previewFile.metadata?.timestamp && new Date(previewFile.metadata.timestamp).toLocaleString()}</p>
              </div>
            </div>
            <div className="space-y-3">
              {previewFile.vectors.map((vector, idx) => (
                <div key={idx} className="p-4 border rounded-lg bg-white shadow-sm">
                  <h3 className="text-lg font-semibold mb-2">向量 {idx + 1}</h3>
                  <p className="text-sm text-gray-500">维度: {vector.dimension}</p>
                  <p className="text-sm text-gray-500 mt-2">内容: {vector.content}</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
};

export default EmbeddingFile;
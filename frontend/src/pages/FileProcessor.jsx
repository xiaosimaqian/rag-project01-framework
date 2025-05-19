import React, { useState, useEffect } from 'react';
import { Button, message, Select, Space, Empty, Table, Modal, Input, Tag, Tooltip, Steps, Upload, Form } from 'antd';
import { DeleteOutlined, EyeOutlined, SearchOutlined, ReloadOutlined, UploadOutlined, FileTextOutlined, CheckCircleOutlined, InboxOutlined } from '@ant-design/icons';
import { apiBaseUrl } from '../config/config';

const FileProcessor = () => {
  // 文件处理状态
  const [currentStep, setCurrentStep] = useState(0);
  const [file, setFile] = useState(null);
  const [fileType, setFileType] = useState('');
  const [loadingMethod, setLoadingMethod] = useState('');
  const [parsingOption, setParsingOption] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [status, setStatus] = useState('');

  // 文件列表状态
  const [parsedFiles, setParsedFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [searchText, setSearchText] = useState('');
  const [previewVisible, setPreviewVisible] = useState(false);
  const [previewFile, setPreviewFile] = useState(null);

  // 加载工具和解析选项
  const [loadingTools, setLoadingTools] = useState([]);
  const [parsingOptions, setParsingOptions] = useState([]);

  // 获取已处理文件列表
  const fetchParsedFiles = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${apiBaseUrl}/documents`);
      if (!response.ok) {
        throw new Error('获取已处理文件列表失败');
      }
      const data = await response.json();
      console.log('获取到的文件列表数据:', data);
      
      // 处理数据格式
      const formattedFiles = (data.documents || [])
        .filter(file => {
          // 排除分块后的文件
          const isChunked = file.name.includes('_chunked_') || 
                          file.name.startsWith('chunked_') || 
                          file.metadata?.chunking_method;
          return !isChunked;
        })
        .map(file => {
          return {
            id: file.id,
            filename: file.name,
            file_type: file.metadata?.file_type || 'unknown',
            loading_method: file.metadata?.loading_method || '未知',
            timestamp: file.metadata?.timestamp,
            key: file.id || Math.random().toString(36).substr(2, 9)
          };
        });
      
      console.log('格式化后的文件列表:', formattedFiles);
      setParsedFiles(formattedFiles);
    } catch (error) {
      console.error('获取文件列表错误:', error);
      message.error(error.message);
    } finally {
      setLoading(false);
    }
  };

  // 初始加载文件列表
  useEffect(() => {
    fetchParsedFiles();
  }, []);

  // 当文件类型改变时更新加载工具和解析选项
  useEffect(() => {
    const tools = getLoadingTools();
    const options = getParsingOptions();
    setLoadingTools(tools);
    setParsingOptions(options);
    
    if (tools.length > 0 && !tools.find(t => t.value === loadingMethod)) {
      setLoadingMethod(tools[0].value);
    }
    
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
      setCurrentStep(1);
      
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
      message.error('请选择所有必需的选项');
      return;
    }

    setStatus('处理中...');
    setIsProcessing(true);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('loading_method', loadingMethod);
      formData.append('parsing_option', parsingOption);
      formData.append('file_type', fileType);

      const response = await fetch(`${apiBaseUrl}/upload`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setStatus('处理完成！');
      message.success('文件处理成功！');
      setCurrentStep(2);
      fetchParsedFiles();
    } catch (error) {
      console.error('Error:', error);
      setStatus(`错误: ${error.message}`);
      message.error('文件处理失败：' + error.message);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setFileType('');
    setLoadingMethod('');
    setParsingOption('');
    setStatus('');
    setIsProcessing(false);
    setCurrentStep(0);
    // 重置文件输入
    const fileInput = document.querySelector('input[type="file"]');
    if (fileInput) {
      fileInput.value = '';
    }
  };

  const handlePreview = async (file) => {
    try {
      setLoading(true);
      // 只处理加载后的文件
      const type = 'loaded';
      const fileName = file.id.endsWith('.json') ? file.id : `${file.id}.json`;
      
      const response = await fetch(`${apiBaseUrl}/documents/${fileName}?type=${type}`);
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`获取文件内容失败: ${response.status} - ${errorText}`);
      }
      const data = await response.json();
      setPreviewFile(data);
      setPreviewVisible(true);
    } catch (error) {
      console.error('预览文件失败:', error);
      message.error(`预览文件失败: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = (file) => {
    Modal.confirm({
      title: '确认删除',
      content: `确定要删除文件 "${file.filename}" 吗？`,
      okText: '确认',
      okType: 'danger',
      cancelText: '取消',
      onOk: async () => {
        try {
          // 使用正确的API路径
          const response = await fetch(`${apiBaseUrl}/files/loaded/${file.id}`, {
            method: 'DELETE'
          });
          if (!response.ok) {
            throw new Error('删除文件失败');
          }
          message.success('文件删除成功');
          fetchParsedFiles();
        } catch (error) {
          message.error(error.message);
        }
      }
    });
  };

  // 表格列定义
  const columns = [
    {
      title: '文件名',
      dataIndex: 'filename',
      key: 'filename',
      width: 200,
      render: (text) => (
        <div className="max-w-[200px] break-words whitespace-normal">
          <span className="font-medium">{text || '未命名文件'}</span>
        </div>
      )
    },
    {
      title: '文件类型',
      dataIndex: 'file_type',
      key: 'file_type',
      render: (type) => {
        if (!type) return <Tag>未知</Tag>;
        return (
          <Tag color={
            type === 'pdf' ? 'blue' :
            type === 'netlist' ? 'green' :
            type === 'lef' ? 'purple' :
            type === 'lib' ? 'orange' : 'default'
          }>
            {type.toUpperCase()}
          </Tag>
        );
      }
    },
    {
      title: '加载方法',
      dataIndex: 'loading_method',
      key: 'loading_method',
      render: (method) => <Tag>{method || '未知'}</Tag>
    },
    {
      title: '处理时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      render: (timestamp) => timestamp ? new Date(timestamp).toLocaleString() : '未知'
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
      )
    }
  ];

  // 过滤文件列表
  const filteredFiles = parsedFiles.filter(file => {
    if (!file) return false;
    const searchLower = searchText.toLowerCase();
    return (
      (file.filename?.toLowerCase() || '').includes(searchLower) ||
      (file.file_type?.toLowerCase() || '').includes(searchLower) ||
      (file.parsing_method?.toLowerCase() || '').includes(searchLower)
    );
  }).map(file => ({
    ...file,
    key: file.id || file.filename || Math.random().toString(36).substr(2, 9)
  }));

  return (
    <div className="p-6">
      <h2 className="text-2xl font-bold mb-6">文件处理</h2>
      
      <div className="grid grid-cols-12 gap-6">
        {/* 左侧面板 (3/12) */}
        <div className="col-span-3 space-y-4">
          <div className="p-4 border rounded-lg bg-white shadow-sm">
            <Steps
              current={currentStep}
              direction="vertical"
              items={[
                {
                  title: '选择文件',
                  icon: <UploadOutlined />
                },
                {
                  title: '配置选项',
                  icon: <FileTextOutlined />
                },
                {
                  title: '完成',
                  icon: <CheckCircleOutlined />
                }
              ]}
              className="mb-6"
            />

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

            {currentStep >= 1 && (
              <>
                <div className="mt-4">
                  <label className="block text-sm font-medium mb-1">加载工具</label>
                  <Select
                    value={loadingMethod}
                    onChange={setLoadingMethod}
                    className="w-full"
                    options={loadingTools}
                    disabled={isProcessing}
                  />
                </div>

                <div className="mt-4">
                  <label className="block text-sm font-medium mb-1">解析选项</label>
                  <Select
                    value={parsingOption}
                    onChange={setParsingOption}
                    className="w-full"
                    options={parsingOptions}
                    disabled={isProcessing}
                  />
                </div>

                <div className="mt-4 space-y-2">
                  <Button
                    type="primary"
                    onClick={handleProcess}
                    loading={isProcessing}
                    disabled={!file || isProcessing}
                    block
                  >
                    {isProcessing ? '处理中...' : '处理文件'}
                  </Button>
                  
                  <Button
                    onClick={handleReset}
                    disabled={isProcessing}
                    block
                  >
                    重置
                  </Button>
                </div>

                {status && (
                  <div className="mt-4 p-2 text-sm rounded">
                    <p className={isProcessing ? 'text-blue-600' : 'text-green-600'}>
                      {status}
                    </p>
                  </div>
                )}
              </>
            )}
          </div>
        </div>

        {/* 右侧面板 (9/12) */}
        <div className="col-span-9 space-y-4">
          {/* 已处理文件列表 */}
          <div className="border rounded-lg bg-white shadow-sm">
            <div className="p-4 border-b">
              <div className="flex justify-between items-center">
                <h3 className="text-lg font-semibold">已处理文件</h3>
                <Space>
                  <Input
                    placeholder="搜索文件..."
                    prefix={<SearchOutlined />}
                    value={searchText}
                    onChange={e => setSearchText(e.target.value)}
                    style={{ width: 200 }}
                  />
                  <Button
                    icon={<ReloadOutlined />}
                    onClick={fetchParsedFiles}
                    loading={loading}
                  />
                </Space>
              </div>
            </div>
            <Table
              columns={columns}
              dataSource={filteredFiles}
              rowKey="key"
              loading={loading}
              pagination={{
                defaultPageSize: 10,
                showSizeChanger: true,
                showTotal: (total) => `共 ${total} 个文件`
              }}
            />
          </div>
        </div>
      </div>

      {/* 文件预览模态框 */}
      <Modal
        title="文件预览"
        open={previewVisible}
        onCancel={() => setPreviewVisible(false)}
        width={800}
        footer={null}
      >
        {previewFile && (
          <div className="max-h-[calc(100vh-200px)] overflow-y-auto">
            <div className="mb-4 p-3 border rounded bg-gray-100">
              <h4 className="font-medium mb-2">文档信息</h4>
              <div className="text-sm text-gray-600">
                <p>文件名: {previewFile.metadata?.filename || previewFile.filename}</p>
                <p>处理方法: {previewFile.metadata?.parsing_method || previewFile.parsing_method}</p>
                <p>时间戳: {(previewFile.metadata?.timestamp || previewFile.timestamp) && 
                  new Date(previewFile.metadata?.timestamp || previewFile.timestamp).toLocaleString()}</p>
              </div>
            </div>
            <div className="space-y-3">
              {Array.isArray(previewFile.content) ? (
                previewFile.content.map((item, idx) => (
                  <div key={idx} className="p-4 border rounded-lg bg-white shadow-sm">
                    <h3 className="text-lg font-semibold mb-2">{item.type}</h3>
                    <p className="text-sm text-gray-500">{item.content}</p>
                  </div>
                ))
              ) : (
                <div className="p-4 border rounded-lg bg-white shadow-sm">
                  <pre className="whitespace-pre-wrap text-sm">{JSON.stringify(previewFile, null, 2)}</pre>
                </div>
              )}
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
};

export default FileProcessor; 
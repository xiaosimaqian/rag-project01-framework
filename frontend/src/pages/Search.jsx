// src/pages/Search.jsx
import React, { useState, useEffect } from 'react';
import RandomImage from '../components/RandomImage';
import { apiBaseUrl } from '../config/config';
import { Button, Input, Space, Table, Tag, message, Select, InputNumber } from 'antd';
import { SearchOutlined, ReloadOutlined } from '@ant-design/icons';

const Search = () => {
  const [query, setQuery] = useState('');
  const [collection, setCollection] = useState('');
  const [results, setResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [topK, setTopK] = useState(5);
  const [threshold, setThreshold] = useState(0.7);
  const [collections, setCollections] = useState([]);
  const [selectedProvider, setSelectedProvider] = useState('milvus');
  const [wordCountThreshold, setWordCountThreshold] = useState(100);
  const [saveResults, setSaveResults] = useState(false);
  const [status, setStatus] = useState('');
  const [searchText, setSearchText] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [loading, setLoading] = useState(false);

  // 定义支持的向量数据库
  const providers = [
    { id: 'milvus', name: 'Milvus' },
    { id: 'chroma', name: 'ChromaDB' }
  ];

  // 加载collections
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

  useEffect(() => {
    fetchCollections();
  }, [selectedProvider]);

  const handleSearch = async () => {
    if (!searchText.trim()) {
      message.warning('请输入搜索内容');
      return;
    }

    if (!collection) {
      message.warning('请选择要搜索的集合');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch(`${apiBaseUrl}/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: searchText,
          collection_name: collection,
          top_k: topK,
          threshold: threshold,
        }),
      });

      if (!response.ok) {
        throw new Error('搜索失败');
      }

      const data = await response.json();
      console.log('搜索响应内容:', data);
      if (data.status === 'success' && Array.isArray(data.results)) {
        setSearchResults(data.results);
        setResults(data.results);
        message.success(data.message);
      } else {
        throw new Error('搜索结果格式错误');
      }
    } catch (error) {
      message.error(error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSaveResults = async () => {
    if (!results.length) {
      setStatus('没有可保存的搜索结果');
      return;
    }

    try {
      const saveParams = {
        query: searchText,
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
      message.success('搜索结果保存成功');
    } catch (error) {
      console.error('保存错误:', error);
      setStatus(`保存失败: ${error.message}`);
      message.error('保存失败: ' + error.message);
    }
  };

  // 聚合所有模块名
  const moduleNames = Array.from(new Set(results
    .map(item => item.source?.module_name)
    .filter(Boolean)
  ));

  // 只显示模块声明chunk的筛选
  const [showModuleDeclOnly, setShowModuleDeclOnly] = useState(false);
  const filteredResults = showModuleDeclOnly
    ? results.filter(item => item.source?.is_module_decl)
    : results;

  // 表格列定义
  const columns = [
    {
      title: '文件名',
      dataIndex: 'file_name',
      key: 'file_name',
      render: (_, record) => record.source?.file_name || record.source?.document_name || '未知',
    },
    {
      title: '页码',
      dataIndex: 'page_number',
      key: 'page_number',
      render: (_, record) => record.source?.page_number ?? '未知',
    },
    {
      title: '块号',
      dataIndex: 'chunk_id',
      key: 'chunk_id',
      render: (_, record) => record.source?.chunk_id ?? '未知',
    },
    {
      title: '相似度',
      dataIndex: 'score',
      key: 'score',
      render: (score) => (score ? (score * 100).toFixed(2) + '%' : '0.00%'),
    },
    {
      title: '内容',
      dataIndex: 'content',
      key: 'content',
    },
    {
      title: '来源',
      key: 'source',
      render: (_, record) => (
        <>
          文件名: {record.source?.file_name || record.source?.document_name || '未知'}<br />
          页码: {record.source?.page_number ?? '未知'}<br />
          块号: {record.source?.chunk_id ?? '未知'}
        </>
      ),
    },
  ];

  return (
    <div className="p-6">
      <h2 className="text-2xl font-bold mb-6">语义搜索</h2>
      <div className="mb-4">
        <b>本次检索涉及模块：</b>
        {moduleNames.length > 0 ? moduleNames.join('，') : '无'}
        <Button
          style={{ marginLeft: 16 }}
          onClick={() => setShowModuleDeclOnly(v => !v)}
        >
          {showModuleDeclOnly ? '显示全部结果' : '只看模块声明'}
        </Button>
        <Button
          type="primary"
          style={{ marginLeft: 16 }}
          onClick={handleSaveResults}
          disabled={!results.length}
        >
          保存搜索结果
        </Button>
      </div>
      <div className="space-y-6">
        {/* 集合选择 */}
        <div className="flex space-x-4">
          <Select
            placeholder="选择要搜索的集合"
            value={collection}
            onChange={setCollection}
            style={{ width: 300 }}
            loading={loading}
          >
            {collections.map((col) => (
              <Select.Option key={col.id} value={col.id}>
                {col.name} ({col.count} 条记录)
              </Select.Option>
            ))}
          </Select>
          <Button
            icon={<ReloadOutlined />}
            onClick={() => fetchCollections()}
            loading={loading}
          >
            刷新集合列表
          </Button>
        </div>

        {/* 搜索框 */}
        <div className="flex space-x-4">
          <Input
            id="search-page-search-input"
            placeholder="输入搜索内容..."
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
            onPressEnter={handleSearch}
            size="large"
            className="flex-1"
          />
          <InputNumber
            min={1}
            max={16384}  // 修改为 Milvus 支持的最大值
            defaultValue={5}
            value={topK}
            onChange={setTopK}
            style={{ width: 100 }}
            size="large"
          />
          <InputNumber
            min={0}
            max={1}
            step={0.1}
            defaultValue={0.7}
            value={threshold}
            onChange={setThreshold}
            style={{ width: 100 }}
            size="large"
            placeholder="相似度阈值"
          />
          <Button
            type="primary"
            icon={<SearchOutlined />}
            onClick={handleSearch}
            loading={loading}
            size="large"
          >
            搜索
          </Button>
        </div>

        {/* 搜索结果 */}
        <div className="border rounded-lg bg-white shadow-sm">
          <div className="p-4 border-b">
            <h3 className="text-lg font-semibold">搜索结果</h3>
          </div>
          <Table
            columns={columns}
            dataSource={filteredResults}
            rowKey={(record) => record.id || record.source?.chunk_id || Math.random().toString(36).substr(2, 9)}
            loading={loading}
            pagination={{
              pageSize: 10,
              showSizeChanger: true,
              pageSizeOptions: [5, 10, 20, 50, 100],
              showTotal: (total) => `共 ${total} 条结果`,
            }}
          />
        </div>
      </div>
    </div>
  );
};

export default Search;
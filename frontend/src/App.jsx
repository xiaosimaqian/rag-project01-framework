// src/App.jsx
import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { Layout, Menu } from 'antd';
import {
  FileTextOutlined,
  AppstoreOutlined,
  SearchOutlined,
  DatabaseOutlined,
  FileSearchOutlined,
  RobotOutlined,
  CodeOutlined,
  FileAddOutlined,
  ScissorOutlined,
  NodeIndexOutlined,
} from '@ant-design/icons';
import FileProcessor from './pages/FileProcessor';
import Chunking from './pages/Chunking';
import EmbeddingFile from './pages/EmbeddingFile';
import Indexing from './pages/Indexing';
import Search from './pages/Search';
import Generation from './pages/Generation';
import NetlistGraph from './pages/NetlistGraph';
import Collections from './pages/Collections';

const { Header, Content, Sider } = Layout;

const AppContent = () => {
  const location = useLocation();
  
  const menuItems = [
    { key: "/", icon: <FileTextOutlined />, label: <Link to="/">文件处理</Link> },
    { key: "/chunk", icon: <ScissorOutlined />, label: <Link to="/chunk">分块</Link> },
    { key: "/embedding", icon: <DatabaseOutlined />, label: <Link to="/embedding">嵌入</Link> },
    { key: "/indexing", icon: <AppstoreOutlined />, label: <Link to="/indexing">索引</Link> },
    { key: "/search", icon: <SearchOutlined />, label: <Link to="/search">搜索</Link> },
    { key: "/generation", icon: <RobotOutlined />, label: <Link to="/generation">生成</Link> },
    { key: "/netlist-graph", icon: <NodeIndexOutlined />, label: <Link to="/netlist-graph">网表图</Link> },
    { key: "/collections", icon: <FileSearchOutlined />, label: <Link to="/collections">集合</Link> },
  ];

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header style={{ padding: 0, background: '#fff' }}>
        <div style={{ padding: '0 24px', display: 'flex', alignItems: 'center' }}>
          <h1 style={{ margin: 0, fontSize: '20px' }}>RAG Framework</h1>
        </div>
      </Header>
      <Layout>
        <Sider width={200} style={{ background: '#fff' }}>
          <Menu
            mode="inline"
            selectedKeys={[location.pathname]}
            style={{ height: '100%', borderRight: 0 }}
            items={menuItems}
          />
        </Sider>
        <Layout style={{ padding: '24px' }}>
          <Content style={{ background: '#fff', padding: 24, margin: 0, minHeight: 280 }}>
            <Routes>
              <Route path="/" element={<FileProcessor />} />
              <Route path="/chunk" element={<Chunking />} />
              <Route path="/embedding" element={<EmbeddingFile />} />
              <Route path="/indexing" element={<Indexing />} />
              <Route path="/search" element={<Search />} />
              <Route path="/generation" element={<Generation />} />
              <Route path="/netlist-graph" element={<NetlistGraph />} />
              <Route path="/collections" element={<Collections />} />
            </Routes>
          </Content>
        </Layout>
      </Layout>
    </Layout>
  );
};

const App = () => {
  return (
    <Router future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <AppContent />
    </Router>
  );
};

export default App;
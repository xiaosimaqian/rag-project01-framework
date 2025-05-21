// src/components/Sidebar.jsx
import React from 'react';
import { NavLink } from 'react-router-dom';

const Sidebar = () => {
  return (
    <div className="fixed left-0 top-0 h-full w-64 bg-gray-800 p-6">
      <div className="mb-8">
        <h1 className="text-xl font-semibold text-white">RAG Framework</h1>
      </div>
      
      <nav>
        <ul className="space-y-1">
          <li>
            <NavLink 
              to="/load-file"
              className={({ isActive }) => 
                `block px-4 py-2 rounded-md ${
                  isActive 
                    ? 'bg-gray-700 text-white' 
                    : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                }`
              }
            >
              加载文件
            </NavLink>
          </li>
          <li>
            <NavLink 
              to="/parse-file"
              className={({ isActive }) => 
                `block px-4 py-2 rounded-md ${
                  isActive 
                    ? 'bg-gray-700 text-white' 
                    : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                }`
              }
            >
              解析文件
            </NavLink>
          </li>
          <li>
            <NavLink 
              to="/chunk-file"
              className={({ isActive }) => 
                `block px-4 py-2 rounded-md ${
                  isActive 
                    ? 'bg-gray-700 text-white' 
                    : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                }`
              }
            >
              分块文件
            </NavLink>
          </li>
          <li>
            <NavLink 
              to="/embedding"
              className={({ isActive }) => 
                `block px-4 py-2 rounded-md ${
                  isActive 
                    ? 'bg-gray-700 text-white' 
                    : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                }`
              }
            >
              向量化
            </NavLink>
          </li>
          <li>
            <NavLink 
              to="/indexing"
              className={({ isActive }) => 
                `block px-4 py-2 rounded-md ${
                  isActive 
                    ? 'bg-gray-700 text-white' 
                    : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                }`
              }
            >
              索引
            </NavLink>
          </li>
          <li>
            <NavLink 
              to="/search"
              className={({ isActive }) => 
                `block px-4 py-2 rounded-md ${
                  isActive 
                    ? 'bg-gray-700 text-white' 
                    : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                }`
              }
            >
              搜索
            </NavLink>
          </li>
          <li>
            <NavLink 
              to="/generation"
              className={({ isActive }) => 
                `block px-4 py-2 rounded-md ${
                  isActive 
                    ? 'bg-gray-700 text-white' 
                    : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                }`
              }
            >
              生成
            </NavLink>
          </li>
          <li>
            <NavLink 
              to="/netlist-graph"
              className={({ isActive }) => 
                `block px-4 py-2 rounded-md ${
                  isActive 
                    ? 'bg-gray-700 text-white' 
                    : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                }`
              }
            >
              网表图分析
            </NavLink>
          </li>
        </ul>
      </nav>
    </div>
  );
};

export default Sidebar;
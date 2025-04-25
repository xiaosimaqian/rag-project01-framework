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
              Load File
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
              Parse File
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
              Chunk File
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
              Embedding
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
              Indexing
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
              Search
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
              Generation
            </NavLink>
          </li>
        </ul>
      </nav>
    </div>
  );
};

export default Sidebar;
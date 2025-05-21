import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  server: {
    host: '0.0.0.0',  // 允许外部 IP 访问
    port: 5174,
    strictPort: true, // 如果端口被占用，则直接退出
    hmr: {
      overlay: true,
      port: 5174 // 固定 HMR 端口
    },
    cors: true,  // 启用 CORS
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, PATCH, OPTIONS',
      'Access-Control-Allow-Headers': 'X-Requested-With, content-type, Authorization'
    },
    proxy: {
      '/api': {
        target: 'http://localhost:8001',
        changeOrigin: true,
        timeout: 300000 // 增加超时时间到 300000 毫秒 (5 分钟)
      }
    }
  },
  plugins: [react()],
  optimizeDeps: {
    include: ['react', 'react-dom', 'd3', 'antd']
  },
  build: {
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          'react-vendor': ['react', 'react-dom'],
          'antd-vendor': ['antd'],
          'd3-vendor': ['d3']
        }
      }
    }
  }
})

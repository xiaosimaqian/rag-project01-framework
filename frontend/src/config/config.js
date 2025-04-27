// 使用 Vite 的环境变量
const env = import.meta.env.MODE || 'development';

// API配置
const config = {
  development: {
    apiBaseUrl: 'http://localhost:8001'  // 使用正确的端口8001
  },
  production: {
    apiBaseUrl: 'http://api.example.com'
  },
  test: {
    apiBaseUrl: 'http://localhost:8001'  // 使用正确的端口8001
  }
};

// 导出当前环境的配置
const apiBaseUrl = config[env].apiBaseUrl;

console.log('Current MODE:', env);
console.log('API Base URL:', apiBaseUrl);
console.log('All env variables:', import.meta.env);

export { apiBaseUrl };
export default config[env];
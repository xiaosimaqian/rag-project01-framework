// 使用 Vite 的环境变量
const env = import.meta.env.MODE || 'development';

// API配置
const config = {
              development: {
    apiBaseUrl: '/api'  // 使用代理路径
              },
              production: {
                apiBaseUrl: 'http://api.example.com'
              },
              test: {
    apiBaseUrl: '/api'  // 使用代理路径
              }
            };
            
// 导出当前环境的配置
const apiBaseUrl = config[env].apiBaseUrl;

console.log('Current MODE:', env);
console.log('API Base URL:', apiBaseUrl);
console.log('All env variables:', import.meta.env);

export { apiBaseUrl };
export default config[env];
# `.\DB-GPT-src\web\utils\ctx-axios.ts`

```py
# 导入 axios 模块
import axios from 'axios';

# 创建一个名为 api 的 axios 实例，设置基础 URL 为环境变量 API_BASE_URL 的值
const api = axios.create({
  baseURL: process.env.API_BASE_URL,
});

# 设置 api 实例的默认超时时间为 10000 毫秒
api.defaults.timeout = 10000;

# 添加响应拦截器，对响应数据进行处理
api.interceptors.response.use(
  # 如果响应成功，返回响应数据的 data 字段
  response => response.data,
  # 如果响应失败，返回一个带有失败原因的 Promise
  err => Promise.reject(err)
);

# 导出 api 实例
export default api;
```
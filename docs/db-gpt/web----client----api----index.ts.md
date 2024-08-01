# `.\DB-GPT-src\web\client\api\index.ts`

```py
# 导入 axios 库，同时引入 AxiosRequestConfig、AxiosError 和 AxiosResponse 类型
import axios, { AxiosRequestConfig, AxiosError, AxiosResponse } from 'axios';

# 定义通用的响应类型 ResponseType，包含数据 data、错误码 err_code、错误信息 err_msg 和成功标志 success
export type ResponseType<T = any> = {
  data: T;
  err_code: string | null;
  err_msg: string | null;
  success: boolean;
};

# 定义泛型类型 ApiResponse，它是 AxiosResponse 的 ResponseType<T> 版本
export type ApiResponse<T = any, D = any> = AxiosResponse<ResponseType<T>, D>;

# 定义成功的元组类型 SuccessTuple，包含错误（null）、数据、ResponseType<T> 和 ApiResponse<T, D>
export type SuccessTuple<T = any, D = any> = [null, T, ResponseType<T>, ApiResponse<T, D>];

# 定义失败的元组类型 FailedTuple，包含错误（Error 或 AxiosError<T, D>）、空数据和四个 null 值
export type FailedTuple<T = any, D = any> = [Error | AxiosError<T, D>, null, null, null];

# 创建一个 axios 实例 ins，基础 URL 从环境变量 process.env.API_BASE_URL 获取，默认为空字符串
const ins = axios.create({
  baseURL: process.env.API_BASE_URL ?? '',
});

# 长时间 API 列表，这些 API 在请求时会使用较长的超时时间
const LONG_TIME_API: string[] = [
  '/db/add',
  '/db/test/connect',
  '/db/summary',
  '/params/file/load',
  '/chat/prepare',
  '/model/start',
  '/model/stop',
  '/editor/sql/run',
  '/sql/editor/submit',
  '/editor/chart/run',
  '/chart/editor/submit',
  '/document/upload',
  '/document/sync',
  '/agent/install',
  '/agent/uninstall',
  '/personal/agent/upload',
];

# 请求拦截器，用于处理请求配置
ins.interceptors.request.use((request) => {
  # 检查当前请求是否属于长时间 API
  const isLongTimeApi = LONG_TIME_API.some((item) => request.url && request.url.indexOf(item) >= 0);
  # 如果请求没有设置超时时间，根据是否是长时间 API 设置超时时间为 10 秒或 60 秒
  if (!request.timeout) {
    request.timeout = isLongTimeApi ? 60000 : 10000;
  }
  # 返回处理后的请求配置
  return request;
});

# 导出 GET 方法，用于发起 GET 请求，支持泛型参数 Params（请求参数）、Response（响应数据）和 D（AxiosRequestConfig 的 data 类型）
export const GET = <Params = any, Response = any, D = any>(url: string, params?: Params, config?: AxiosRequestConfig<D>) => {
  return ins.get<Params, ApiResponse<Response>>(url, { params, ...config });
};

# 导出 POST 方法，用于发起 POST 请求，支持泛型参数 Data（请求数据）、Response（响应数据）和 D（AxiosRequestConfig 的 data 类型）
export const POST = <Data = any, Response = any, D = any>(url: string, data?: Data, config?: AxiosRequestConfig<D>) => {
  return ins.post<Data, ApiResponse<Response>>(url, data, config);
};

# 导出 PATCH 方法，用于发起 PATCH 请求，支持泛型参数 Data（请求数据）、Response（响应数据）和 D（AxiosRequestConfig 的 data 类型）
export const PATCH = <Data = any, Response = any, D = any>(url: string, data?: Data, config?: AxiosRequestConfig<D>) => {
  return ins.patch<Data, ApiResponse<Response>>(url, data, config);
};

# 导出 PUT 方法，用于发起 PUT 请求，支持泛型参数 Data（请求数据）、Response（响应数据）和 D（AxiosRequestConfig 的 data 类型）
export const PUT = <Data = any, Response = any, D = any>(url: string, data?: Data, config?: AxiosRequestConfig<D>) => {
  return ins.put<Data, ApiResponse<Response>>(url, data, config);
};

# 导出 DELETE 方法，用于发起 DELETE 请求，支持泛型参数 Params（请求参数）、Response（响应数据）和 D（AxiosRequestConfig 的 data 类型）
export const DELETE = <Params = any, Response = any, D = any>(url: string, params?: Params, config?: AxiosRequestConfig<D>) => {
  return ins.delete<Params, ApiResponse<Response>>(url, { params, ...config });
};

# 导出所有来自 tools 模块的内容
export * from './tools';

# 导出所有来自 request 模块的内容
export * from './request';
```
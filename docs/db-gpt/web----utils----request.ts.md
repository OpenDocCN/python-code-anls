# `.\DB-GPT-src\web\utils\request.ts`

```py
// 引入antd库中的message组件，用于显示错误信息
import { message } from 'antd';
// 引入ctx-axios模块，作为axios的自定义实例
import axios from './ctx-axios';
// 引入lodash库中的isPlainObject函数，用于检查对象是否是普通对象
import { isPlainObject } from 'lodash';

// 默认的请求头设置为JSON格式
const DEFAULT_HEADERS = {
  'content-type': 'application/json',
};

// 对请求体中的字符串字段进行trim操作的函数
const sanitizeBody = (obj: Record<string, any>): string => {
  // 如果不是普通对象，则直接将其转换为JSON字符串并返回
  if (!isPlainObject(obj)) return JSON.stringify(obj);
  // 使用浅拷贝创建一个结果对象，以避免修改原始对象
  const resObj = { ...obj };
  // 遍历结果对象的每个键值对
  for (const key in resObj) {
    const val = resObj[key];
    // 如果值是字符串类型，则对其进行trim操作
    if (typeof val === 'string') {
      resObj[key] = val.trim();
    }
  }
  // 返回处理后的JSON字符串
  return JSON.stringify(resObj);
};

// 发送GET请求的函数，支持传入查询字符串参数
export const sendGetRequest = (url: string, qs?: { [key: string]: any }) => {
  // 如果存在查询字符串参数qs，则将其转换为URL查询参数格式
  if (qs) {
    const str = Object.keys(qs)
      .filter((k) => qs[k] !== undefined && qs[k] !== '')
      .map((k) => `${k}=${qs[k]}`)
      .join('&');
    // 如果生成的查询参数字符串不为空，则附加在URL后面
    if (str) {
      url += `?${str}`;
    }
  }
  // 使用axios发起GET请求到指定的API地址，并返回响应数据或处理错误
  return axios
    .get<null, any>('/api' + url, {
      headers: DEFAULT_HEADERS,
    })
    .then((res) => res)
    .catch((err) => {
      // 发生错误时，在页面上显示错误信息
      message.error(err);
      // 返回一个被拒绝的Promise，将错误继续传播
      Promise.reject(err);
    });
};

// 发送带空格GET请求的函数，支持传入查询字符串参数
export const sendSpaceGetRequest = (url: string, qs?: { [key: string]: any }) => {
  // 如果存在查询字符串参数qs，则将其转换为URL查询参数格式
  if (qs) {
    const str = Object.keys(qs)
      .filter((k) => qs[k] !== undefined && qs[k] !== '')
      .map((k) => `${k}=${qs[k]}`)
      .join('&');
    // 如果生成的查询参数字符串不为空，则附加在URL后面
    if (str) {
      url += `?${str}`;
    }
  }
  // 使用axios发起GET请求到指定的URL地址，并返回响应数据或处理错误
  return axios
    .get<null, any>(url, {
      headers: DEFAULT_HEADERS,
    })
    .then((res) => res)
    .catch((err) => {
      // 发生错误时，在页面上显示错误信息
      message.error(err);
      // 返回一个被拒绝的Promise，将错误继续传播
      Promise.reject(err);
    });
};

// 发送POST请求的函数，支持传入请求体参数
export const sendPostRequest = (url: string, body?: any) => {
  // 对请求体进行trim操作，以去除字符串字段的多余空格
  const reqBody = sanitizeBody(body);
  // 使用axios发起POST请求到指定的API地址，并返回响应数据或处理错误
  return axios
    .post<null, any>('/api' + url, {
      body: reqBody,
      headers: DEFAULT_HEADERS,
    })
    .then((res) => res)
    .catch((err) => {
      // 发生错误时，在页面上显示错误信息
      message.error(err);
      // 返回一个被拒绝的Promise，将错误继续传播
      Promise.reject(err);
    });
};

// 发送带空格POST请求的函数，支持传入请求体参数
export const sendSpacePostRequest = (url: string, body?: any) => {
  // 使用axios发起POST请求到指定的URL地址，并返回响应数据或处理错误
  return axios
    .post<null, any>(url, body, {
      headers: DEFAULT_HEADERS,
    })
    .then((res) => res)
    .catch((err) => {
      // 发生错误时，在页面上显示错误信息
      message.error(err);
      // 返回一个被拒绝的Promise，将错误继续传播
      Promise.reject(err);
    });
};

// 发送带空格的上传POST请求的函数，支持传入请求体参数
export const sendSpaceUploadPostRequest = (url: string, body?: any) => {
  // 使用axios发起POST请求到指定的URL地址，并返回响应数据或处理错误
  return axios
    .post<null, any>(url, body)
    .then((res) => res)
    .catch((err) => {
      // 发生错误时，在页面上显示错误信息
      message.error(err);
      // 返回一个被拒绝的Promise，将错误继续传播
      Promise.reject(err);
    });
};
```
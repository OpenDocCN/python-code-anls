# `.\DB-GPT-src\web\components\layout\top-progress-bar.tsx`

```py
import Router from 'next/router';  // 导入 next.js 的路由器对象
import NProgress from 'nprogress';  // 导入进度条库 NProgress

let timer: any;  // 定义计时器变量
let state: any;  // 定义状态变量
let activeRequests = 0;  // 定义活跃的请求数量初始为 0
const delay = 250;  // 定义延迟时间为 250 毫秒

function load() {  // 定义加载函数
  if (state === 'loading') {  // 如果状态已经是加载中，直接返回
    return;
  }

  state = 'loading';  // 设置状态为加载中

  timer = setTimeout(function () {  // 设置定时器，在延迟后启动进度条
    NProgress.start();  // 启动进度条
  }, delay);  // 只有当延迟时间之后才显示进度条
}

function stop() {  // 定义停止函数
  if (activeRequests > 0) {  // 如果有活跃的请求，直接返回
    return;
  }

  state = 'stop';  // 设置状态为停止

  clearTimeout(timer);  // 清除定时器
  NProgress.done();  // 完成进度条显示
}

Router.events.on('routeChangeStart', load);  // 监听路由变化开始事件，调用 load 函数
Router.events.on('routeChangeComplete', stop);  // 监听路由变化完成事件，调用 stop 函数
Router.events.on('routeChangeError', stop);  // 监听路由变化出错事件，调用 stop 函数

if (typeof window !== 'undefined' && typeof window?.fetch === 'function') {  // 在浏览器环境中且支持 fetch 方法时
  const originalFetch = window.fetch;  // 保存原始的 fetch 方法
  window.fetch = async function (...args) {  // 重写全局的 fetch 方法
    if (activeRequests === 0) {  // 如果当前没有活跃的请求
      load();  // 调用 load 函数显示进度条
    }
  
    activeRequests++;  // 活跃请求数量加一
  
    try {
      const response = await originalFetch(...args);  // 调用原始的 fetch 方法发起请求
      return response;  // 返回响应对象
    } catch (error) {
      return Promise.reject(error);  // 发生错误时返回 Promise.reject
    } finally {
      activeRequests -= 1;  // 活跃请求数量减一
      if (activeRequests === 0) {  // 如果没有活跃的请求
        stop();  // 调用 stop 函数停止进度条显示
      }
    }
  };
}

export default function TopProgressBar() {  // 导出一个无内容的组件 TopProgressBar
  return null;  // 返回 null
}
```
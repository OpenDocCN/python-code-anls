# `.\DB-GPT-src\web\components\chat\ob-editor\ob-plugin.ts`

```py
import type Plugin from '@oceanbase-odc/monaco-plugin-ob';

let plugin: Plugin;

// 导出一个异步函数，用于注册插件并返回插件对象
export async function register(): Promise<Plugin> {
  // 将一个全局对象 obMonaco 注册到 window 上，定义了一个函数 getWorkerUrl
  window.obMonaco = {
    // 根据类型获取相应的 worker 脚本 URL
    getWorkerUrl: (type: string) => {
      switch (type) {
        // 如果类型是 'mysql'，返回 MySQL worker 脚本的 URL
        case 'mysql': {
          return location.origin + '/_next/static/ob-workers/mysql.js';
        }
        // 如果类型是 'obmysql'，返回 OBMySQL worker 脚本的 URL
        case 'obmysql': {
          return location.origin + '/_next/static/ob-workers/obmysql.js';
        }
        // 如果类型是 'oboracle'，返回 OBOracle worker 脚本的 URL
        case 'oboracle': {
          return location.origin + '/_next/static/ob-workers/oracle.js';
        }
      }
      // 默认情况下返回空字符串
      return "";
    }
  }

  // 异步导入 '@oceanbase-odc/monaco-plugin-ob' 模块
  const module = await import('@oceanbase-odc/monaco-plugin-ob');
  // 从模块中获取默认导出对象 Plugin
  const Plugin = module.default;

  // 如果插件已经存在，直接返回
  if (plugin) {
    return plugin;
  }

  // 创建新的 Plugin 对象并赋值给 plugin 变量
  plugin = new Plugin();
  // 设置插件的配置，这里传入了一个字符串数组 ['mysql']
  plugin.setup(["mysql"]);
  
  // 返回创建或已存在的插件对象
  return plugin;
}
```
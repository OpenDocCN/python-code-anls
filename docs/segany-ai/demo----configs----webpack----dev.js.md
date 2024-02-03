# `.\segment-anything\demo\configs\webpack\dev.js`

```py
// 引入 webpack-merge 模块，用于合并配置
const { merge } = require("webpack-merge");
// 引入通用配置
const commonConfig = require("./common");

// 导出合并后的开发环境配置
module.exports = merge(commonConfig, {
  // 设置模式为开发模式
  mode: "development",
  // 配置开发服务器
  devServer: {
    hot: true, // 在服务器上启用热模块替换
    open: true,
    // 这些头部信息启用跨域隔离状态，用于启用 SharedArrayBuffer 用于 ONNX 多线程
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "credentialless",
    },
  },
  // 配置 source map 为 cheap-module-source-map
  devtool: "cheap-module-source-map",
});
```
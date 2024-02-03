# `.\segment-anything\demo\configs\webpack\prod.js`

```
// 引入 webpack-merge 模块，用于合并配置
// 引入 path 模块的 resolve 方法，用于处理路径
// 引入 dotenv-webpack 模块，用于加载环境变量
// 引入 common 配置文件
const { merge } = require("webpack-merge");
const { resolve } = require("path");
const Dotenv = require("dotenv-webpack");
const commonConfig = require("./common");

// 导出合并后的生产环境配置
module.exports = merge(commonConfig, {
  // 设置模式为生产环境
  mode: "production",
  // 配置输出文件名和路径
  output: {
    filename: "js/bundle.[contenthash].min.js",
    path: resolve(__dirname, "../../dist"),
    publicPath: "/",
  },
  // 配置 source map 生成方式
  devtool: "source-map",
  // 添加加载环境变量的插件
  plugins: [new Dotenv()],
});
```
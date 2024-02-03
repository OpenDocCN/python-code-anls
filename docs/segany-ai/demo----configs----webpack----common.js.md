# `.\segment-anything\demo\configs\webpack\common.js`

```
// 引入所需的模块
const { resolve } = require("path");
const HtmlWebpackPlugin = require("html-webpack-plugin");
const FriendlyErrorsWebpackPlugin = require("friendly-errors-webpack-plugin");
const CopyPlugin = require("copy-webpack-plugin");
const webpack = require("webpack");

// 导出配置对象
module.exports = {
  // 指定入口文件
  entry: "./src/index.tsx",
  // 配置模块解析
  resolve: {
    // 指定文件扩展名
    extensions: [".js", ".jsx", ".ts", ".tsx"],
  },
  // 配置输出
  output: {
    // 指定输出路径
    path: resolve(__dirname, "dist"),
  },
  // 配置模块规则
  module: {
    rules: [
      {
        // 匹配 mjs 文件
        test: /\.mjs$/,
        // 只包含 node_modules 目录
        include: /node_modules/,
        // 指定类型为 javascript/auto
        type: "javascript/auto",
        // 配置解析选项
        resolve: {
          fullySpecified: false,
        },
      },
      {
        // 匹配 js、jsx、ts、tsx 文件
        test: [/\.jsx?$/, /\.tsx?$/],
        // 使用 ts-loader 处理
        use: ["ts-loader"],
        // 排除 node_modules 目录
        exclude: /node_modules/,
      },
      {
        // 匹配 css 文件
        test: /\.css$/,
        // 使用 style-loader 和 css-loader 处理
        use: ["style-loader", "css-loader"],
      },
      {
        // 匹配 scss 和 sass 文件
        test: /\.(scss|sass)$/,
        // 使用 style-loader、css-loader 和 postcss-loader 处理
        use: ["style-loader", "css-loader", "postcss-loader"],
      },
      {
        // 匹配 jpg、jpeg、png、gif、svg 文件
        test: /\.(jpe?g|png|gif|svg)$/i,
        // 使用 file-loader 和 image-webpack-loader 处理
        use: [
          "file-loader?hash=sha512&digest=hex&name=img/[contenthash].[ext]",
          "image-webpack-loader?bypassOnDebug&optipng.optimizationLevel=7&gifsicle.interlaced=false",
        ],
      },
      {
        // 匹配 woff、woff2、ttf 文件
        test: /\.(woff|woff2|ttf)$/,
        // 使用 url-loader 处理
        use: {
          loader: "url-loader",
        },
      },
    ],
  },
  // 配置插件
  plugins: [
    // 复制文件插件配置
    new CopyPlugin({
      patterns: [
        {
          from: "node_modules/onnxruntime-web/dist/*.wasm",
          to: "[name][ext]",
        },
        {
          from: "model",
          to: "model",
        },
        {
          from: "src/assets",
          to: "assets",
        },
      ],
    }),
    // HTML 模板插件配置
    new HtmlWebpackPlugin({
      template: "./src/assets/index.html",
    }),
    // 友好的错误提示插件
    new FriendlyErrorsWebpackPlugin(),
    // 使用 webpack.ProvidePlugin 插件，为所有模块提供一个 process 变量，指向 "process/browser" 模块
    new webpack.ProvidePlugin({
      process: "process/browser",
    }),
  ],
# 代码块结束的标志，表示一个代码块的结束
```
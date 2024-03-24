# `.\lucidrains\liquid-conway\webpack.config.js`

```
# 引入 Node.js 的 path 模块和 ExtractTextPlugin 插件
const path = require('path');
const ExtractTextPlugin = require('extract-text-webpack-plugin');

# 定义源代码目录和输出目录的绝对路径
const src = path.resolve(__dirname, 'src');
const dist = path.resolve(__dirname, 'dist');

# 配置对象，包括上下文、入口文件、输出文件、模块规则和插件
const config = {
  # 指定上下文为源代码目录
  context: src,
  # 配置入口文件，包括 regular 和 liquid 两个入口
  entry: {
    regular: './app.js',
    liquid: './liquid.js'
  },
  # 配置输出文件的路径和文件名
  output: {
    path: dist,
    filename: '[name].js'
  },
  # 配置模块规则，包括处理 js 文件和 css 文件的规则
  module: {
    rules: [{
      test: /\.js$/,
      include: src,
      use: [{
        loader: 'babel-loader',
        options: {
          presets: [
            ['es2015', { modules: false }]
          ]
        }
      }]
    }, {
      test: /\.css$/,
      use: ExtractTextPlugin.extract({
        fallback: 'style-loader',
        use: ['css-loader']
      })
    },
    {
      test: /\.*(sass|scss)$/,
      use: ExtractTextPlugin.extract({
        fallback: 'style-loader',
        use: ['css-loader', 'sass-loader']
      })
    }]
  },
  # 配置插件，使用 ExtractTextPlugin 插件生成样式文件
  plugins: [
    new ExtractTextPlugin('styles.css')
  ]
};

# 导出配置对象
module.exports = config;
```
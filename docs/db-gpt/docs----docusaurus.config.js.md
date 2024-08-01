# `.\DB-GPT-src\docs\docusaurus.config.js`

```py
// 引入 webpack 提供的 ProvidePlugin，用于自动加载模块
const { ProvidePlugin } = require("webpack");
// 引入 Node.js 的 path 模块，用于处理文件路径
const path = require("path");

// 从 prism-react-renderer 包中引入主题对象 themes
const {themes} = require('prism-react-renderer');
// 设置明亮和暗色代码主题
const lightCodeTheme = themes.github;
const darkCodeTheme = themes.dracula;

// 检查当前环境是否为开发模式
const isDev = process.env.NODE_ENV === "development";
// 检查是否启用了快速构建
const isBuildFast = !!process.env.BUILD_FAST;
// 检查是否禁用版本控制
const isVersioningDisabled = !!process.env.DISABLE_VERSIONING;

// 从 JSON 文件中读取版本信息
const versions = require("./versions.json");

// 打印版本信息到控制台
console.log("versions", versions)

// 检查版本是否为预发布版本
function isPrerelease(version) {
  return (
    version.includes('-') ||
    version.includes('alpha') ||
    version.includes('beta') ||
    version.includes('rc')
  );
}

// 获取最新的稳定版本
function getLastStableVersion() {
  const lastStableVersion = versions.find((version) => !isPrerelease(version));
  if (!lastStableVersion) {
    throw new Error('unexpected, no stable Docusaurus version?');
  }
  return lastStableVersion;
}

// 返回下一个版本名称
function getNextVersionName() {
  return 'dev';
}

/** @type {import('@docusaurus/types').Config} */
// Docusaurus 配置对象，指定站点的基本信息和设置
const config = {
  title: 'DB-GPT',
  tagline: 'Revolutionizing Database Interactions with Private LLM Technology',
  favicon: 'img/eosphoros.jpeg',

  // 设置站点的生产环境 URL
  url: 'http://docs.dbgpt.site',
  // 设置站点的根路径
  baseUrl: '/',

  // GitHub pages 部署配置
  // 如果不使用 GitHub pages，可以忽略这部分
  organizationName: 'eosphoros-ai', // GitHub 组织或用户名
  projectName: 'DB-GPT', // GitHub 仓库名

  // 设置链接失效时的处理方式
  onBrokenLinks: isDev ? 'throw' : 'warn',
  // 设置 Markdown 链接失效时的处理方式
  onBrokenMarkdownLinks: isDev ? 'throw' : 'warn',

  // 设置页面的语言国际化配置
  i18n: {
    defaultLocale: 'en', // 默认语言
    locales: ['en', 'zh-CN'], // 支持的语言列表
  },

  // 在页面中引入自定义脚本
  scripts: [
    {
      src: '/redirect.js', // 脚本文件的路径
      async: true, // 异步加载脚本
    },
  ],

  // Markdown 渲染配置，启用 mermaid 图表功能
  markdown: {
    mermaid: true,
  },

  // 配置使用的主题列表
  themes: [
    '@docusaurus/theme-mermaid',
    '@easyops-cn/docusaurus-search-local',
  ],

  // 配置使用的插件列表
  plugins: [
    () => ({
      // 返回一个对象，包含配置名称和配置Webpack的函数
      name: "custom-webpack-config",
      configureWebpack: () => ({
        // 配置Webpack的plugins选项，添加ProvidePlugin插件以提供process对象
        plugins: [
          new ProvidePlugin({
            process: require.resolve("process/browser"),
          }),
        ],
        // 配置Webpack的resolve选项，指定fallback为path和url，禁用它们的默认解析
        resolve: {
          fallback: {
            path: false,
            url: false,
          },
        },
        // 配置Webpack的module选项，定义规则用于处理不同类型的文件
        module: {
          rules: [
            {
              // 匹配所有以.mjs结尾的文件
              test: /\.m?js/,
              resolve: {
                // 不要强制要求模块具有完全指定的导入
                fullySpecified: false,
              },
            },
            {
              // 匹配所有以.py结尾的文件
              test: /\.py$/,
              // 使用raw-loader加载Python文件
              loader: "raw-loader",
              resolve: {
                // 不要强制要求模块具有完全指定的导入
                fullySpecified: false,
              },
            },
            {
              // 匹配所有以.ipynb结尾的文件
              test: /\.ipynb$/,
              // 使用raw-loader加载IPython笔记本文件
              loader: "raw-loader",
              resolve: {
                // 不要强制要求模块具有完全指定的导入
                fullySpecified: false,
              },
            },
          ],
        },
      }),
    }),
};

module.exports = config;
```
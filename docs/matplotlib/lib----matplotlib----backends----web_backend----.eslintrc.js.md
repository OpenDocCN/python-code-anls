# `D:\src\scipysrc\matplotlib\lib\matplotlib\backends\web_backend\.eslintrc.js`

```py
module.exports = {
  // 根对象导出，定义 ESLint 配置
  root: true,
  // 忽略以下文件夹下的所有文件
  ignorePatterns: ["jquery-ui-*/", "node_modules/"],
  // 设置环境为浏览器环境和支持 jQuery
  env: {
    browser: true,
    jquery: true,
  },
  // 继承推荐的 ESLint 标准规则和 Prettier 格式化插件
  extends: ["eslint:recommended", "prettier"],
  // 全局变量声明，指定为只读
  globals: {
    IPython: "readonly",
    MozWebSocket: "readonly",
  },
  // 规则配置
  rules: {
    // 缩进规则：2 个空格，switch 语句缩进风格
    indent: ["error", 2, { SwitchCase: 1 }],
    // 不允许未使用的变量，但允许以下划线开头的变量名
    "no-unused-vars": [
      "error",
      {
        argsIgnorePattern: "^_",
      },
    ],
    // 引号风格：双引号，避免转义
    quotes: ["error", "double", { avoidEscape: true }],
  },
  // 对指定文件进行特定配置覆盖
  overrides: [
    {
      // 匹配所有 js 文件夹下的 js 文件
      files: "js/**/*.js",
      // 覆盖规则：缩进改为 4 个空格，switch 语句缩进风格
      rules: {
        indent: ["error", 4, { SwitchCase: 1 }],
        // 引号风格：单引号，避免转义
        quotes: ["error", "single", { avoidEscape: true }],
      },
    },
  ],
};
```
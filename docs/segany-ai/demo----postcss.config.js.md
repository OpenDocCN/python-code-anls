# `.\segment-anything\demo\postcss.config.js`

```py
// 导入tailwindcss模块
const tailwindcss = require("tailwindcss");
// 导出PostCSS配置对象
module.exports = {
  // 使用postcss-preset-env插件
  plugins: ["postcss-preset-env", 'tailwindcss/nesting', tailwindcss],
};
```
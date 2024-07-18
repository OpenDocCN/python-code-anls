# `.\graphrag\docsite\.eleventy.js`

```py
const { EleventyHtmlBasePlugin } = require("@11ty/eleventy");
// 引入 EleventyHtmlBasePlugin 对象解构赋值

const syntaxHighlight = require("@11ty/eleventy-plugin-syntaxhighlight");
// 引入 syntaxHighlight 插件

const codeClipboard = require("eleventy-plugin-code-clipboard");
// 引入 codeClipboard 插件

const pluginMermaid = require("@kevingimbel/eleventy-plugin-mermaid");
// 引入 pluginMermaid 插件

const markdownIt = require('markdown-it');
// 引入 markdown-it 库

module.exports = (eleventyConfig) => {
  // 导出模块，传入 eleventyConfig 对象

  eleventyConfig.addPlugin(syntaxHighlight);
  // 向 eleventyConfig 中添加 syntaxHighlight 插件

  eleventyConfig.addPlugin(codeClipboard);
  // 向 eleventyConfig 中添加 codeClipboard 插件

  eleventyConfig.addPlugin(pluginMermaid);
  // 向 eleventyConfig 中添加 pluginMermaid 插件

  eleventyConfig.addPlugin(EleventyHtmlBasePlugin, {
    baseHref: process.env.DOCSITE_BASE_URL || ""
  });
  // 向 eleventyConfig 中添加 EleventyHtmlBasePlugin 插件，并配置 baseHref

  eleventyConfig.addPassthroughCopy("data");
  // 拷贝 "data" 目录到输出目录

  eleventyConfig.addPassthroughCopy("img");
  // 拷贝 "img" 目录到输出目录

  // Ignore auto-generated content
  // 设置不使用 Git 忽略文件列表
  eleventyConfig.setUseGitIgnore(false);

  const markdownLibrary = markdownIt({
    html: true
  }).use(codeClipboard.markdownItCopyButton);
  // 创建 markdownIt 实例，启用 HTML 解析，并使用 codeClipboard 插件

  eleventyConfig.setLibrary("md", markdownLibrary);
  // 配置 eleventyConfig 使用自定义的 markdown 解析库

};
```
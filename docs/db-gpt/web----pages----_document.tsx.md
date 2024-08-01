# `.\DB-GPT-src\web\pages\_document.tsx`

```py
import { createCache, StyleProvider } from '@ant-design/cssinjs';  // 导入需要的模块，从 '@ant-design/cssinjs' 中导入 createCache 和 StyleProvider
import Document, { DocumentContext, Head, Html, Main, NextScript } from 'next/document';  // 导入 Next.js 的 Document 类和相关组件
import { doExtraStyle } from '../genAntdCss';  // 从 '../genAntdCss' 导入 doExtraStyle 函数

class MyDocument extends Document {
  static async getInitialProps(ctx: DocumentContext) {
    const cache = createCache();  // 创建一个缓存对象
    let fileName = '';  // 初始化一个文件名为空字符串
    const originalRenderPage = ctx.renderPage;  // 保存原始的 renderPage 方法引用
    ctx.renderPage = () =>
      originalRenderPage({
        enhanceApp: (App) => (props) =>
          (
            <StyleProvider cache={cache} hashPriority="high">  {/* 使用 StyleProvider 提供的样式，传入缓存和哈希优先级 */}
              <App {...props} />
            </StyleProvider>
          ),
      });
    const initialProps = await Document.getInitialProps(ctx);  // 获取原始 Document 组件的初始 props

    fileName = doExtraStyle({  // 调用 doExtraStyle 函数，并获取返回的文件名
      cache,
    });

    return {
      ...initialProps,  // 返回原始的初始 props
      styles: (
        <>
          {initialProps.styles}  {/* 包含原始的样式 */}
          {/* 1.2 inject css */}
          {fileName && <link rel="stylesheet" href={`/${fileName}`} />}  {/* 如果 fileName 存在，则添加对应的样式表链接 */}
        </>
      ),
    };
  }

  render() {
    return (
      <Html lang="en">  {/* HTML 根元素，设置语言为英文 */}
        <Head>  {/* 文档头部 */}
          <link rel="icon" href="/favicon.ico" />  {/* 网站图标链接 */}
          <meta name="description" content="Revolutionizing Database Interactions with Private LLM Technology" />  {/* 网页描述 */}
          <meta property="og:site_name" content="dbgpt.site" />  {/* Open Graph 网站名称 */}
          <meta property="og:description" content="eosphoros-ai" />  {/* Open Graph 网页描述 */}
          <meta property="og:title" content="DB-GPT" />  {/* Open Graph 网页标题 */}
        </Head>
        <body>  {/* 文档主体 */}
          <Main />  {/* Next.js 中渲染主体内容的组件 */}
          <NextScript />  {/* Next.js 用于渲染 HTML 页面所需的脚本标签 */}
        </body>
      </Html>
    );
  }
}

export default MyDocument;  // 导出 MyDocument 类作为默认导出
```
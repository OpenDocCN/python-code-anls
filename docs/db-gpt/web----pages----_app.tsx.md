# `.\DB-GPT-src\web\pages\_app.tsx`

```py
import type { AppProps } from 'next/app';
import React, { useContext, useEffect, useRef } from 'react';
import SideBar from '@/components/layout/side-bar';
import TopProgressBar from '@/components/layout/top-progress-bar';
import { useTranslation } from 'react-i18next';
import { ChatContext, ChatContextProvider } from '@/app/chat-context';
import classNames from 'classnames';
import '../styles/globals.css';
import '../nprogress.css';
import '../app/i18n';
import { STORAGE_LANG_KEY } from '@/utils';
import { ConfigProvider, MappingAlgorithm, theme } from 'antd';
import zhCN from 'antd/locale/zh_CN';
import enUS from 'antd/locale/en_US';
import { CssVarsProvider, ThemeProvider, useColorScheme } from '@mui/joy';
import { joyTheme } from '@/defaultTheme';

// 定义 Ant Design 暗色主题的映射算法
const antdDarkTheme: MappingAlgorithm = (seedToken, mapToken) => {
  return {
    ...theme.darkAlgorithm(seedToken, mapToken),
    colorBgBase: '#232734',
    colorBorder: '#828282',
    colorBgContainer: '#232734',
  };
};

// 包裹 CSS 样式和国际化语言设置的组件
function CssWrapper({ children }: { children: React.ReactElement }) {
  const { mode } = useContext(ChatContext);  // 使用 ChatContext 中的 mode 状态
  const { i18n } = useTranslation();  // 使用 i18n 国际化翻译钩子
  const { setMode: setMuiMode } = useColorScheme();  // 使用 MUI 的颜色方案钩子

  // 当 mode 状态变化时设置 MUI 的主题模式
  useEffect(() => {
    setMuiMode(mode);
  }, [mode]);

  // 当 mode 状态变化时更新文档的 body 类以反映当前主题
  useEffect(() => {
    if (mode) {
      document.body?.classList?.add(mode);  // 添加当前 mode 类名到 body
      if (mode === 'light') {
        document.body?.classList?.remove('dark');  // 移除 'dark' 类名
      } else {
        document.body?.classList?.remove('light');  // 移除 'light' 类名
      }
    }
  }, [mode]);

  // 页面加载时，根据存储在本地的语言设置，更新 i18n 的语言
  useEffect(() => {
    i18n.changeLanguage && i18n.changeLanguage(window.localStorage.getItem(STORAGE_LANG_KEY) || 'en');
  }, [i18n]);

  // 返回带有顶部进度条和子元素的 div 容器
  return (
    <div>
      <TopProgressBar />  {/* 顶部进度条组件 */}
      {children}
    </div>
  );
}

// 页面布局组件，包含侧边栏和内容区域
function LayoutWrapper({ children }: { children: React.ReactNode }) {
  const { isMenuExpand, mode } = useContext(ChatContext);  // 使用 ChatContext 中的 isMenuExpand 和 mode 状态
  const { i18n } = useTranslation();  // 使用 i18n 国际化翻译钩子

  // 返回带有 Ant Design 配置和两个列的 div 容器
  return (
    <ConfigProvider
      locale={i18n.language === 'en' ? enUS : zhCN}  // 根据语言设置 Ant Design 的本地化语言
      theme={{  // 设置 Ant Design 的主题配置
        token: {
          colorPrimary: '#0069FE',  // 主色调设置为蓝色
          borderRadius: 4,  // 边框圆角半径为 4px
        },
        algorithm: mode === 'dark' ? antdDarkTheme : undefined,  // 根据当前主题模式选择映射算法
      }}
    >
      <div className="flex w-screen h-screen overflow-hidden">  {/* 弹性布局容器，占据整个屏幕 */}
        <div className={classNames('transition-[width]', isMenuExpand ? 'w-60' : 'w-20', 'hidden', 'md:block')}>
          <SideBar />  {/* 侧边栏组件 */}
        </div>
        <div className="flex flex-col flex-1 relative overflow-hidden">{children}</div>  {/* 弹性布局，包含子元素的垂直布局 */}
      </div>
    </ConfigProvider>
  );
}

// 主应用组件，包含 ChatContextProvider、ThemeProvider 和 CssVarsProvider 的根组件
function MyApp({ Component, pageProps }: AppProps) {
  return (
    <ChatContextProvider>  {/* 提供 ChatContext 的上下文 */}
      <ThemeProvider theme={joyTheme}>  {/* 提供 MUI 的主题 */}
        <CssVarsProvider theme={joyTheme} defaultMode="light">  {/* 提供 CSS 变量的主题和默认模式 */}
          <CssWrapper>  {/* CSS 样式和国际化设置的包裹组件 */}
            <LayoutWrapper>  {/* 页面布局包裹组件 */}
              <Component {...pageProps} />  {/* 渲染传入的页面组件 */}
            </LayoutWrapper>
          </CssWrapper>
        </CssVarsProvider>
      </ThemeProvider>
    </ChatContextProvider>
  );
}

export default MyApp;  // 导出主应用组件
```
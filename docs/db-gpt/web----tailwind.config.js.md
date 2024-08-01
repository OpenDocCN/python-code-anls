# `.\DB-GPT-src\web\tailwind.config.js`

```py
/** 
 * @type {import('tailwindcss').Config} 
 * 导出一个符合 Tailwind CSS 配置规范的对象
 */
module.exports = {
  content: ['./pages/**/*.{js,ts,jsx,tsx,mdx}', './components/**/*.{js,ts,jsx,tsx,mdx}', './app/**/*.{js,ts,jsx,tsx,mdx}'],
  // 指定需要用于提取 CSS 类的文件路径模式列表
  theme: {
    extend: {
      fontFamily: {
        // 扩展默认的 sans 字体系列，添加 "Josefin Sans" 字体
        sans: ['"Josefin Sans"', ...defaultTheme.fontFamily.sans],
      },
      colors: {
        // 扩展颜色主题，定义自定义主题色调
        theme: {
          primary: '#0069fe',           // 主要颜色
          light: '#f7f7f7',             // 浅色背景
          dark: '#151622',              // 深色背景
          'dark-container': '#232734',  // 深色容器背景
        },
      },
    },
  },
  important: true,    // 设置重要性以确保样式覆盖顺序
  darkMode: 'class',  // 启用根据 class 切换的暗色模式
  /**
   * @see https://www.tailwindcss-animated.com/configurator.html
   * 引入 tailwindcss-animated 插件，用于动画支持
   */
  plugins: [require('tailwindcss-animated')],
};
```
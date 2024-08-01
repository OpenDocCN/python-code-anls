# `.\DB-GPT-src\web\defaultTheme.ts`

```py
// 导入扩展主题函数从 '@mui/joy/styles' 模块
import { extendTheme } from '@mui/joy/styles';
// 导入颜色变量和调色板配置从 '@mui/joy/colors' 模块
import colors from '@mui/joy/colors';

// 定义 joyTheme 主题对象，使用 extendTheme 函数扩展 MUI 主题
export const joyTheme = extendTheme({
  // 定义不同颜色方案的配置
  colorSchemes: {
    // 亮色主题配置
    light: {
      palette: {
        // 设定主题模式为暗色
        mode: 'dark',
        primary: {
          // 使用 colors.grey 的颜色作为 primary 的基础
          ...colors.grey,
          // 设置不同的颜色变量
          solidBg: '#e6f4ff',
          solidColor: '#1677ff',
          solidHoverBg: '#e6f4ff',
        },
        neutral: {
          // 设置不同的中性颜色变量
          plainColor: '#4d4d4d',
          plainHoverColor: '#131318',
          plainHoverBg: '#EBEBEF',
          plainActiveBg: '#D8D8DF',
          plainDisabledColor: '#B9B9C6',
        },
        background: {
          // 设置不同的背景颜色变量
          body: '#F7F7F7',
          surface: '#fff',
        },
        text: {
          // 设置文本颜色变量
          primary: '#505050',
        },
      },
    },
    // 暗色主题配置
    dark: {
      palette: {
        // 设定主题模式为亮色
        mode: 'light',
        primary: {
          // 使用 colors.grey 的颜色作为 primary 的基础
          ...colors.grey,
          // 设置不同的颜色变量
          softBg: '#353539',
          softHoverBg: '#35353978',
          softDisabledBg: '#353539',
          solidBg: '#51525beb',
          solidHoverBg: '#51525beb',
        },
        neutral: {
          // 设置不同的中性颜色变量
          plainColor: '#D8D8DF',
          plainHoverColor: '#F7F7F8',
          plainHoverBg: '#353539',
          plainActiveBg: '#434356',
          plainDisabledColor: '#434356',
          outlinedBorder: '#353539',
          outlinedHoverBorder: '#454651',
        },
        text: {
          // 设置文本颜色变量
          primary: '#FDFDFC',
        },
        background: {
          // 设置不同的背景颜色变量
          body: '#151622',
          surface: '#51525beb',
        },
      },
    },
  },
  // 设定字体族变量
  fontFamily: {
    body: 'Josefin Sans, sans-serif',
    display: 'Josefin Sans, sans-serif',
  },
  // 设定 z-index 变量
  zIndex: {
    modal: 1001,
  },
});
```
# `.\AutoGPT\benchmark\frontend\tailwind.config.ts`

```py
# 导入 Config 类型从 "tailwindcss" 模块
import { type Config } from "tailwindcss";

# 导出默认配置对象
export default {
  # 指定需要处理的文件路径模式
  content: ["./src/**/*.{js,ts,jsx,tsx}"],
  # 主题配置，可以通过 extend 属性进行扩展
  theme: {
    extend: {},
  },
  # 插件列表
  plugins: [],
} satisfies Config;
```
# `.\DB-GPT-src\web\components\chat\chat-content\code-preview.tsx`

```py
# 导入 antd 库中的 Button 和 message 组件
# 导入 @ant-design/icons 库中的 CopyOutlined 图标
# 导入 react-syntax-highlighter 库中的 oneDark 和 coldarkDark 样式
# 导入 react-syntax-highlighter 库中的 Prism 和 SyntaxHighlighter 组件
# 导入 copy-to-clipboard 库中的 copy 函数
# 导入 react 库中的 CSSProperties 和 useContext 函数
# 导入 chat-context 文件中的 ChatContext 上下文
import { Button, message } from 'antd';
import { CopyOutlined } from '@ant-design/icons';
import { oneDark, coldarkDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import copy from 'copy-to-clipboard';
import { CSSProperties, useContext } from 'react';
import { ChatContext } from '@/app/chat-context';

# 定义 Props 接口，包含 code、language、customStyle、light 和 dark 属性
interface Props {
  code: string;
  language: string;
  customStyle?: CSSProperties;
  light?: { [key: string]: CSSProperties };
  dark?: { [key: string]: CSSProperties };
}

# 定义 CodePreview 组件，接受 Props 参数
export function CodePreview({ code, light, dark, language, customStyle }: Props) {
  # 使用 useContext 获取 ChatContext 上下文中的 mode 属性
  const { mode } = useContext(ChatContext);

  # 返回包含代码预览和复制按钮的 div 元素
  return (
    <div className="relative">
      # 复制按钮，点击按钮时复制代码内容，并根据复制结果显示提示信息
      <Button
        className="absolute right-3 top-2 text-gray-300 hover:!text-gray-200 bg-gray-700"
        type="text"
        icon={<CopyOutlined />}
        onClick={() => {
          const success = copy(code);
          message[success ? 'success' : 'error'](success ? 'Copy success' : 'Copy failed');
        }}
      />
      # 代码高亮显示组件，根据当前模式选择不同的样式
      <SyntaxHighlighter customStyle={customStyle} language={language} style={mode === 'dark' ? dark ?? coldarkDark : light ?? oneDark}>
        {code}
      </SyntaxHighlighter>
    </div>
  );
}
```
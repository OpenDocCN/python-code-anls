# `.\DB-GPT-src\web\components\chat\chat-content\vis-code.tsx`

```py
import ReactMarkdown from 'react-markdown';  // 导入用于渲染 Markdown 的 React 组件
import remarkGfm from 'remark-gfm';  // 导入用于支持 GFM（GitHub Flavored Markdown）的 Markdown 插件
import markdownComponents from './config';  // 导入自定义的 Markdown 渲染组件配置
import { CodePreview } from './code-preview';  // 导入代码预览组件
import classNames from 'classnames';  // 导入用于动态添加 CSS 类名的工具函数
import { useState } from 'react';  // 导入 React 的 useState 钩子
import { CheckOutlined, CloseOutlined } from '@ant-design/icons';  // 导入 Ant Design 的图标组件
import { useTranslation } from 'react-i18next';  // 导入用于多语言支持的 i18n 钩子
import { oneLight, oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';  // 导入 Prism 代码高亮的样式配置

interface Props {
  data: {
    code: string[];  // 代表代码片段的数组
    exit_success: true;  // 表示任务是否成功退出
    language: string;  // 代码的语言
    log: string;  // 包含任务日志的 Markdown 格式文本
  };
}

function VisCode({ data }: Props) {
  const { t } = useTranslation();  // 获取翻译函数

  const [show, setShow] = useState(0);  // 设置用于控制显示代码索引的状态

  return (
    <div className="bg-[#EAEAEB] rounded overflow-hidden border border-theme-primary dark:bg-theme-dark text-sm">
      <div>
        <div className="flex">
          {data.code.map((item, index) => (
            <div
              key={index}
              className={classNames('px-4 py-2 text-[#121417] dark:text-white cursor-pointer', {
                'bg-white dark:bg-theme-dark-container': index === show,
              })}
              onClick={() => {
                setShow(index);  // 点击时更新显示的代码索引
              }}
            >
              CODE {index + 1}: {item[0]}  // 渲染代码块标题
            </div>
          ))}
        </div>
        {data.code.length && (
          <CodePreview
            language={data.code[show][0]}  // 传递当前选中代码语言给预览组件
            code={data.code[show][1]}  // 传递当前选中代码内容给预览组件
            customStyle={{ maxHeight: 300, margin: 0 }}  // 自定义预览组件的样式
            light={oneLight}  // 传递亮色主题给代码高亮组件
            dark={oneDark}  // 传递暗色主题给代码高亮组件
          />
        )}
      </div>
      <div>
        <div className="flex">
          <div className="bg-white dark:bg-theme-dark-container px-4 py-2 text-[#121417] dark:text-white">
            {t('Terminal')} {data.exit_success ? <CheckOutlined className="text-green-600" /> : <CloseOutlined className="text-red-600" />}
            {/* 渲染终端标题及成功或失败的状态图标 */}
          </div>
        </div>
        <div className="p-4 max-h-72 overflow-y-auto whitespace-normal bg-white dark:dark:bg-theme-dark">
          <ReactMarkdown components={markdownComponents} remarkPlugins={[remarkGfm]}>
            {data.log}  // 使用自定义组件和插件渲染任务日志内容
          </ReactMarkdown>
        </div>
      </div>
    </div>
  );
}

export default VisCode;
```
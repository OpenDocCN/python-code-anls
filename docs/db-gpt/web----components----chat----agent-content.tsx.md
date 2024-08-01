# `.\DB-GPT-src\web\components\chat\agent-content.tsx`

```py
# 导入所需模块，包括 ChatContext 和 IChatDialogueMessageSchema
import { ChatContext } from '@/app/chat-context';
import { IChatDialogueMessageSchema } from '@/types/chat';
# 导入 classNames 模块，用于动态设置 CSS 类名
import classNames from 'classnames';
# 导入 memo 和 useContext，用于性能优化和获取 React 上下文
import { memo, useContext } from 'react';
# 导入 ReactMarkdown 组件，用于将 Markdown 文本转换为 React 组件
import ReactMarkdown from 'react-markdown';
# 导入 markdownComponents，这是一个配置对象，包含在 Markdown 中使用的自定义组件
import markdownComponents from './chat-content/config';
# 导入 rehypeRaw，用于在 ReactMarkdown 中处理原始 HTML
import rehypeRaw from 'rehype-raw';

# 定义 Props 接口，指定 content 属性的类型为 IChatDialogueMessageSchema
interface Props {
  content: IChatDialogueMessageSchema;
}

# 定义函数 formatMarkdownVal，用于调整 Markdown 中的表格标签格式
function formatMarkdownVal(val: string) {
  return val.replace(/<table(\w*=[^>]+)>/gi, '<table $1>').replace(/<tr(\w*=[^>]+)>/gi, '<tr $1>');
}

# 定义 React 组件 AgentContent，接受一个 Props 对象
function AgentContent({ content }: Props) {
  # 使用 useContext 获取 ChatContext 中的 scene 属性
  const { scene } = useContext(ChatContext);

  # 根据 content.role 判断是否为视图消息
  const isView = content.role === 'view';

  # 返回一个 div 元素，根据条件动态设置 CSS 类名和样式
  return (
    <div
      className={classNames('relative w-full p-2 md:p-4 rounded-xl break-words', {
        'bg-white dark:bg-[#232734]': isView,
        'lg:w-full xl:w-full pl-0': ['chat_with_db_execute', 'chat_dashboard'].includes(scene),
      })}
    >
      # 根据 isView 条件渲染不同的内容：使用 ReactMarkdown 显示 Markdown 内容，应用自定义组件和 rehypeRaw 插件；
      # 否则显示纯文本内容
      {isView ? (
        <ReactMarkdown components={markdownComponents} rehypePlugins={[rehypeRaw]}>
          {formatMarkdownVal(content.context)}
        </ReactMarkdown>
      ) : (
        <div className="">{content.context}</div>
      )}
    </div>
  );
}

# 使用 memo 函数包装 AgentContent 组件，以提高性能
export default memo(AgentContent);
```
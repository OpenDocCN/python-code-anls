# `.\DB-GPT-src\web\components\chat\chat-content\agent-messages.tsx`

```py
import ReactMarkdown from 'react-markdown';  # 导入ReactMarkdown组件，用于渲染Markdown内容
import markdownComponents from './config';  # 导入Markdown组件配置
import { renderModelIcon } from '../header/model-selector';  # 从头部模型选择器中导入renderModelIcon函数
import { SwapRightOutlined } from '@ant-design/icons';  # 从Ant Design图标库中导入SwapRightOutlined图标

interface Props {  # 定义Props接口，描述组件所需的属性
  data: {  # 定义data属性，包含多个消息对象的数组
    sender: string;  # 发送者名称
    receiver: string;  # 接收者名称
    model: string | null;  # 消息相关的模型，可以为字符串或null
    markdown: string;  # 包含Markdown内容的字符串
  }[];
}

function AgentMessages({ data }: Props) {  # 定义AgentMessages组件，接受Props作为参数
  if (!data || !data.length) return null;  # 如果data不存在或者data为空数组，则返回null

  return (
    <>  # React fragment，用于包裹返回的多个元素
      {data.map((item, index) => (  # 遍历data数组中的每个消息对象
        <div key={index} className="rounded my-4 md:my-6">  # 每个消息的外层div，使用index作为key，设置样式类名
          <div className="flex items-center mb-3 text-sm">  # 内部div，包含灵活布局和文本大小的样式类
            {item.model ? renderModelIcon(item.model) : <div className="rounded-full w-6 h-6 bg-gray-100" />}  # 根据item.model是否存在选择渲染模型图标或默认灰色圆形
            <div className="ml-2 opacity-70">  # 内部div，包含左边距和不透明度的样式类
              {item.sender}  # 显示发送者名称
              <SwapRightOutlined className="mx-2 text-base" />  # 渲染SwapRightOutlined图标，设置间距和文本大小的样式类
              {item.receiver}  # 显示接收者名称
            </div>
          </div>
          <div className="whitespace-normal text-sm">  # 内容div，包含样式类以定义空白处理和文本大小
            <ReactMarkdown components={markdownComponents}>{item.markdown}</ReactMarkdown>  # 使用ReactMarkdown组件渲染Markdown内容，应用指定的组件配置
          </div>
        </div>
      ))}
    </>
  );
}

export default AgentMessages;  # 导出AgentMessages组件作为默认导出
```
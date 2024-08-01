# `.\DB-GPT-src\web\components\chat\chat-content\agent-plans.tsx`

```py
import { CaretRightOutlined, CheckOutlined, ClockCircleOutlined } from '@ant-design/icons';
import { Collapse } from 'antd';
import ReactMarkdown from 'react-markdown';
import markdownComponents from './config';

interface Props {
  data: {
    name: string;
    num: number;
    status: 'complete' | 'todo';
    agent: string;
    markdown: string;
  }[];
}

// 显示代理计划的组件
function AgentPlans({ data }: Props) {
  // 如果数据为空或未定义，返回空
  if (!data || !data.length) return null;

  // 返回折叠面板组件
  return (
    <Collapse
      bordered  // 设置折叠面板有边框
      className="my-3"  // 设置自定义样式类名
      expandIcon={({ isActive }) => <CaretRightOutlined rotate={isActive ? 90 : 0} />}  // 定义展开图标，根据状态旋转
      items={data.map((item, index) => {
        return {
          key: index,  // 设置折叠项的唯一键
          label: (  // 定义折叠项的标签内容
            <div className="whitespace-normal">  // 设置容器的文本换行方式
              <span>
                {item.name} - {item.agent}  // 显示项目名称和代理名称
              </span>
              {item.status === 'complete' ? (  // 根据项目状态显示不同的图标
                <CheckOutlined className="!text-green-500 ml-2" />  // 完成状态图标
              ) : (
                <ClockCircleOutlined className="!text-gray-500 ml-2" />  // 待办状态图标
              )}
            </div>
          ),
          children: <ReactMarkdown components={markdownComponents}>{item.markdown}</ReactMarkdown>,  // 使用ReactMarkdown组件渲染Markdown内容
        };
      })}
    />
  );
}

export default AgentPlans;  // 导出AgentPlans组件作为默认组件
```
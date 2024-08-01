# `.\DB-GPT-src\web\components\chat\chat-content\vis-plugin.tsx`

```py
import { CheckOutlined, ClockCircleOutlined, CloseOutlined, LoadingOutlined } from '@ant-design/icons';
// 从 Ant Design 的图标库中导入四个特定的图标组件

import classNames from 'classnames';
// 导入 classNames 库，用于动态生成 CSS 类名

import { ReactNode } from 'react';
// 从 React 中导入 ReactNode 类型，用于表示可以渲染为 React 组件的任何内容

import ReactMarkdown from 'react-markdown';
// 导入 ReactMarkdown 组件，用于将 Markdown 文本渲染为 React 元素

import markdownComponents from './config';
// 导入 Markdown 组件配置，这里假设是一个配置文件

import rehypeRaw from 'rehype-raw';
// 导入 rehypeRaw 模块，用于在 ReactMarkdown 中处理原始 HTML

interface IVisPlugin {
  name: string;
  args: {
    query: string;
  };
  status: 'todo' | 'runing' | 'failed' | 'complete' | (string & {});
  logo: string | null;
  result: string;
  err_msg: string | null;
}
// 定义接口 IVisPlugin，描述了一个视觉插件的数据结构，包括名称、参数、状态、Logo、结果和错误信息

interface Props {
  data: IVisPlugin;
}
// 定义 Props 接口，包含一个 data 属性，类型为 IVisPlugin，用于传递给 VisPlugin 组件的数据

const pluginViewStatusMapper: Record<IVisPlugin['status'], { bgClass: string; icon: ReactNode }> = {
  // 定义 pluginViewStatusMapper 对象，根据状态值映射对应的背景类名和图标
  todo: {
    bgClass: 'bg-gray-500',
    icon: <ClockCircleOutlined className="ml-2" />,
  },
  runing: {
    bgClass: 'bg-blue-500',
    icon: <LoadingOutlined className="ml-2" />,
  },
  failed: {
    bgClass: 'bg-red-500',
    icon: <CloseOutlined className="ml-2" />,
  },
  complete: {
    bgClass: 'bg-green-500',
    icon: <CheckOutlined className="ml-2" />,
  },
};

function VisPlugin({ data }: Props) {
  const { bgClass, icon } = pluginViewStatusMapper[data.status] ?? {};
  // 根据传入的 data.status 从 pluginViewStatusMapper 中获取对应的 bgClass 和 icon

  return (
    <div className="bg-theme-light dark:bg-theme-dark-container rounded overflow-hidden my-2 flex flex-col lg:max-w-[80%]">
      <div className={classNames('flex px-4 md:px-6 py-2 items-center text-white text-sm', bgClass)}>
        {data.name}
        {icon}
      </div>
      {data.result ? (
        // 根据 data.result 的有无决定渲染不同的内容区域
        <div className="px-4 md:px-6 py-4 text-sm whitespace-normal">
          <ReactMarkdown components={markdownComponents} rehypePlugins={[rehypeRaw]}>
            {data.result ?? ''}
          </ReactMarkdown>
        </div>
      ) : (
        <div className="px-4 md:px-6 py-4 text-sm">{data.err_msg}</div>
      )}
    </div>
  );
}
// 定义 VisPlugin 组件，根据传入的 data 对象渲染不同的视觉插件信息

export default VisPlugin;
// 导出 VisPlugin 组件，使其可以在其他地方使用
```
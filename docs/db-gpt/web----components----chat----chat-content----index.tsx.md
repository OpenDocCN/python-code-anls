# `.\DB-GPT-src\web\components\chat\chat-content\index.tsx`

```py
import { PropsWithChildren, ReactNode, memo, useContext, useMemo } from 'react';
import { CheckOutlined, ClockCircleOutlined, CloseOutlined, CodeOutlined, LoadingOutlined, RobotOutlined, UserOutlined } from '@ant-design/icons';
import ReactMarkdown from 'react-markdown';
import { IChatDialogueMessageSchema } from '@/types/chat';
import rehypeRaw from 'rehype-raw';
import classNames from 'classnames';
import { Tag } from 'antd';
import { renderModelIcon } from '../header/model-selector';
import { ChatContext } from '@/app/chat-context';
import markdownComponents from './config';

// 定义组件的 Props 接口
interface Props {
  // 消息内容，包含 IChatDialogueMessageSchema 中除 context 外的属性，及 context 对象或字符串
  content: Omit<IChatDialogueMessageSchema, 'context'> & {
    context:
      | string
      | {
          template_name: string;
          template_introduce: string;
        };
  };
  // 是否是图表聊天
  isChartChat?: boolean;
  // 点击链接的回调函数
  onLinkClick?: () => void;
}

// 定义 ReactMarkdown 组件的 Props 类型
type MarkdownComponent = Parameters<typeof ReactMarkdown>['0']['components'];

// 定义 DBGPTView 类型
type DBGPTView = {
  name: string;
  status: 'todo' | 'runing' | 'failed' | 'completed' | (string & {});
  result?: string;
  err_msg?: string;
};

// 定义状态与样式、图标的映射关系
const pluginViewStatusMapper: Record<DBGPTView['status'], { bgClass: string; icon: ReactNode }> = {
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
  completed: {
    bgClass: 'bg-green-500',
    icon: <CheckOutlined className="ml-2" />,
  },
};

// 格式化 Markdown 中的特定值
function formatMarkdownVal(val: string) {
  return val
    .replaceAll('\\n', '\n') // 替换所有 '\\n' 为换行符 '\n'
    .replace(/<table(\w*=[^>]+)>/gi, '<table $1>') // 保留 table 标签的属性
    .replace(/<tr(\w*=[^>]+)>/gi, '<tr $1>'); // 保留 tr 标签的属性
}

// 聊天内容组件，接收 PropsWith
    const result = value.replace(/<dbgpt-view[^>]*>[^<]*<\/dbgpt-view>/gi, (matchVal) => {
      try {
        // 替换匹配到的字符串中的换行符，并移除标签及其内容，转换为插件值的字符串表示
        const pluginVal = matchVal.replaceAll('\n', '\\n').replace(/<[^>]*>|<\/[^>]*>/gm, '');
        // 将插件值的字符串表示解析为 DBGPTView 类型的对象
        const pluginContext = JSON.parse(pluginVal) as DBGPTView;
        // 构建替换用的自定义视图标记
        const replacement = `<custom-view>${cacheIndex}</custom-view>`;

        // 将解析后的插件上下文信息加入缓存数组中
        cachePluginContext.push({
          ...pluginContext,
          // 格式化插件结果的 Markdown 值，如果为空则使用空字符串
          result: formatMarkdownVal(pluginContext.result ?? ''),
        });
        // 增加缓存索引
        cacheIndex++;

        // 返回替换后的字符串
        return replacement;
      } catch (e) {
        // 捕获异常并记录到控制台，然后返回原始匹配字符串
        console.log((e as any).message, e);
        return matchVal;
      }
    });
    // 返回处理后的结果对象，包括关系、缓存的插件上下文和处理后的值
    return {
      relations,
      cachePluginContext,
      value: result,
    };
  }, [context]);

  // 使用 useMemo 定义额外的 Markdown 组件，当 context 或 cachePluginContext 发生变化时重新计算
  const extraMarkdownComponents = useMemo<MarkdownComponent>(
    () => ({
      'custom-view'({ children }) {
        // 将 children 转换为数字索引，如果对应缓存不存在则直接返回 children
        const index = +children.toString();
        if (!cachePluginContext[index]) {
          return children;
        }
        // 从缓存中获取插件信息的各个字段
        const { name, status, err_msg, result } = cachePluginContext[index];
        // 根据插件状态映射获取背景类和图标
        const { bgClass, icon } = pluginViewStatusMapper[status] ?? {};
        // 返回包含插件信息的 JSX 结构
        return (
          <div className="bg-white dark:bg-[#212121] rounded-lg overflow-hidden my-2 flex flex-col lg:max-w-[80%]">
            <div className={classNames('flex px-4 md:px-6 py-2 items-center text-white text-sm', bgClass)}>
              {name}
              {icon}
            </div>
            {result ? (
              <div className="px-4 md:px-6 py-4 text-sm">
                {/* 使用 ReactMarkdown 渲染 Markdown 内容，可配置的组件和插件 */}
                <ReactMarkdown components={markdownComponents} rehypePlugins={[rehypeRaw]}>
                  {result ?? ''}
                </ReactMarkdown>
              </div>
            ) : (
              <div className="px-4 md:px-6 py-4 text-sm">{err_msg}</div>
            )}
          </div>
        );
      },
    }),
    // useMemo 依赖于 context 和 cachePluginContext
    [context, cachePluginContext],
  );

  // 如果既不是机器人也没有上下文，则返回一个高度为 12 的空 div
  if (!isRobot && !context) return <div className="h-12"></div>;

  // 返回根据条件动态类名和样式的主要 JSX 结构
  return (
    <div
      className={classNames('relative flex flex-wrap w-full p-2 md:p-4 rounded-xl break-words', {
        'bg-white dark:bg-[#232734]': isRobot,
        'lg:w-full xl:w-full pl-0': ['chat_with_db_execute', 'chat_dashboard'].includes(scene),
      })}
    >
    <div className="mr-2 flex flex-shrink-0 items-center justify-center h-7 w-7 rounded-full text-lg sm:mr-4">
      {/* 根据 isRobot 变量决定渲染不同的图标 */}
      {isRobot ? renderModelIcon(model_name) || <RobotOutlined /> : <UserOutlined />}
    </div>
    <div className="flex-1 overflow-hidden items-center text-md leading-8 pb-2">
      {/* 显示用户输入的文本内容，前提是非机器人模式且 context 是字符串 */}
      {!isRobot && typeof context === 'string' && context}
      {/* 显示报告的渲染，条件为机器人模式、是图表聊天、context 是对象 */}
      {isRobot && isChartChat && typeof context === 'object' && (
        <div>
          {`[${context.template_name}]: `}
          <span className="text-theme-primary cursor-pointer" onClick={onLinkClick}>
            <CodeOutlined className="mr-1" />
            {context.template_introduce || 'More Details'}
          </span>
        </div>
      )}
      {/* 显示 Markdown 渲染的内容，条件为机器人模式且 context 是字符串 */}
      {isRobot && typeof context === 'string' && (
        <ReactMarkdown components={{ ...markdownComponents, ...extraMarkdownComponents }} rehypePlugins={[rehypeRaw]}>
          {formatMarkdownVal(value)}
        </ReactMarkdown>
      )}
      {/* 如果 relations 数组有内容，则渲染标签 */}
      {!!relations?.length && (
        <div className="flex flex-wrap mt-2">
          {relations?.map((value, index) => (
            <Tag color="#108ee9" key={value + index}>
              {value}
            </Tag>
          ))}
        </div>
      )}
    </div>
    {children}
}

export default memo(ChatContent);
```
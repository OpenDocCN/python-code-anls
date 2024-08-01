# `.\DB-GPT-src\web\components\common\chat-dialog.tsx`

```py
import useChat from '@/hooks/use-chat';  
// 导入自定义的 useChat 钩子函数

import CompletionInput from './completion-input';  
// 导入完成输入组件

import { useCallback, useState } from 'react';  
// 从 React 库中导入 useCallback 和 useState 钩子

import { IChatDialogueMessageSchema, IChatDialogueSchema } from '@/types/chat';  
// 导入与聊天相关的数据结构类型定义

import AgentContent from '../chat/agent-content';  
// 导入代理内容组件

import { renderModelIcon } from '../chat/header/model-selector';  
// 从聊天头部模型选择器中导入渲染模型图标的函数

import MyEmpty from './MyEmpty';  
// 导入自定义空组件

import { CaretLeftOutlined } from '@ant-design/icons';  
// 从 Ant Design 图标库中导入向左的插入符图标

import classNames from 'classnames';  
// 导入用于动态设置类名的 classNames 函数

import { useRequest } from 'ahooks';  
// 从 ahooks 库中导入 useRequest 钩子函数

import { apiInterceptors, newDialogue } from '@/client/api';  
// 从客户端 API 中导入请求拦截器和新对话函数

import ChatContent from '../chat/chat-content';  
// 导入聊天内容组件

interface Props {  
// 定义 Props 接口，包含 title、completionApi、chatMode、chatParams 和 model 属性
  title?: string;  
  // 可选的标题属性
  completionApi?: string;  
  // 可选的完成 API 地址属性
  chatMode: IChatDialogueSchema['chat_mode'];  
  // 聊天模式，基于 IChatDialogueSchema 结构中的 chat_mode 属性
  chatParams?: {  
    select_param?: string;  
    // 可选的选择参数属性
  } & Record<string, string>;  
  // 字符串键值对类型的聊天参数
  model?: string;  
  // 可选的模型名称属性
}

function ChatDialog({ title, chatMode, completionApi, chatParams, model = '' }: Props) {  
// 聊天对话框组件，接受 Props 参数
  const chat = useChat({ queryAgentURL: completionApi });  
  // 使用自定义 useChat 钩子函数，传入 completionApi 作为 queryAgentURL 参数

  const [loading, setLoading] = useState(false);  
  // 定义 loading 状态和其更新函数，默认为 false
  const [list, setList] = useState<IChatDialogueMessageSchema[]>([]);  
  // 定义对话消息列表状态和其更新函数，初始化为空数组
  const [open, setOpen] = useState(false);  
  // 定义打开状态和其更新函数，默认为 false

  const { data } = useRequest(  
    async () => {
      const [, res] = await apiInterceptors(newDialogue({ chat_mode: chatMode }));
      return res;
    },
    {
      ready: !!chatMode,
    },
  );
  // 使用 useRequest 钩子函数，异步获取数据，调用 apiInterceptors 和 newDialogue 函数，返回响应数据

  const handleChat = useCallback(  
    (content: string) => {
      if (!data) return;
      // 如果数据不存在，直接返回
      return new Promise<void>((resolve) => {
        const tempList: IChatDialogueMessageSchema[] = [
          ...list,
          { role: 'human', context: content, model_name: model, order: 0, time_stamp: 0 },
          { role: 'view', context: '', model_name: model, order: 0, time_stamp: 0 },
        ];
        // 创建临时对话消息列表，包含人类角色和视图角色的初始消息对象

        const index = tempList.length - 1;
        // 获取临时列表的最后一个索引

        setList([...tempList]);
        // 更新对话消息列表状态，使用临时列表副本

        setLoading(true);
        // 设置加载状态为 true

        chat({
          chatId: data?.conv_uid,
          // 设置对话 ID 为 data 对象的 conv_uid 属性
          data: { ...chatParams, chat_mode: chatMode, model_name: model, user_input: content },
          // 传递聊天参数，包括 chatParams、chatMode、model 和用户输入的内容
          onMessage: (message) => {
            tempList[index].context = message;
            // 当接收到消息时，更新临时列表中指定索引的上下文内容
            setList([...tempList]);
            // 更新对话消息列表状态，使用临时列表副本
          },
          onDone: () => {
            resolve();
            // 当完成时，执行 resolve 函数
          },
          onClose: () => {
            resolve();
            // 当关闭时，执行 resolve 函数
          },
          onError: (message) => {
            tempList[index].context = message;
            // 当发生错误时，更新临时列表中指定索引的上下文内容
            setList([...tempList]);
            // 更新对话消息列表状态，使用临时列表副本
            resolve();
            // 执行 resolve 函数
          },
        }).finally(() => {
          setLoading(false);
          // 不论如何，最终将加载状态设置为 false
        });
      });
    },
    [chat, list, data?.conv_uid],
    // 依赖于 chat、list 和 data 对象的 conv_uid 属性
  );

  return (
    <div
      className={classNames(
        'fixed top-0 right-0 w-[30rem] h-screen flex flex-col bg-white dark:bg-theme-dark-container shadow-[-5px_0_40px_-4px_rgba(100,100,100,.1)] transition-transform duration-300',
        // 动态设置组件的类名，根据 open 状态应用不同的样式
        {
          'translate-x-0': open,
          // 如果 open 为 true，则应用 translate-x-0 类名
          'translate-x-full': !open,
          // 如果 open 为 false，则应用 translate-x-full 类名
        },
      )}
    >
      // 如果存在标题，则渲染一个包含标题的 div 元素
      {title && <div className="p-4 border-b border-solid border-gray-100">{title}</div>}
      // 渲染一个具有滚动条的容器
      <div className="flex-1 overflow-y-auto px-2">
        // 遍历列表中的每个元素，根据 chatParams 中的 chat_mode 属性选择渲染 AgentContent 或 ChatContent 组件
        {list.map((item, index) => (
          <>{chatParams?.chat_mode === 'chat_agent' ? <AgentContent key={index} content={item} /> : <ChatContent key={index} content={item} />}</>
        ))}
        // 如果列表为空，则渲染一个 MyEmpty 组件
        {!list.length && <MyEmpty description="" />}
      </div>
      // 渲染一个包含模型图标和输入框的 div 元素
      <div className="flex w-full p-4 border-t border-solid border-gray-100 items-center">
        // 如果存在模型，则渲染一个包含模型图标的 div 元素
        {model && <div className="mr-2 flex">{renderModelIcon(model)}</div>}
        // 渲染一个包含加载状态和提交事件的 CompletionInput 组件
        <CompletionInput loading={loading} onSubmit={handleChat} />
      </div>
      // 渲染一个包含向左箭头图标的 div 元素，点击时触发 setOpen 函数
      <div
        className="flex items-center justify-center rounded-tl rounded-bl cursor-pointer w-5 h-11 absolute top-[50%] -left-5 -translate-y-[50%] bg-white"
        onClick={() => {
          setOpen(!open);
        }}
      >
        <CaretLeftOutlined />
      </div>
    </div>
  );
}

export default ChatDialog;
```
# `.\DB-GPT-src\web\hooks\use-chat.ts`

```py
import { EventStreamContentType, fetchEventSource } from '@microsoft/fetch-event-source';
import { message } from 'antd';
import { useCallback, useContext, useEffect, useMemo } from 'react';
import i18n from '@/app/i18n';
import { ChatContext } from '@/app/chat-context';

type Props = {
  queryAgentURL?: string;
};

type ChatParams = {
  chatId: string;
  data?: Record<string, any>;
  query?: Record<string, string>;
  onMessage: (message: string) => void;
  onClose?: () => void;
  onDone?: () => void;
  onError?: (content: string, error?: Error) => void;
};

// 自定义 Hook：用于处理聊天相关逻辑
const useChat = ({ queryAgentURL = '/api/v1/chat/completions' }: Props) => {
  // 创建一个用于中止请求的 AbortController 实例
  const ctrl = useMemo(() => new AbortController(), []);

  // 从 ChatContext 中获取当前的 scene 上下文
  const { scene } = useContext(ChatContext);

  // useCallback 用于创建一个记忆化的回调函数 chat，依赖于 queryAgentURL 和 ctrl
  const chat = useCallback(
    async ({ data, chatId, onMessage, onClose, onDone, onError }: ChatParams) => {
      // 如果 data 中缺少 user_input 和 doc_id，则显示警告信息并返回
      if (!data?.user_input && !data?.doc_id) {
        message.warning(i18n.t('no_context_tip'));
        return;
      }

      // 构造请求参数对象 parmas，包括传入的 data 和 chatId
      const parmas = {
        ...data,
        conv_uid: chatId,
      };

      // 如果 conv_uid 不存在，则显示错误信息并返回
      if (!parmas.conv_uid) {
        message.error('conv_uid 不存在，请刷新后重试');
        return;
      }

      try {
        // 发起使用 fetchEventSource 函数进行的事件源请求，POST 方法，发送 JSON 格式的 body 数据
        await fetchEventSource(`${process.env.API_BASE_URL ?? ''}${queryAgentURL}`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(parmas),
          signal: ctrl.signal, // 使用 AbortController 的信号来控制请求中止
          openWhenHidden: true, // 在页面隐藏时继续打开连接

          // 当连接建立时的回调函数
          async onopen(response) {
            if (response.ok && response.headers.get('content-type') === EventStreamContentType) {
              return;
            }
            // 如果响应内容类型为 application/json，则解析并处理数据
            if (response.headers.get('content-type') === 'application/json') {
              response.json().then((data) => {
                onMessage?.(data); // 调用 onMessage 回调函数处理收到的数据
                onDone?.(); // 调用 onDone 回调函数
                ctrl.abort(); // 中止请求
              });
            }
          },

          // 连接关闭时的回调函数
          onclose() {
            ctrl.abort(); // 中止请求
            onClose?.(); // 调用 onClose 回调函数
          },

          // 发生错误时的回调函数
          onerror(err) {
            throw new Error(err); // 抛出错误
          },

          // 收到消息时的回调函数
          onmessage: (event) => {
            let message = event.data; // 获取消息数据
            try {
              // 尝试解析消息数据为 JSON 格式，根据当前场景进行适当处理
              if (scene === 'chat_agent') {
                message = JSON.parse(message).vis;
              } else {
                message = JSON.parse(message);
              }
            } catch (e) {
              message.replaceAll('\\n', '\n'); // 处理异常，替换消息中的转义字符
            }

            // 根据消息类型进行不同的处理
            if (typeof message === 'string') {
              if (message === '[DONE]') {
                onDone?.(); // 调用 onDone 回调函数
              } else if (message?.startsWith('[ERROR]')) {
                onError?.(message?.replace('[ERROR]', '')); // 调用 onError 回调函数处理错误消息
              } else {
                onMessage?.(message); // 调用 onMessage 回调函数处理普通消息
              }
            } else {
              onMessage?.(message); // 调用 onMessage 回调函数处理消息对象
              onDone?.(); // 调用 onDone 回调函数
            }
          },
        });
      } catch (err) {
        ctrl.abort(); // 中止请求
        onError?.('Sorry, We meet some error, please try agin later.', err as Error); // 调用 onError 处理错误信息
      }
    },
    [queryAgentURL, ctrl, scene] // useCallback 的依赖项
  );

  // 返回 chat 方法和中止控制器 ctrl
  return { chat, ctrl };
};
    [queryAgentURL],
  );



// 将 queryAgentURL 作为依赖项传递给 useEffect 钩子



  useEffect(() => {
    return () => {
      ctrl.abort();
    };
  }, []);



// 使用 useEffect 钩子设置副作用，返回一个清理函数以取消操作
// 清理函数在组件卸载时调用，用于中止控制器的操作



  return chat;



// 返回 chat 变量作为组件的渲染结果
};

export default useChat;
```
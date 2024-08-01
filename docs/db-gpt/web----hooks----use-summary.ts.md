# `.\DB-GPT-src\web\hooks\use-summary.ts`

```py
import { ChatContext } from '@/app/chat-context';
import { ChatHistoryResponse } from '@/types/chat';
import { useCallback, useContext } from 'react';
import useChat from './use-chat';
import { apiInterceptors, getChatHistory } from '@/client/api';

const useSummary = () => {
  // 从 ChatContext 中获取所需的状态和函数
  const { history, setHistory, chatId, model, docId } = useContext(ChatContext);
  // 使用 useChat 自定义 hook 来获取与聊天相关的函数和状态
  const chat = useChat({ queryAgentURL: '/knowledge/document/summary' });

  // 定义 summary 回调函数，用于获取聊天历史并更新状态
  const summary = useCallback(
    async (curDocId?: number) => {
      // 发起 API 请求获取聊天历史
      const [, res] = await apiInterceptors(getChatHistory(chatId));
      // 创建临时的聊天历史数组，包含初始和重试信息
      const tempHistory: ChatHistoryResponse = [
        ...res!,
        { role: 'human', context: '', model_name: model, order: 0, time_stamp: 0 },
        { role: 'view', context: '', model_name: model, order: 0, time_stamp: 0, retry: true },
      ];
      // 获取最后一个元素的索引
      const index = tempHistory.length - 1;
      // 更新 ChatContext 中的聊天历史状态
      setHistory([...tempHistory]);
      // 调用 chat 函数发送消息并更新聊天历史
      await chat({
        data: {
          doc_id: curDocId || docId,
          model_name: model,
        },
        chatId,
        // 在收到消息时更新临时历史记录和 ChatContext 中的历史状态
        onMessage: (message) => {
          tempHistory[index].context = message;
          setHistory([...tempHistory]);
        },
      });
    },
    // 指定依赖项数组，确保 useCallback 只在这些状态或函数变化时重新创建 summary 函数
    [history, model, docId, chatId],
  );
  // 返回 summary 回调函数
  return summary;
};

export default useSummary;
```
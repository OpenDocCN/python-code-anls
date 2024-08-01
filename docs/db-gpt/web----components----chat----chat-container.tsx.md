# `.\DB-GPT-src\web\components\chat\chat-container.tsx`

```py
import React, { useCallback, useContext, useEffect, useState } from 'react';
import { useAsyncEffect } from 'ahooks'; // 引入从ahooks库中导入的useAsyncEffect钩子
import useChat from '@/hooks/use-chat'; // 引入自定义的useChat钩子
import Completion from './completion'; // 引入当前目录下的completion组件
import { ChartData, ChatHistoryResponse } from '@/types/chat'; // 导入来自类型定义文件中的ChartData和ChatHistoryResponse类型
import { apiInterceptors, getChatHistory } from '@/client/api'; // 导入来自API客户端文件中的apiInterceptors和getChatHistory函数
import { ChatContext } from '@/app/chat-context'; // 导入当前应用中的聊天上下文
import Header from './header'; // 引入当前目录下的header组件
import Chart from '../chart'; // 引入上级目录下的chart组件
import classNames from 'classnames'; // 引入classnames库
import MuiLoading from '../common/loading'; // 引入公共目录下的loading组件
import { useSearchParams } from 'next/navigation'; // 从next/navigation中导入useSearchParams钩子
import { getInitMessage } from '@/utils'; // 从工具函数文件中导入getInitMessage函数
import MyEmpty from '../common/MyEmpty'; // 引入公共目录下的MyEmpty组件

const ChatContainer = () => {
  const searchParams = useSearchParams(); // 使用useSearchParams钩子获取当前URL中的查询参数
  const { scene, chatId, model, agent, setModel, history, setHistory } = useContext(ChatContext); // 从ChatContext上下文中解构出相关变量和函数
  const chat = useChat({}); // 使用自定义的useChat钩子，传入空对象作为初始参数
  const initMessage = (searchParams && searchParams.get('initMessage')) ?? ''; // 获取initMessage参数，如果不存在则为空字符串

  const [loading, setLoading] = useState<boolean>(false); // 定义loading状态，并初始化为false
  const [chartsData, setChartsData] = useState<Array<ChartData>>(); // 定义chartsData状态，初始化为undefined

  // 异步获取聊天历史记录的函数
  const getHistory = async () => {
    setLoading(true); // 设置loading状态为true，表示开始加载
    const [, res] = await apiInterceptors(getChatHistory(chatId)); // 发送API请求获取聊天历史记录
    setHistory(res ?? []); // 将获取的聊天历史记录设置到状态中，如果没有获取到则设置为空数组
    setLoading(false); // 设置loading状态为false，加载结束
  };

  // 处理获取到的聊天历史记录，提取最后一条记录中的图表数据
  const getChartsData = (list: ChatHistoryResponse) => {
    const contextTemp = list[list.length - 1]?.context; // 获取最后一条聊天历史记录的上下文信息
    if (contextTemp) { // 如果上下文信息存在
      try {
        const contextObj = typeof contextTemp === 'string' ? JSON.parse(contextTemp) : contextTemp; // 尝试解析上下文信息为对象
        setChartsData(contextObj?.template_name === 'report' ? contextObj?.charts : undefined); // 如果上下文中的模板名称为'report'，则设置图表数据；否则设置为undefined
      } catch (e) {
        setChartsData(undefined); // 解析出错时，设置图表数据为undefined
      }
    }
  };

  // 使用useAsyncEffect钩子，在组件挂载后执行初始化操作
  useAsyncEffect(async () => {
    const initMessage = getInitMessage(); // 调用getInitMessage函数获取初始消息
    if (initMessage && initMessage.id === chatId) return; // 如果获取到的初始消息存在且其id等于chatId，则直接返回
    await getHistory(); // 异步获取聊天历史记录
  }, [initMessage, chatId]); // 依赖项为initMessage和chatId，当它们变化时重新执行

  // 当history状态变化时执行的副作用，用于更新模型名称和图表数据
  useEffect(() => {
    if (!history.length) return; // 如果聊天历史记录为空，则直接返回
    /** use last view model_name as default model name */
    const lastView = history.filter((i) => i.role === 'view')?.slice(-1)?.[0]; // 获取最后一次视图操作的记录
    lastView?.model_name && setModel(lastView.model_name); // 如果最后一次视图操作有模型名称，则设置模型名称
    getChartsData(history); // 获取聊天历史记录中的图表数据
  }, [history.length]); // 依赖项为history.length，当其变化时重新执行

  // 组件卸载时执行的副作用，用于清空聊天历史记录
  useEffect(() => {
    return () => {
      setHistory([]); // 清空聊天历史记录
    };
  }, []); // 依赖项为空数组，表示只在组件卸载时执行一次

  const handleChat = useCallback(
    // 定义一个异步函数，接收一个字符串内容和可选的数据对象作为参数
    (content: string, data?: Record<string, any>) => {
      return new Promise<void>((resolve) => {
        // 创建临时聊天历史记录数组，包括用户输入的消息和模型相关信息
        const tempHistory: ChatHistoryResponse = [
          ...history,
          { role: 'human', context: content, model_name: model, order: 0, time_stamp: 0 },
          { role: 'view', context: '', model_name: model, order: 0, time_stamp: 0 },
        ];
        const index = tempHistory.length - 1;
        // 更新全局的聊天历史记录
        setHistory([...tempHistory]);
        // 调用 chat 函数进行聊天交互
        chat({
          // 构建聊天数据对象，包括场景模式、模型名称和用户输入的内容
          data: { ...data, chat_mode: scene || 'chat_normal', model_name: model, user_input: content },
          chatId,
          // 处理收到消息时的回调函数
          onMessage: (message) => {
            // 根据 incremental 标志决定是追加消息还是替换当前消息
            if (data?.incremental) {
              tempHistory[index].context += message;
            } else {
              tempHistory[index].context = message;
            }
            // 更新聊天历史记录
            setHistory([...tempHistory]);
          },
          // 聊天完成时的回调函数
          onDone: () => {
            // 获取并更新图表数据
            getChartsData(tempHistory);
            resolve(); // 解析 Promise
          },
          // 聊天关闭时的回调函数
          onClose: () => {
            // 获取并更新图表数据
            getChartsData(tempHistory);
            resolve(); // 解析 Promise
          },
          // 聊天错误时的回调函数
          onError: (message) => {
            // 更新错误信息到聊天历史记录
            tempHistory[index].context = message;
            // 更新聊天历史记录
            setHistory([...tempHistory]);
            resolve(); // 解析 Promise
          },
        });
      });
    },
    // 依赖项数组，包括用到的全局变量和函数
    [history, chat, chatId, model, agent, scene],
  );

  return (
    <>
      {/* 显示加载状态组件，根据 loading 状态控制显示与隐藏 */}
      <MuiLoading visible={loading} />
      {/* 头部组件，包含刷新历史和模型切换功能 */}
      <Header
        refreshHistory={getHistory} // 刷新历史记录的回调函数
        modelChange={(newModel: string) => {
          setModel(newModel); // 更新当前模型的回调函数
        }}
      />
      <div className="px-4 flex flex-1 flex-wrap overflow-hidden relative">
        {!!chartsData?.length && (
          <div className="w-full pb-4 xl:w-3/4 h-1/2 xl:pr-4 xl:h-full overflow-y-auto">
            {/* 图表组件，根据图表数据动态渲染 */}
            <Chart chartsData={chartsData} />
          </div>
        )}
        {!chartsData?.length && scene === 'chat_dashboard' && <MyEmpty className="w-full xl:w-3/4 h-1/2 xl:h-full" />}
        {/* 聊天面板 */}
        <div
          className={classNames('flex flex-1 flex-col overflow-hidden', {
            // 根据场景不同添加不同的 CSS 类
            'px-0 xl:pl-4 h-1/2 w-full xl:w-auto xl:h-full border-t xl:border-t-0 xl:border-l dark:border-gray-800': scene === 'chat_dashboard',
            'h-full lg:px-8': scene !== 'chat_dashboard',
          })}
        >
          {/* 消息输入框和自动完成组件 */}
          <Completion messages={history} onSubmit={handleChat} />
        </div>
      </div>
    </>
  );
};

export default ChatContainer;
```
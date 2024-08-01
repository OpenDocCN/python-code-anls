# `.\DB-GPT-src\web\app\chat-context.tsx`

```py
import { createContext, useEffect, useMemo, useState } from 'react';
import { apiInterceptors, getDialogueList, getUsableModels } from '@/client/api';  // 导入必要的 API 函数和模块
import { useRequest } from 'ahooks';  // 导入自定义的 hooks 库
import { ChatHistoryResponse, DialogueListResponse, IChatDialogueSchema } from '@/types/chat';  // 导入聊天历史和对话列表相关的类型定义
import { useSearchParams } from 'next/navigation';  // 导入用于处理 URL 查询参数的 hooks
import { STORAGE_THEME_KEY } from '@/utils';  // 导入存储主题模式的键值

type ThemeMode = 'dark' | 'light';  // 定义主题模式类型，只能是 'dark' 或 'light'

interface IChatContext {  // 定义聊天上下文接口
  mode: ThemeMode;  // 当前主题模式
  isContract?: boolean;  // 是否合约状态（可选）
  isMenuExpand?: boolean;  // 是否展开菜单（可选）
  scene: IChatDialogueSchema['chat_mode'] | (string & {});  // 聊天场景模式
  chatId: string;  // 聊天 ID
  model: string;  // 当前模型
  dbParam?: string;  // 数据库参数（可选）
  modelList: Array<string>;  // 可用模型列表
  agent: string;  // 代理人
  dialogueList?: DialogueListResponse;  // 对话列表（可选）
  setAgent?: (val: string) => void;  // 设置代理人的方法（可选）
  setMode: (mode: ThemeMode) => void;  // 设置主题模式的方法
  setModel: (val: string) => void;  // 设置模型的方法
  setIsContract: (val: boolean) => void;  // 设置是否合约状态的方法
  setIsMenuExpand: (val: boolean) => void;  // 设置是否展开菜单的方法
  setDbParam: (val: string) => void;  // 设置数据库参数的方法
  queryDialogueList: () => void;  // 查询对话列表的方法
  refreshDialogList: () => void;  // 刷新对话列表的方法
  currentDialogue?: DialogueListResponse[0];  // 当前对话（可选）
  history: ChatHistoryResponse;  // 聊天历史
  setHistory: (val: ChatHistoryResponse) => void;  // 设置聊天历史的方法
  docId?: number;  // 文档 ID（可选）
  setDocId: (docId: number) => void;  // 设置文档 ID 的方法
}

function getDefaultTheme(): ThemeMode {  // 获取默认主题的函数
  const theme = localStorage.getItem(STORAGE_THEME_KEY) as ThemeMode;  // 从 localStorage 中获取存储的主题模式
  if (theme) return theme;  // 如果获取到主题模式则返回该模式
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';  // 否则根据系统偏好返回默认的主题模式
}

const ChatContext = createContext<IChatContext>({  // 创建聊天上下文的 Context 对象，并设置默认值
  mode: 'light',  // 默认主题模式为 'light'
  scene: '',  // 默认聊天场景为空字符串
  chatId: '',  // 默认聊天 ID 为空字符串
  modelList: [],  // 默认模型列表为空数组
  model: '',  // 默认模型为空字符串
  dbParam: undefined,  // 默认数据库参数为 undefined
  dialogueList: [],  // 默认对话列表为空数组
  agent: '',  // 默认代理人为空字符串
  setAgent: () => {},  // 设置代理人方法默认为空函数
  setModel: () => {},  // 设置模型方法默认为空函数
  setIsContract: () => {},  // 设置是否合约状态方法默认为空函数
  setIsMenuExpand: () => {},  // 设置是否展开菜单方法默认为空函数
  setDbParam: () => void 0,  // 设置数据库参数方法默认为空函数
  queryDialogueList: () => {},  // 查询对话列表方法默认为空函数
  refreshDialogList: () => {},  // 刷新对话列表方法默认为空函数
  setMode: () => void 0,  // 设置主题模式方法默认为空函数
  history: [],  // 默认聊天历史为空数组
  setHistory: () => {},  // 设置聊天历史方法默认为空函数
  docId: undefined,  // 默认文档 ID 为 undefined
  setDocId: () => {},  // 设置文档 ID 方法默认为空函数
});

const ChatContextProvider = ({ children }: { children: React.ReactElement }) => {  // 创建聊天上下文的 Provider 组件
  const searchParams = useSearchParams();  // 使用 useSearchParams hook 获取当前页面的查询参数
  const chatId = searchParams?.get('id') ?? '';  // 获取查询参数中的聊天 ID，若不存在则为空字符串
  const scene = searchParams?.get('scene') ?? '';  // 获取查询参数中的聊天场景，若不存在则为空字符串
  const db_param = searchParams?.get('db_param') ?? '';  // 获取查询参数中的数据库参数，若不存在则为空字符串

  const [isContract, setIsContract] = useState(false);  // 使用 useState 创建是否合约状态的状态和设置方法，默认为 false
  const [model, setModel] = useState<string>('');  // 使用 useState 创建当前模型及设置方法，默认为空字符串
  const [isMenuExpand, setIsMenuExpand] = useState<boolean>(scene !== 'chat_dashboard');  // 使用 useState 创建是否展开菜单状态及设置方法，默认根据场景值判断
  const [dbParam, setDbParam] = useState<string>(db_param);  // 使用 useState 创建数据库参数及设置方法，默认为查询参数中的值
  const [agent, setAgent] = useState<string>('');  // 使用 useState 创建代理人及设置方法，默认为空字符串
  const [history, setHistory] = useState<ChatHistoryResponse>([]);  // 使用 useState 创建聊天历史及设置方法，默认为空数组
  const [docId, setDocId] = useState<number>();  // 使用 useState 创建文档 ID 及设置方法，默认为 undefined
  const [mode, setMode] = useState<ThemeMode>('light');  // 使用 useState 创建主题模式及设置方法，默认为 'light'

  const {
    run: queryDialogueList,  // 使用 useRequest hook 发起查询对话列表请求的方法
    data: dialogueList = [],  // 获取 useRequest 返回的对话列表数据，默认为空数组
    refresh: refreshDialogList,  // 刷新对话列表数据的方法
  } = useRequest(
    async () => {
      const [, res] = await apiInterceptors(getDialogueList());  // 发起对话列表请求，并通过 apiInterceptors 进行拦截处理
      return res ?? [];  // 返回对话列表数据，若无数据则返回空数组
    },
    {
      manual: true,  // 手动触发请求
    },
  );

  useEffect(() => {
    // 如果对话列表不为空且场景为 'chat_agent'，则执行以下操作
    if (dialogueList.length && scene === 'chat_agent') {
      // 查找对话列表中满足条件的项，获取其中的 select_param 字段作为 agent
      const agent = dialogueList.find((item) => item.conv_uid === chatId)?.select_param;
      // 如果 agent 存在，则设置 agent 状态
      agent && setAgent(agent);
    }
  }, [dialogueList, scene, chatId]);

  // 使用 useRequest hook 发起异步请求，获取模型列表数据，并设置默认值为空数组
  const { data: modelList = [] } = useRequest(async () => {
    // 发起带有拦截器的 API 请求，获取可用模型列表
    const [, res] = await apiInterceptors(getUsableModels());
    // 返回响应数据，如果为空则返回空数组
    return res ?? [];
  });

  // 在组件挂载时，设置默认主题模式
  useEffect(() => {
    setMode(getDefaultTheme());
  }, []);

  // 当模型列表或其长度发生变化时，设置第一个模型作为当前模型
  useEffect(() => {
    setModel(modelList[0]);
  }, [modelList, modelList?.length]);

  // 使用 useMemo 计算当前对话，根据 chatId 从对话列表中获取
  const currentDialogue = useMemo(() => dialogueList.find((item: any) => item.conv_uid === chatId), [chatId, dialogueList]);

  // 定义上下文值，包含组件需要的各种状态和操作函数
  const contextValue = {
    isContract,
    isMenuExpand,
    scene,
    chatId,
    modelList,
    model,
    dbParam: dbParam || db_param,
    dialogueList,
    agent,
    setAgent,
    mode,
    setMode,
    setModel,
    setIsContract,
    setIsMenuExpand,
    setDbParam,
    queryDialogueList,
    refreshDialogList,
    currentDialogue,
    history,
    setHistory,
    docId,
    setDocId,
  };
  // 返回 ChatContext.Provider，将上下文值传递给其消费者组件
  return <ChatContext.Provider value={contextValue}>{children}</ChatContext.Provider>;
};

export { ChatContext, ChatContextProvider };
```
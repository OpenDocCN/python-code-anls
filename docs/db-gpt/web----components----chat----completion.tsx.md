# `.\DB-GPT-src\web\components\chat\completion.tsx`

```py
import { useState, useRef, useEffect, useMemo, useContext } from 'react';
import { useSearchParams } from 'next/navigation';
import MonacoEditor from './monaco-editor';
import ChatContent from './chat-content';
import ChatFeedback from './chat-feedback';
import { ChatContext } from '@/app/chat-context';
import { FeedBack, IChatDialogueMessageSchema } from '@/types/chat';
import classNames from 'classnames';
import { Modal, message, Tooltip } from 'antd';
import { renderModelIcon } from './header/model-selector';
import { cloneDeep } from 'lodash';
import copy from 'copy-to-clipboard';
import { useTranslation } from 'react-i18next';
import CompletionInput from '../common/completion-input';
import { useAsyncEffect } from 'ahooks';
import { STORAGE_INIT_MESSAGE_KET } from '@/utils';
import { Button, IconButton } from '@mui/joy';
import { CopyOutlined, RedoOutlined } from '@ant-design/icons';
import { getInitMessage } from '@/utils';
import { apiInterceptors, getChatFeedBackSelect } from '@/client/api';
import useSummary from '@/hooks/use-summary';
import AgentContent from './agent-content';
import MyEmpty from '../common/MyEmpty';

type Props = {
  messages: IChatDialogueMessageSchema[];
  onSubmit: (message: string, otherQueryBody?: Record<string, any>) => Promise<void>;
};

const Completion = ({ messages, onSubmit }: Props) => {
  // 获取全局聊天上下文
  const { dbParam, currentDialogue, scene, model, refreshDialogList, chatId, agent, docId } = useContext(ChatContext);
  // 国际化翻译
  const { t } = useTranslation();
  // 获取当前页面的查询参数
  const searchParams = useSearchParams();

  // 从查询参数中获取流程选择参数和空间名称原始值，若无则为空字符串
  const flowSelectParam = (searchParams && searchParams.get('select_param')) ?? '';
  const spaceNameOriginal = (searchParams && searchParams.get('spaceNameOriginal')) ?? '';

  // 状态管理：加载状态、JSON 模态框状态、消息展示、JSON 值、选择参数
  const [isLoading, setIsLoading] = useState(false);
  const [jsonModalOpen, setJsonModalOpen] = useState(false);
  const [showMessages, setShowMessages] = useState(messages);
  const [jsonValue, setJsonValue] = useState<string>('');
  const [select_param, setSelectParam] = useState<FeedBack>();

  // 滚动容器的引用
  const scrollableRef = useRef<HTMLDivElement>(null);

  // 根据场景缓存是否为图表聊天
  const isChartChat = useMemo(() => scene === 'chat_dashboard', [scene]);

  // 使用 useSummary 自定义 hook 获取概要信息
  const summary = useSummary();

  // 根据场景选择不同的选择参数
  const selectParam = useMemo(() => {
    switch (scene) {
      case 'chat_agent':
        return agent;
      case 'chat_excel':
        return currentDialogue?.select_param;
      case 'chat_flow':
        return flowSelectParam;
      default:
        return spaceNameOriginal || dbParam;
    }
  }, [scene, agent, currentDialogue, dbParam, spaceNameOriginal, flowSelectParam]);

  // 处理发送聊天消息的方法
  const handleChat = async (content: string) => {
    if (isLoading || !content.trim()) return;
    // 如果当前场景为 chat_agent 且未选择代理，则给出提示并返回
    if (scene === 'chat_agent' && !agent) {
      message.warning(t('choice_agent_tip'));
      return;
    }
    try {
      setIsLoading(true);
      // 调用外部传入的 onSubmit 方法发送消息，并附带选择参数
      await onSubmit(content, {
        select_param: selectParam ?? '',
        // incremental,
      });
  } finally {
    // 最终处理：设置isLoading为false，表示加载状态结束
    setIsLoading(false);
  }
};

const handleJson2Obj = (jsonStr: string) => {
  try {
    // 尝试解析JSON字符串为对象
    return JSON.parse(jsonStr);
  } catch (e) {
    // 解析失败时，返回原始JSON字符串
    return jsonStr;
  }
};

const [messageApi, contextHolder] = message.useMessage();

const onCopyContext = async (context: any) => {
  // 去除context中的特定字符串段，返回处理后的字符串
  const pureStr = context?.replace(/\trelations:.*/g, '');
  // 复制pureStr到剪贴板
  const result = copy(pureStr);
  if (result) {
    if (pureStr) {
      // 如果复制成功且pureStr不为空，则显示复制成功消息
      messageApi.open({ type: 'success', content: t('Copy_success') });
    } else {
      // 如果pureStr为空，则显示未复制任何内容的警告消息
      messageApi.open({ type: 'warning', content: t('Copy_nothing') });
    }
  } else {
    // 复制失败时显示复制错误消息
    messageApi.open({ type: 'error', content: t('Copry_error') });
  }
};

const handleRetry = async () => {
  if (isLoading || !docId) {
    // 如果正在加载或者docId不存在，则直接返回
    return;
  }
  // 设置isLoading为true，表示开始加载
  setIsLoading(true);
  // 执行summary函数，等待其完成
  await summary(docId);
  // 加载完成，设置isLoading为false
  setIsLoading(false);
};

useAsyncEffect(async () => {
  // 获取初始化消息
  const initMessage = getInitMessage();
  // 如果存在初始化消息且消息ID与chatId相同
  if (initMessage && initMessage.id === chatId) {
    // 处理初始消息，等待完成
    await handleChat(initMessage.message);
    // 刷新对话列表
    refreshDialogList();
    // 移除本地存储中的初始化消息
    localStorage.removeItem(STORAGE_INIT_MESSAGE_KET);
  }
}, [chatId]);

useEffect(() => {
  // 复制messages数组的副本到tempMessage变量
  let tempMessage: IChatDialogueMessageSchema[] = messages;
  // 如果是图表聊天模式
  if (isChartChat) {
    // 深拷贝messages数组，并对每个消息项进行处理
    tempMessage = cloneDeep(messages).map((item) => {
      // 如果消息角色是'view'且context是字符串类型
      if (item?.role === 'view' && typeof item?.context === 'string') {
        // 将context字段的JSON字符串转换为对象
        item.context = handleJson2Obj(item?.context);
      }
      return item;
    });
  }
  // 过滤并设置显示的消息，仅保留角色为'view'或'human'的消息项
  setShowMessages(tempMessage.filter((item) => ['view', 'human'].includes(item.role)));
}, [isChartChat, messages]);

useEffect(() => {
  // 调用apiInterceptors函数，并处理返回的Promise
  apiInterceptors(getChatFeedBackSelect())
    .then((res) => {
      // 设置selectParam状态为返回结果的第二项，如果不存在则为空对象
      setSelectParam(res[1] ?? {});
    })
    .catch((err) => {
      // 捕获并输出控制台错误信息
      console.log(err);
    });
}, []);

useEffect(() => {
  // 在50毫秒后，滚动到可滚动引用的底部
  setTimeout(() => {
    scrollableRef.current?.scrollTo(0, scrollableRef.current.scrollHeight);
  }, 50);
}, [messages]);

return (
  // 返回空片段
  <>
  </>
);
};

// 导出 Completion 组件作为默认模块导出
export default Completion;
```
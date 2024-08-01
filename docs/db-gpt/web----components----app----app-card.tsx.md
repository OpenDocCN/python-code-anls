# `.\DB-GPT-src\web\components\app\app-card.tsx`

```py
import React, { useContext, useEffect, useState } from 'react';
import { Modal } from 'antd';
import { apiInterceptors, collectApp, delApp, newDialogue, unCollectApp } from '@/client/api';
import { IApp } from '@/types/app';
import { DeleteFilled, MessageFilled, StarFilled, WarningOutlined } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { useRouter } from 'next/router';
import { ChatContext } from '@/app/chat-context';
import GPTCard from '../common/gpt-card';

interface IProps {
  updateApps: (data?: { is_collected: boolean }) => void;  // 函数签名，用于更新应用程序状态
  app: IApp;  // 应用程序对象
  handleEdit: (app: any) => void;  // 处理编辑应用程序的回调函数
  isCollected: boolean;  // 是否已收藏的标志位
}

const { confirm } = Modal;

export default function AppCard(props: IProps) {
  const { updateApps, app, handleEdit, isCollected } = props;  // 解构props，获取更新函数、应用程序对象、处理编辑函数、收藏状态
  const { model } = useContext(ChatContext);  // 获取聊天上下文中的模型信息
  const router = useRouter();  // 获取路由器对象

  const [isCollect, setIsCollect] = useState<string>(app.is_collected);  // 状态管理是否已收藏的状态

  const { t } = useTranslation();  // 获取国际化翻译函数

  // 语言映射表
  const languageMap = {
    en: t('English'),  // 英语对应的翻译
    zh: t('Chinese'),  // 中文对应的翻译
  };

  // 显示删除确认对话框
  const showDeleteConfirm = () => {
    confirm({
      title: t('Tips'),  // 对话框标题，提示
      icon: <WarningOutlined />,  // 提示图标
      content: `do you want delete the application?`,  // 对话框内容，确认是否删除应用程序
      okText: 'Yes',  // 确认按钮文本
      okType: 'danger',  // 确认按钮类型（危险）
      cancelText: 'No',  // 取消按钮文本
      async onOk() {
        await apiInterceptors(delApp({ app_code: app.app_code }));  // 调用API删除应用程序
        updateApps(isCollected ? { is_collected: isCollected } : undefined);  // 更新应用程序列表
      },
    });
  };

  // 当应用程序对象变化时更新是否收藏状态
  useEffect(() => {
    setIsCollect(app.is_collected);
  }, [app]);

  // 收藏或取消收藏应用程序
  const collect = async () => {
    const [error] = await apiInterceptors(isCollect === 'true' ? unCollectApp({ app_code: app.app_code }) : collectApp({ app_code: app.app_code }));
    if (error) return;
    updateApps(isCollected ? { is_collected: isCollected } : undefined);  // 更新应用程序列表
    setIsCollect(isCollect === 'true' ? 'false' : 'true');  // 更新是否已收藏状态
  };

  // 处理与客服聊天
  const handleChat = async () => {
    setAgentToChat?.(app.app_code);  // 设置聊天代理
    const [, res] = await apiInterceptors(newDialogue({ chat_mode: 'chat_agent' }));  // 创建新的对话
    if (res) {
      router.push(`/chat/?scene=chat_agent&id=${res.conv_uid}${model ? `&model=${model}` : ''}`);  // 导航到聊天页面
    }
  };

  return (
    // 渲染一个名为 app.app_name 的卡片组件
    <GPTCard
      // 设置卡片的标题为 app.app_name
      title={app.app_name}
      // 设置卡片的图标路径为 '/icons/node/vis.png'
      icon={'/icons/node/vis.png'}
      // 禁用卡片的图标边框
      iconBorder={false}
      // 设置卡片的描述为 app.app_describe
      desc={app.app_describe}
      // 设置卡片的标签数组，包括 app.language 和 app.team_mode 的文本与颜色
      tags={[
        { text: languageMap[app.language], color: 'default' },
        { text: app.team_mode, color: 'default' },
      ]}
      // 当卡片被点击时调用 handleEdit(app) 函数
      onClick={() => {
        handleEdit(app);
      }}
      // 设置卡片的操作按钮，包括 'Chat' 按钮，显示 <MessageFilled /> 图标，点击时调用 handleChat 函数
      operations={[
        {
          label: t('Chat'),
          children: <MessageFilled />,
          onClick: handleChat,
        },
        // 设置 'collect' 按钮，显示 <StarFilled /> 图标，根据 app.is_collected 属性不同显示不同的颜色，
        // 灰色或黄色，点击时调用 collect 函数
        {
          label: t('collect'),
          children: <StarFilled className={app.is_collected === 'false' ? 'text-gray-400' : 'text-yellow-400'} />,
          onClick: collect,
        },
        // 设置 'Delete' 按钮，显示 <DeleteFilled /> 图标，点击时显示删除确认框
        {
          label: t('Delete'),
          children: <DeleteFilled />,
          onClick: () => {
            showDeleteConfirm();
          },
        },
      ]}
    />
}



# 闭合一个代码块，这里是一个函数的结束
```
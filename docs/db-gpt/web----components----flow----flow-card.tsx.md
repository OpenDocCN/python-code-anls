# `.\DB-GPT-src\web\components\flow\flow-card.tsx`

```py
import { apiInterceptors, deleteFlowById, newDialogue } from '@/client/api';
// 从客户端 API 中导入所需的函数和对象

import { IFlow } from '@/types/flow';
// 导入流程类型定义 IFlow

import {
  CopyFilled,
  DeleteFilled,
  EditFilled,
  ExclamationCircleFilled,
  ExclamationCircleOutlined,
  MessageFilled,
  WarningOutlined,
} from '@ant-design/icons';
// 从 Ant Design 图标库中导入需要使用的图标组件

import { Modal, Tooltip } from 'antd';
// 导入 Ant Design 的模态框和工具提示组件

import React, { useContext } from 'react';
// 导入 React 和 useContext 钩子

import { useTranslation } from 'react-i18next';
// 导入国际化翻译钩子 useTranslation

import FlowPreview from './preview-flow';
// 导入自定义的流程预览组件 FlowPreview

import { useRouter } from 'next/router';
// 导入 Next.js 的路由钩子 useRouter

import GptCard from '../common/gpt-card';
// 导入通用的 GPT 卡片组件 GptCard

import { ChatContext } from '@/app/chat-context';
// 从应用上下文中导入聊天上下文 ChatContext

import qs from 'querystring';
// 导入 querystring 模块

interface FlowCardProps {
  flow: IFlow;
  deleteCallback: (uid: string) => void;
  onCopy: (flow: IFlow) => void;
}
// 定义 FlowCardProps 接口，描述流程卡片组件的属性

const FlowCard: React.FC<FlowCardProps> = ({ flow, onCopy, deleteCallback }) => {
  // 定义 React 函数组件 FlowCard，接受 FlowCardProps 属性

  const { model } = useContext(ChatContext);
  // 从 ChatContext 上下文中获取 model 对象

  const { t } = useTranslation();
  // 使用 useTranslation 钩子获取国际化翻译函数 t

  const [modal, contextHolder] = Modal.useModal();
  // 使用 Modal.useModal() 创建模态框 modal 和 contextHolder

  const router = useRouter();
  // 使用 useRouter 获取路由对象 router

  async function deleteFlow() {
    // 定义异步函数 deleteFlow，用于删除流程

    const [, , res] = await apiInterceptors(deleteFlowById(flow.uid));
    // 使用 deleteFlowById 删除指定流程的 API 请求，并通过 apiInterceptors 进行拦截

    if (res?.success) {
      // 如果删除请求成功

      deleteCallback && deleteCallback(flow.uid);
      // 调用删除回调函数 deleteCallback 删除流程
    }
  }

  function cardClick() {
    // 定义 cardClick 函数，处理卡片点击事件

    router.push('/flow/canvas?id=' + flow.uid);
    // 使用 router.push 实现路由跳转到指定流程详情页
  }

  const handleChat = async () => {
    // 定义异步函数 handleChat，处理聊天功能触发事件

    const [, res] = await apiInterceptors(newDialogue({ chat_mode: 'chat_agent' }));
    // 创建新对话的 API 请求，并通过 apiInterceptors 进行拦截

    if (res) {
      // 如果返回结果有效

      const queryStr = qs.stringify({
        scene: 'chat_flow',
        id: res.conv_uid,
        model: model,
        select_param: flow.uid,
      });
      // 构造查询字符串包含场景、对话 ID、模型和选择参数

      router.push(`/chat?${queryStr}`);
      // 使用 router.push 实现路由跳转到聊天页面，带上查询字符串
    }
  };

  const handleDel = () => {
    // 定义 handleDel 函数，处理删除确认弹窗的事件

    modal.confirm({
      title: t('Tips'),
      // 设置弹窗标题为 Tips，使用 t 函数进行国际化翻译

      icon: <WarningOutlined />,
      // 设置弹窗图标为警告图标

      content: t('delete_flow_confirm'),
      // 设置弹窗内容为删除流程确认信息，使用 t 函数进行国际化翻译

      okText: 'Yes',
      // 设置确认按钮文本为 Yes

      okType: 'danger',
      // 设置确认按钮类型为危险按钮

      cancelText: 'No',
      // 设置取消按钮文本为 No

      async onOk() {
        // 定义确认按钮点击后的异步处理函数 onOk

        deleteFlow();
        // 调用 deleteFlow 函数执行流程删除操作
      },
    });
  };

  return (
    <>
      {contextHolder}
      <GptCard
        className="w-[26rem] max-w-full"
        title={flow.name}
        desc={flow.description}
        tags={[
          // 标签数组，包含流程来源、编辑状态和状态信息
          { text: flow.source, color: flow.source === 'DBGPT-WEB' ? 'green' : 'blue', border: true },
          { text: flow.editable ? 'Editable' : 'Can not Edit', color: flow.editable ? 'green' : 'gray', border: true },
          {
            text: (
              // 如果流程有错误消息，则使用 Tooltip 显示错误消息和状态图标；否则仅显示状态信息
              <>
                {flow.error_message ? (
                  <Tooltip placement="bottom" title={flow.error_message}>
                    {flow.state}
                    <ExclamationCircleOutlined className="ml-1" />
                  </Tooltip>
                ) : (
                  flow.state
                )}
              </>
            ),
            // 根据状态设置不同的颜色
            color: flow.state === 'load_failed' ? 'red' : flow.state === 'running' ? 'green' : 'blue',
            border: true,
          },
        ]}
        operations={[
          // 操作数组，包含聊天、编辑、复制和删除操作
          {
            label: t('Chat'),         // 操作标签：聊天
            children: <MessageFilled />,  // 操作图标：消息填充
            onClick: handleChat,      // 点击操作时触发 handleChat 函数
          },
          {
            label: t('Edit'),         // 操作标签：编辑
            children: <EditFilled />, // 操作图标：编辑填充
            onClick: cardClick,       // 点击操作时触发 cardClick 函数
          },
          {
            label: t('Copy'),         // 操作标签：复制
            children: <CopyFilled />, // 操作图标：复制填充
            onClick: () => {
              onCopy(flow);           // 点击操作时触发 onCopy 函数，传递当前流程数据
            },
          },
          {
            label: t('Delete'),       // 操作标签：删除
            children: <DeleteFilled />, // 操作图标：删除填充
            onClick: handleDel,       // 点击操作时触发 handleDel 函数
          },
        ]}
      >
        <div className="w-full h-40 shadow-[inset_0_0_16px_rgba(50,50,50,.05)]">
          <FlowPreview flowData={flow.flow_data} />  // 渲染流程预览组件，传递流程数据
        </div>
      </GptCard>
    </>
  );
};

export default FlowCard;
```
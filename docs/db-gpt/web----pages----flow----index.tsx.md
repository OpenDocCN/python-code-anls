# `.\DB-GPT-src\web\pages\flow\index.tsx`

```py
import { addFlow, apiInterceptors, getFlows } from '@/client/api';
// 导入用于添加流程、API 拦截器和获取流程的函数

import MyEmpty from '@/components/common/MyEmpty';
// 导入自定义空状态组件 MyEmpty

import MuiLoading from '@/components/common/loading';
// 导入通用加载组件 MuiLoading

import FlowCard from '@/components/flow/flow-card';
// 导入流程卡片组件 FlowCard

import { IFlow, IFlowUpdateParam } from '@/types/flow';
// 导入流程类型 IFlow 和流程更新参数类型 IFlowUpdateParam

import { PlusOutlined } from '@ant-design/icons';
// 导入 Ant Design 的加号图标组件 PlusOutlined

import { Button, Checkbox, Form, Input, Modal, message } from 'antd';
// 导入 Ant Design 的按钮、复选框、表单、输入框、模态框和消息组件

import Link from 'next/link';
// 导入 Next.js 的 Link 组件

import React, { useEffect, useRef, useState } from 'react';
// 导入 React 库中的相关 Hook 和核心模块

import { useTranslation } from 'react-i18next';
// 导入用于国际化的 React Hook useTranslation

function Flow() {
  const { t } = useTranslation();
  // 使用 useTranslation Hook 获取 t 函数用于国际化

  const [showModal, setShowModal] = useState(false);
  // 定义用于控制模态框显示的状态变量

  const [loading, setLoading] = useState(false);
  // 定义用于控制加载状态的状态变量

  const [flowList, setFlowList] = useState<Array<IFlow>>([]);
  // 定义用于存储流程列表的状态变量

  const [deploy, setDeploy] = useState(false);
  // 定义用于存储是否部署的状态变量

  const [messageApi, contextHolder] = message.useMessage();
  // 使用 message 的 useMessage 方法获取消息 API 和 contextHolder

  const [form] = Form.useForm<Pick<IFlow, 'label' | 'name'>>();
  // 使用 Form 的 useForm 方法创建表单实例 form，限定字段为流程的 label 和 name

  const copyFlowTemp = useRef<IFlow>();
  // 使用 useRef 创建保存流程副本的引用 copyFlowTemp

  async function getFlowList() {
    setLoading(true);
    // 设置加载状态为 true
    const [_, data] = await apiInterceptors(getFlows());
    // 调用 API 获取流程列表，并通过 API 拦截器处理返回值
    setLoading(false);
    // 设置加载状态为 false
    setFlowList(data?.items ?? []);
    // 更新流程列表数据
  }

  useEffect(() => {
    getFlowList();
    // 组件挂载时调用获取流程列表的方法
  }, []);

  function updateFlowList(uid: string) {
    setFlowList((flows) => flows.filter((flow) => flow.uid !== uid));
    // 根据流程的 uid 更新流程列表，移除指定 uid 的流程
  }

  const handleCopy = (flow: IFlow) => {
    copyFlowTemp.current = flow;
    // 将选中的流程赋值给 copyFlowTemp 引用
    form.setFieldValue('label', `${flow.label} Copy`);
    // 设置表单字段 label 为流程名称后加上 Copy
    form.setFieldValue('name', `${flow.name}_copy`);
    // 设置表单字段 name 为流程名称后加上 _copy
    setDeploy(false);
    // 设置 deploy 状态为 false
    setShowModal(true);
    // 显示模态框
  };

  const onFinish = async (val: { name: string; label: string }) => {
    if (!copyFlowTemp.current) return;
    // 如果没有选中的流程副本，则直接返回
    const { source, uid, dag_id, gmt_created, gmt_modified, state, ...params } = copyFlowTemp.current;
    // 解构流程副本中的属性
    const data: IFlowUpdateParam = {
      ...params,
      editable: true,
      state: deploy ? 'deployed' : 'developing',
      ...val,
    };
    // 创建流程更新参数对象
    const [err] = await apiInterceptors(addFlow(data));
    // 调用 API 添加流程，并通过 API 拦截器处理返回值
    if (!err) {
      messageApi.success(t('save_flow_success'));
      // 显示保存流程成功的消息
      setShowModal(false);
      // 隐藏模态框
      getFlowList();
      // 重新获取流程列表
    }
  };

  return (
    <div className="relative p-4 md:p-6 min-h-full overflow-y-auto">
      {contextHolder}
      {/* 显示全局的上下文信息 */}
      <MuiLoading visible={loading} />
      {/* 根据 loading 状态显示加载中的组件 */}
      <div className="mb-4">
        {/* 跳转到 /flow/canvas 的链接按钮 */}
        <Link href="/flow/canvas">
          <Button type="primary" className="flex items-center" icon={<PlusOutlined />}>
            New AWEL Flow
          </Button>
        </Link>
      </div>
      {/* 显示已有流程卡片列表 */}
      <div className="flex flex-wrap gap-2 md:gap-4 justify-start items-stretch">
        {flowList.map((flow) => (
          // 渲染每个流程卡片，关联删除和复制操作
          <FlowCard key={flow.uid} flow={flow} deleteCallback={updateFlowList} onCopy={handleCopy} />
        ))}
        {/* 如果没有流程卡片，则显示空状态 */}
        {flowList.length === 0 && <MyEmpty description="No flow found" />}
      </div>
      {/* 显示复制流程的模态框 */}
      <Modal
        open={showModal}
        title="Copy AWEL Flow"
        onCancel={() => {
          setShowModal(false);
        }}
        footer={false}
      >
        {/* 表单用于输入新流程的名称、标签和部署选项 */}
        <Form form={form} onFinish={onFinish} className="mt-6">
          <Form.Item name="name" label="Name" rules={[{ required: true }]}>
            <Input />
          </Form.Item>
          <Form.Item name="label" label="Label" rules={[{ required: true }]}>
            <Input />
          </Form.Item>
          <Form.Item label="Deploy">
            {/* 复选框用于选择是否部署流程 */}
            <Checkbox
              value={deploy}
              onChange={(e) => {
                const val = e.target.checked;
                setDeploy(val);
              }}
            />
          </Form.Item>
          <div className="flex justify-end">
            {/* 提交按钮 */}
            <Button type="primary" htmlType="submit">
              {t('Submit')}
            </Button>
          </div>
        </Form>
      </Modal>
    </div>
}

export default Flow;
```
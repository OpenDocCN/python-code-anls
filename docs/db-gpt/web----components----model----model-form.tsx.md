# `.\DB-GPT-src\web\components\model\model-form.tsx`

```py
import { apiInterceptors, getSupportModels, startModel } from '@/client/api';
// 导入 API 调用相关函数和对象

import { SupportModel, SupportModelParams } from '@/types/model';
// 导入模型和模型参数的类型定义

import { Button, Form, Select, Tooltip, message } from 'antd';
// 导入 Ant Design 组件中的 Button、Form、Select、Tooltip 和 message 组件

import { useEffect, useState } from 'react';
// 导入 React Hooks 中的 useEffect 和 useState

import { useTranslation } from 'react-i18next';
// 导入 React 国际化组件中的 useTranslation 函数

import { renderModelIcon } from '@/components/chat/header/model-selector';
// 导入自定义组件中的 renderModelIcon 函数

import ModelParams from './model-params';
// 导入当前目录下的 model-params 组件

const { Option } = Select;
// 从 Select 组件中解构出 Option 组件

function ModelForm({ onCancel, onSuccess }: { onCancel: () => void; onSuccess: () => void }) {
  // 定义 ModelForm 函数组件，接受 onCancel 和 onSuccess 两个函数作为 props

  const { t } = useTranslation();
  // 使用 useTranslation Hook 获取 t 函数，用于国际化翻译

  const [models, setModels] = useState<Array<SupportModel> | null>([]);
  // 使用 useState Hook 定义 models 状态，存储支持的模型列表，默认为空数组或 null

  const [selectedModel, setSelectedModel] = useState<SupportModel>();
  // 使用 useState Hook 定义 selectedModel 状态，存储当前选中的模型对象

  const [params, setParams] = useState<Array<SupportModelParams> | null>(null);
  // 使用 useState Hook 定义 params 状态，存储当前选中模型的参数列表，默认为空数组或 null

  const [loading, setLoading] = useState<boolean>(false);
  // 使用 useState Hook 定义 loading 状态，表示数据加载状态，默认为 false

  const [form] = Form.useForm();
  // 使用 Form 自定义 Hook 创建表单实例 form

  async function getModels() {
    // 异步函数 getModels，用于获取支持的模型列表

    const [, res] = await apiInterceptors(getSupportModels());
    // 使用 apiInterceptors 发起 API 请求获取模型列表数据，并解构出响应数据 res

    if (res && res.length) {
      // 如果 res 有数据且长度大于 0
      setModels(
        // 设置 models 状态
        res.sort((a: SupportModel, b: SupportModel) => {
          // 对获取到的模型列表进行排序
          if (a.enabled && !b.enabled) {
            return -1;
          } else if (!a.enabled && b.enabled) {
            return 1;
          } else {
            return a.model.localeCompare(b.model);
          }
        }),
      );
    }
    setModels(res);
    // 更新模型列表状态
  }

  useEffect(() => {
    // useEffect Hook，组件挂载时调用 getModels 函数获取模型列表
    getModels();
  }, []);

  function handleChange(value: string, option: any) {
    // 处理 Select 组件的选择变化事件，更新 selectedModel 和 params 状态
    setSelectedModel(option.model);
    setParams(option.model.params);
  }

  async function onFinish(values: any) {
    // 处理表单提交事件，启动选定的模型
    if (!selectedModel) {
      return;
    }
    delete values.model;
    // 删除 values 对象中的 model 属性

    setLoading(true);
    // 设置 loading 状态为 true，表示开始加载数据

    const [, , data] = await apiInterceptors(
      // 使用 apiInterceptors 发起 API 请求启动模型
      startModel({
        host: selectedModel.host,
        port: selectedModel.port,
        model: selectedModel.model,
        worker_type: selectedModel?.worker_type,
        params: values,
      }),
    );

    setLoading(false);
    // 设置 loading 状态为 false，表示数据加载完成

    if (data?.success === true) {
      // 如果 API 请求成功启动模型
      onSuccess && onSuccess();
      // 调用 onSuccess 回调函数
      return message.success(t('start_model_success'));
      // 显示成功消息
    }
  }

  return (
    <Form labelCol={{ span: 8 }} wrapperCol={{ span: 16 }} onFinish={onFinish} form={form}>
      {/* 表单组件，设置标签列宽度为8格，包装列宽度为16格，提交表单时调用 onFinish 函数 */}
      <Form.Item label="Model" name="model" rules={[{ required: true, message: t('model_select_tips') }]}>
        {/* 表单项，标签为"Model"，字段名为"model"，设置必填规则，验证失败时提示信息为 model_select_tips */}
        <Select showSearch onChange={handleChange}>
          {/* 下拉选择框组件，当搜索时显示，选择项变化时调用 handleChange 函数 */}
          {models?.map((model) => (
            {/* 遍历 models 数组中的每个模型对象 */}
            <Option key={model.model} value={model.model} label={model.model} model={model} disabled={!model.enabled}>
              {/* 下拉选项，每个选项的唯一键为模型名称，选项值为模型名称，禁用状态取决于 model.enabled */}
              {renderModelIcon(model.model)}
              {/* 渲染模型图标 */}
              <Tooltip title={model.enabled ? model.model : t('download_model_tip')}>
                {/* 提示框，标题根据模型是否可用显示模型名称或下载提示 */}
                <span className="ml-2">{model.model}</span>
              </Tooltip>
              <Tooltip title={model.enabled ? `${model.host}:${model.port}` : t('download_model_tip')}>
                {/* 提示框，内容根据模型是否可用显示主机地址和端口号或下载提示 */}
                <p className="inline-block absolute right-4">
                  <span>{model.host}:</span>
                  <span>{model.port}</span>
                </p>
              </Tooltip>
            </Option>
          ))}
        </Select>
      </Form.Item>
      <ModelParams params={params} form={form} />
      {/* 自定义组件，用于显示模型参数，传递参数和表单对象作为属性 */}
      <div className="flex justify-center">
        {/* 弹性盒子布局，内容居中显示 */}
        <Button type="primary" htmlType="submit" loading={loading}>
          {/* 提交按钮，类型为主要按钮，点击时提交表单，加载状态为 loading */}
          {t('submit')}
          {/* 显示国际化文本"submit" */}
        </Button>
        <Button className="ml-10" onClick={onCancel}>
          {/* 普通按钮，点击时执行 onCancel 函数 */}
          Cancel
          {/* 显示"Cancel"文本 */}
        </Button>
      </div>
    </Form>
}

export default ModelForm;
```
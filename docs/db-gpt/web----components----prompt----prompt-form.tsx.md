# `.\DB-GPT-src\web\components\prompt\prompt-form.tsx`

```py
import React, { Ref, forwardRef, useEffect, useState } from 'react';
import { Form, Input, Spin, Select, FormInstance } from 'antd';
import { useTranslation } from 'react-i18next';
import { IPrompt } from '@/types/prompt';

interface IProps {
  prompt?: IPrompt;  // 可选属性：表示可能传入的提示对象
  onFinish: (prompt: IPrompt) => void;  // 完成回调函数：接收并处理完成编辑的提示对象
  scenes?: Array<Record<string, string>>;  // 场景数组：包含键值对形式的场景信息
}

export default forwardRef(function PromptForm(props: IProps, ref: Ref<FormInstance<any>> | undefined) {
  const { t } = useTranslation();  // 多语言翻译函数
  const [form] = Form.useForm();  // 使用表单钩子，获取表单实例
  const { prompt, onFinish, scenes } = props;  // 解构获取传入的提示对象、完成回调函数和场景数组
  const [loading, setLoading] = useState<boolean>(false);  // 加载状态钩子：控制表单提交时的加载状态

  useEffect(() => {
    if (prompt) {
      form.setFieldsValue(prompt);  // 初始化表单值：如果有传入提示对象，则设置表单的初始值
    }
  }, []);

  const submit = async () => {
    const values = form.getFieldsValue();  // 获取表单当前值
    setLoading(true);  // 设置加载状态为 true，显示加载中状态
    await onFinish(values);  // 调用完成回调函数，传入表单当前值
    setLoading(false);  // 设置加载状态为 false，结束加载中状态
  };

  return (
    <Spin spinning={loading}>  // 自旋加载组件：根据加载状态显示加载动画
      <Form form={form} ref={ref} name={`prompt-item-${prompt?.prompt_name || 'new'}`} layout="vertical" className="mt-4" onFinish={submit}>
        {/* 表单项：选择聊天场景 */}
        <Form.Item name="chat_scene" label={t('Prompt_Info_Scene')} rules={[{ required: true, message: t('Please_Input') + t('Prompt_Info_Scene') }]}>
          <Select options={scenes}></Select>  // 下拉选择框：根据场景数组渲染选项
        </Form.Item>
        {/* 表单项：输入子聊天场景 */}
        <Form.Item
          name="sub_chat_scene"
          label={t('Prompt_Info_Sub_Scene')}
          rules={[{ required: true, message: t('Please_Input') + t('Prompt_Info_Sub_Scene') }]}
        >
          <Input />  // 输入框：用于输入子聊天场景
        </Form.Item>
        {/* 表单项：输入提示名称 */}
        <Form.Item name="prompt_name" label={t('Prompt_Info_Name')} rules={[{ required: true, message: t('Please_Input') + t('Prompt_Info_Name') }]}>
          <Input disabled={!!prompt} />  // 输入框：用于输入提示名称，如果有提示对象则禁用
        </Form.Item>
        {/* 表单项：输入提示内容 */}
        <Form.Item
          name="content"
          label={t('Prompt_Info_Content')}
          rules={[{ required: true, message: t('Please_Input') + t('Prompt_Info_Content') }]}
        >
          <Input.TextArea rows={6} />  // 文本域：用于输入提示内容，显示多行文本输入框
        </Form.Item>
      </Form>
    </Spin>
  );
});
```
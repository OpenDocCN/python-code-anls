# `.\DB-GPT-src\web\components\knowledge\space-form.tsx`

```py
import { addSpace, apiInterceptors } from '@/client/api';
import { IStorage, StepChangeParams } from '@/types/knowledge';
import { Button, Form, Input, Spin, Select } from 'antd';
import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';

type FieldType = {
  spaceName: string;
  owner: string;
  description: string;
  storage: string;
  field: string;
};

type IProps = {
  handleStepChange: (params: StepChangeParams) => void;  // Props接口定义，包含处理步骤改变的函数
  spaceConfig: IStorage | null;  // Props接口定义，表示存储配置信息或为空
};

export default function SpaceForm(props: IProps) {  // React组件，用于展示和处理空间表单
  const { t } = useTranslation();  // 国际化函数，用于获取翻译文本
  const { handleStepChange, spaceConfig } = props;  // 从props中解构出处理步骤改变的函数和存储配置信息
  const [spinning, setSpinning] = useState<boolean>(false);  // 状态钩子，用于控制加载状态的布尔值
  const [storage, setStorage] = useState<string>();  // 状态钩子，用于存储选择的存储类型名称

  const [form] = Form.useForm();  // 使用antd的Form钩子，获取表单实例

  useEffect(() => {
    // 当spaceConfig变化时执行的副作用函数
    form.setFieldValue('storage', spaceConfig?.[0].name);  // 设置表单字段storage的初始值为spaceConfig中第一个元素的名称
    setStorage(spaceConfig?.[0].name);  // 设置当前选择的存储类型名称为spaceConfig中第一个元素的名称
  }, [spaceConfig]);

  const handleStorageChange = (data: string) => {
    // 处理存储类型变化的函数
    setStorage(data);  // 设置当前选择的存储类型名称为传入的data值
  };

  const handleFinish = async (fieldsValue: FieldType) => {
    // 处理表单提交的函数
    const { spaceName, owner, description, storage, field } = fieldsValue;  // 解构出表单字段值
    setSpinning(true);  // 设置加载状态为true，显示加载中状态

    let vector_type = storage;  // 将存储类型赋值给vector_type变量
    let domain_type = field;  // 将领域类型赋值给domain_type变量

    const [_, data, res] = await apiInterceptors(
      // 调用API请求拦截器，发送添加空间的请求
      addSpace({
        name: spaceName,
        vector_type: vector_type,
        owner,
        desc: description,
        domain_type: domain_type,
      }),
    );

    setSpinning(false);  // 设置加载状态为false，隐藏加载中状态

    const is_financial = domain_type === 'FinancialReport';  // 检查领域类型是否为FinancialReport

    // 如果请求成功且领域类型是FinancialReport，根据情况更新步骤和文档类型
    res?.success && handleStepChange({ label: 'forward', spaceName, pace: is_financial ? 2 : 1, docType: is_financial ? 'DOCUMENT' : '' });
  };

  return (
    <Spin spinning={spinning}>
      {/* 使用 Spin 组件显示加载状态，spinning 属性控制是否显示加载状态 */}
      <Form
        form={form}
        size="large"
        className="mt-4"
        layout="vertical"
        name="basic"
        initialValues={{ remember: true }}
        autoComplete="off"
        onFinish={handleFinish}
      >
        {/* 表单组件，使用 form 控制表单数据，size 设置表单尺寸，className 添加额外样式 */}
        <Form.Item<FieldType>
          label={t('Knowledge_Space_Name')}
          name="spaceName"
          rules={[
            { required: true, message: t('Please_input_the_name') },
            // 自定义验证规则，确保输入只包含中文、英文、数字、下划线和短横线
            () => ({
              validator(_, value) {
                if (/[^\u4e00-\u9fa50-9a-zA-Z_-]/.test(value)) {
                  return Promise.reject(new Error(t('the_name_can_only_contain')));
                }
                return Promise.resolve();
              },
            }),
          ]}
        >
          {/* 输入框，label 显示标签文字，placeholder 设置输入框占位文字 */}
          <Input className="mb-5 h-12" placeholder={t('Please_input_the_name')} />
        </Form.Item>
        <Form.Item<FieldType> label={t('Owner')} name="owner" rules={[{ required: true, message: t('Please_input_the_owner') }]}>
          {/* 输入框，label 显示标签文字，placeholder 设置输入框占位文字 */}
          <Input className="mb-5 h-12" placeholder={t('Please_input_the_owner')} />
        </Form.Item>
        <Form.Item<FieldType> label={t('Storage')} name="storage" rules={[{ required: true, message: t('Please_select_the_storage') }]}>
          {/* 下拉选择框，label 显示标签文字，placeholder 设置占位文字，onChange 处理选择变化 */}
          <Select className="mb-5 h-12" placeholder={t('Please_select_the_storage')} onChange={handleStorageChange}>
            {/* 根据 spaceConfig 渲染选项列表 */}
            {spaceConfig?.map((item) => {
              return <Select.Option value={item.name}>{item.desc}</Select.Option>;
            })}
          </Select>
        </Form.Item>
        <Form.Item<FieldType> label={t('Domain')} name="field" rules={[{ required: true, message: t('Please_select_the_domain_type') }]}>
          {/* 下拉选择框，label 显示标签文字，placeholder 设置占位文字 */}
          <Select className="mb-5 h-12" placeholder={t('Please_select_the_domain_type')}>
            {/* 根据 storage 选择对应的 domain_types 渲染选项列表 */}
            {spaceConfig
              ?.find((item) => item.name === storage)
              ?.domain_types.map((item) => {
                return <Select.Option value={item.name}>{item.desc}</Select.Option>;
              })}
          </Select>
        </Form.Item>
        <Form.Item<FieldType> label={t('Description')} name="description" rules={[{ required: true, message: t('Please_input_the_description') }]}>
          {/* 输入框，label 显示标签文字，placeholder 设置输入框占位文字 */}
          <Input className="mb-5 h-12" placeholder={t('Please_input_the_description')} />
        </Form.Item>
        <Form.Item>
          {/* 提交按钮 */}
          <Button type="primary" htmlType="submit">
            {t('Next')}
          </Button>
        </Form.Item>
      </Form>
    </Spin>
# 闭合大括号，表示代码块的结束
```
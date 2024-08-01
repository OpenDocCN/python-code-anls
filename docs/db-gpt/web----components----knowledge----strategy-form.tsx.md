# `.\DB-GPT-src\web\components\knowledge\strategy-form.tsx`

```py
import { IChunkStrategyResponse } from '@/types/knowledge';
import { Alert, Checkbox, Form, FormListFieldData, Input, InputNumber, Radio, RadioChangeEvent } from 'antd';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
const { TextArea } = Input;

type IProps = {
  strategies: Array<IChunkStrategyResponse>;  // 定义组件的 props 类型，包括策略数组和文档类型等信息
  docType: string;  // 文档类型
  field: FormListFieldData;  // 表单字段的数据结构
  fileName: string;  // 文件名
};

/**
 * 根据文档类型和文件后缀渲染策略
 */
export default function StrategyForm({ strategies, docType, fileName, field }: IProps) {
  const [selectedStrategy, setSelectedStrategy] = useState<string>();  // 状态钩子，用于存储选中的策略
  let filleSuffix = '';  // 存储文件后缀的变量

  if (docType === 'DOCUMENT') {  // 如果文档类型为 'DOCUMENT'
    // 根据文件名获取文件后缀
    const arr = fileName.split('.');
    filleSuffix = arr[arr.length - 1];
  }

  // 根据文件后缀过滤可用的策略数组
  const ableStrategies = filleSuffix ? strategies.filter((i) => i.suffix.indexOf(filleSuffix) > -1) : strategies;

  const { t } = useTranslation();  // i18n 国际化钩子
  const DEFAULT_STRATEGY = {
    strategy: 'Automatic',  // 默认策略名称
    name: t('Automatic'),  // 默认策略显示名称，国际化
    desc: t('Automatic_desc'),  // 默认策略描述，国际化
  };

  // 处理单选按钮变更事件
  function radioChange(e: RadioChangeEvent) {
    setSelectedStrategy(e.target.value);  // 更新选中的策略状态
  }

  // 渲染策略参数表单
  function renderStrategyParamForm() {
    if (!selectedStrategy) {
      return null;  // 如果没有选中策略，返回空
    }
    if (selectedStrategy === DEFAULT_STRATEGY.strategy) {
      return <p className="my-4">{DEFAULT_STRATEGY.desc}</p>;  // 如果选中了默认策略，显示默认策略的描述
    }
    // 获取选中策略的参数数组
    const parameters = ableStrategies?.filter((i) => i.strategy === selectedStrategy)[0].parameters;
    if (!parameters || !parameters.length) {
      return <Alert className="my-2" type="warning" message={t('No_parameter')} />;  // 如果没有参数，显示警告信息
    }
    // 渲染参数表单项
    return (
      <div className="mt-2">
        {parameters?.map((param) => (
          <Form.Item
            key={`param_${param.param_name}`}
            label={param.param_name}  // 参数名称
            name={[field!.name, 'chunk_parameters', param.param_name]}  // 表单项的名称路径
            rules={[{ required: true, message: t('Please_input_the_name') }]}  // 表单验证规则
            initialValue={param.default_value}  // 初始值
            valuePropName={param.param_type === 'boolean' ? 'checked' : 'value'}  // 值属性名称，根据参数类型决定
            tooltip={param.description}  // 提示信息
          >
            {renderParamByType(param.param_type)}  // 根据参数类型渲染不同的输入组件
          </Form.Item>
        ))}
      </div>
    );
  }

  // 根据参数类型渲染不同的输入组件
  function renderParamByType(type: string) {
    switch (type) {
      case 'int':
        return <InputNumber className="w-full" min={1} />;  // 整数输入框
      case 'string':
        return <TextArea className="w-full" rows={2} />;  // 文本域输入框
      case 'boolean':
        return <Checkbox />;  // 复选框
    }
  }

  return (
    <Form.Item name={[field!.name, 'chunk_parameters', 'chunk_strategy']} initialValue={DEFAULT_STRATEGY.strategy}>
      {/* 创建一个表单项，名称是由 field!.name 和 'chunk_parameters' 组成的数组，初始值是 DEFAULT_STRATEGY.strategy */}
      <Radio.Group style={{ marginTop: 16 }} onChange={radioChange}>
        {/* 创建一个单选按钮组件组，设置样式 marginTop 为 16 像素，当选项改变时触发 radioChange 函数 */}
        <Radio value={DEFAULT_STRATEGY.strategy}>{DEFAULT_STRATEGY.name}</Radio>
        {/* 创建一个单选按钮，值为 DEFAULT_STRATEGY.strategy，显示文本为 DEFAULT_STRATEGY.name */}
        {ableStrategies.map((strategy) => (
          // 遍历 ableStrategies 数组，对于每个 strategy 创建一个单选按钮
          <Radio key={`strategy_radio_${strategy.strategy}`} value={strategy.strategy}>
            {strategy.name}
          </Radio>
        ))}
      </Radio.Group>
    </Form.Item>
    {renderStrategyParamForm()}
    {/* 渲染 renderStrategyParamForm 函数返回的内容 */}
}



# 这是一个单独的右大括号 '}'，用于结束某个代码块或语句。
```
# `.\DB-GPT-src\web\components\model\model-params.tsx`

```py
// 导入必要的模块和类型定义
import { SupportModelParams } from '@/types/model';
import { Checkbox, Form, FormInstance, Input, InputNumber } from 'antd';
import { useEffect } from 'react';

// 定义参数的键值对类型
interface ParamValues {
  [key: string]: string | number | boolean;
}

// 声明组件 ModelParams，接收参数和表单实例作为 props
function ModelParams({ params, form }: { params: Array<SupportModelParams> | null; form: FormInstance<any> }) {
  // 使用 useEffect 处理副作用，当 params 发生变化时设置表单初始值
  useEffect(() => {
    if (params) {
      // 初始化 initialValues 为一个空对象
      const initialValues: ParamValues = {};
      // 遍历 params 数组，将每个参数的默认值设置为初始值
      params.forEach((param) => {
        initialValues[param.param_name] = param.default_value;
      });
      // 使用 form 实例的 setFieldsValue 方法设置表单字段的初始值
      form.setFieldsValue(initialValues); // 设置表单字段的初始值
    }
  }, [params, form]);

  // 如果 params 为 null 或长度小于 1，返回 null
  if (!params || params?.length < 1) {
    return null;
  }

  // 渲染每个参数对应的表单项
  function renderItem(param: SupportModelParams) {
    switch (param.param_type) {
      case 'str':
        return <Input />; // 返回一个输入框组件
      case 'int':
        return <InputNumber />; // 返回一个数字输入框组件
      case 'bool':
        return <Checkbox />; // 返回一个复选框组件
    }
  }

  // 返回组件的 JSX 结构
  return (
    <>
      {params?.map((param: SupportModelParams) => (
        <Form.Item
          key={param.param_name} // 使用参数名作为 key
          label={
            // 设置标签，根据描述长度决定显示参数名或完整描述
            <p className="whitespace-normal overflow-wrap-break-word">
              {param.description?.length > 20 ? param.param_name : param.description}
            </p>
          }
          name={param.param_name} // 设置表单项的名称
          initialValue={param.default_value} // 设置初始值
          valuePropName={param.param_type === 'bool' ? 'checked' : 'value'} // 根据参数类型设置 valuePropName
          tooltip={param.description} // 提示信息为参数的描述
          rules={[{ required: param.required, message: `Please input ${param.description}` }]} // 设置表单项的验证规则
        >
          {renderItem(param)} // 渲染相应的表单项组件
        </Form.Item>
      ))}
    </>
  );
}

// 导出 ModelParams 组件作为默认组件
export default ModelParams;
```
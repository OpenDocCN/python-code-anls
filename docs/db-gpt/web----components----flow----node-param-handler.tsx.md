# `.\DB-GPT-src\web\components\flow\node-param-handler.tsx`

```py
# 导入所需的模块和组件
import { IFlowNode, IFlowNodeParameter } from '@/types/flow';
import { Checkbox, Input, InputNumber, Select, Tooltip } from 'antd';
import React from 'react';
import RequiredIcon from './required-icon';
import NodeHandler from './node-handler';
import { InfoCircleOutlined } from '@ant-design/icons';

# 定义 NodeParamHandlerProps 接口，包含节点、节点参数、标签和索引
interface NodeParamHandlerProps {
  node: IFlowNode;
  data: IFlowNodeParameter;
  label: 'inputs' | 'outputs' | 'parameters';
  index: number; # 数组的索引
}

# 渲染节点参数项
const NodeParamHandler: React.FC<NodeParamHandlerProps> = ({ node, data, label, index }) => {
  # 处理值变化的函数
  function handleChange(value: any) {
    data.value = value;
  }

  # 如果参数类别为 'resource'，则返回 NodeHandler 组件
  if (data.category === 'resource') {
    return <NodeHandler node={node} data={data} type="target" label={label} index={index} />;
  } else if (data.category === 'common') {
    # 设置默认值为数据值或默认值
    let defaultValue = data.value !== null && data.value !== undefined ? data.value : data.default;
    # 根据数据类型名称进行不同的处理
    switch (data.type_name) {
      case 'int':
      case 'float':
        # 如果数据类型为整数或浮点数，显示输入框
        return (
          <div className="p-2 text-sm">
            <p>
              {data.label}:<RequiredIcon optional={data.optional} />
              # 如果有描述信息，显示提示框
              {data.description && (
                <Tooltip title={data.description}>
                  <InfoCircleOutlined className="ml-2 cursor-pointer" />
                </Tooltip>
              )}
            </p>
            # 显示数字输入框
            <InputNumber
              className="w-full"
              defaultValue={defaultValue}
              onChange={(value: number | null) => {
                handleChange(value);
              }}
            />
          </div>
        );
      case 'str':
        # 如果数据类型为字符串，根据是否有选项显示下拉框或输入框
        return (
          <div className="p-2 text-sm">
            <p>
              {data.label}:<RequiredIcon optional={data.optional} />
              # 如果有描述信息，显示提示框
              {data.description && (
                <Tooltip title={data.description}>
                  <InfoCircleOutlined className="ml-2 cursor-pointer" />
                </Tooltip>
              )}
            </p>
            # 如果有选项，显示下拉框，否则显示输入框
            {data.options?.length > 0 ? (
              <Select
                className="w-full nodrag"
                defaultValue={defaultValue}
                options={data.options.map((item: any) => ({ label: item.label, value: item.value }))}
                onChange={handleChange}
              />
            ) : (
              <Input
                className="w-full"
                defaultValue={defaultValue}
                onChange={(e) => {
                  handleChange(e.target.value);
                }}
              />
            )}
          </div>
        );
      case 'bool':
        # 如果数据类型为布尔值，显示复选框
        defaultValue = defaultValue === 'False' ? false : defaultValue;
        defaultValue = defaultValue === 'True' ? true : defaultValue;
        return (
          <div className="p-2 text-sm">
            <p>
              {data.label}:<RequiredIcon optional={data.optional} />
              # 如果有描述信息，显示提示框
              {data.description && (
                <Tooltip title={data.description}>
                  <InfoCircleOutlined className="ml-2 cursor-pointer" />
                </Tooltip>
              )}
              # 显示复选框
              <Checkbox
                className="ml-2"
                defaultChecked={defaultValue}
                onChange={(e) => {
                  handleChange(e.target.checked);
                }}
              />
            </p>
          </div>
        );
    }
  }
};

export default NodeParamHandler;
```
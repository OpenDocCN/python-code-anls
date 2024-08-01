# `.\DB-GPT-src\web\components\app\resource-card.tsx`

```py
import { apiInterceptors, getResource } from '@/client/api';
import { DeleteFilled } from '@ant-design/icons';
import { Button, Card, ConfigProvider, Input, Select, Switch } from 'antd';
import React, { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';

interface IProps {
  resourceTypeOptions: any[]; // 接收资源类型选项的数组
  updateResourcesByIndex: (data: any, index: number) => void; // 更新资源函数，根据索引更新数据
  index: number; // 当前资源在列表中的索引
  resource: any; // 资源对象
}

export default function ResourceCard(props: IProps) {
  const { resourceTypeOptions, updateResourcesByIndex, index, resource: editResource } = props;

  const { t } = useTranslation(); // 获取翻译函数

  // 状态管理：资源类型和资源数据
  const [resourceType, setResourceType] = useState<string>(editResource.type || resourceTypeOptions?.[0].label);
  const [resourceValueOptions, setResourceValueOptions] = useState<any[]>([]);
  const [resource, setResource] = useState<any>({
    name: editResource.name,
    type: editResource.type,
    value: editResource.value,
    is_dynamic: editResource.is_dynamic || false,
  });

  // 异步函数：获取资源
  const fetchResource = async () => {
    const [_, data] = await apiInterceptors(getResource({ type: resourceType }));

    if (data) {
      // 设置资源数值选项
      setResourceValueOptions(
        data?.map((item) => {
          return { label: item, value: item };
        }),
      );
    } else {
      setResourceValueOptions([]); // 数据为空时，清空资源数值选项
    }
  };

  // 处理资源类型变化
  const handleChange = (value: string) => {
    setResourceType(value); // 设置资源类型
  };

  // 更新资源数据
  const updateResource = (value: any, type: string) => {
    const tempResource = resource;

    tempResource[type] = value; // 更新临时资源数据
    setResource(tempResource); // 更新资源状态
    updateResourcesByIndex(tempResource, index); // 调用更新资源函数
  };

  // 处理删除资源操作
  const handleDeleteResource = () => {
    updateResourcesByIndex(null, index); // 通过索引删除资源
  };

  // 当资源类型变化时执行效果
  useEffect(() => {
    fetchResource(); // 获取资源数据
    updateResource(resource.type || resourceType, 'type'); // 更新资源类型
  }, [resourceType]);

  // 当资源数值选项变化时执行效果
  useEffect(() => {
    // 修复 bug：解决应用编辑下资源参数回显不正确的问题
    updateResource(editResource.value || resourceValueOptions[0]?.label, 'value'); // 更新资源数值
    setResource({ ...resource, value: editResource.value || resourceValueOptions[0]?.label }); // 设置资源状态
  }, [resourceValueOptions]);

  // 返回资源卡片组件
  return (
    <Card
      className="mb-3 dark:bg-[#232734] border-gray-200" // 卡片样式类名
      title={`${t('resource')} ${index + 1}`} // 标题，显示资源索引
      extra={
        <DeleteFilled
          className="text-[#ff1b2e] !text-lg" // 删除图标样式类名
          onClick={() => {
            handleDeleteResource(); // 点击删除图标时执行删除资源操作
          }}
        />
      }
    >
      <div className="flex-1">
        <div className="flex items-center  mb-6">
          {/* 资源名称部分 */}
          <div className="font-bold mr-4 w-32 text-center">
            {/* 必填项标志 */}
            <span className="text-[#ff4d4f] font-normal">*</span>&nbsp;{t('resource_name')}:
          </div>
          {/* 输入框组件，用于编辑资源名称 */}
          <Input
            className="w-1/3"
            required
            value={resource.name}
            onInput={(e: React.ChangeEvent<HTMLInputElement>) => {
              updateResource(e.target.value, 'name');
            }}
          />
          {/* 动态资源开关部分 */}
          <div className="flex items-center">
            <div className="font-bold w-32 text-center">{t('resource_dynamic')}</div>
            {/* 开关组件，用于切换资源是否为动态资源 */}
            <Switch
              defaultChecked={editResource.is_dynamic || false}
              style={{ background: resource.is_dynamic ? '#1677ff' : '#ccc' }}
              onChange={(value) => {
                updateResource(value, 'is_dynamic');
              }}
            />
          </div>
        </div>
        {/* 资源类型和值部分 */}
        <div className="flex mb-5  items-center">
          <div className="font-bold mr-4 w-32  text-center">{t('resource_type')}: </div>
          {/* 下拉选择框，用于选择资源类型 */}
          <Select
            className="w-1/3"
            options={resourceTypeOptions}
            value={resource.type || resourceTypeOptions?.[0]}
            onChange={(value) => {
              updateResource(value, 'type');
              handleChange(value);
            }}
          />
          {/* 资源值部分 */}
          <div className="font-bold w-32 text-center">{t('resource_value')}:</div>
          {/* 根据资源值选项是否存在选择下拉框或输入框 */}
          {resourceValueOptions?.length > 0 ? (
            <Select
              value={resource.value}
              className="flex-1"
              options={resourceValueOptions}
              onChange={(value) => {
                updateResource(value, 'value');
              }}
            />
          ) : (
            <Input
              className="flex-1"
              value={resource.value || editResource.value}
              onInput={(e: React.ChangeEvent<HTMLInputElement>) => {
                updateResource(e.target.value, 'value');
              }}
            />
          )}
        </div>
      </div>
    </Card>
  );


注释：
- `<div className="flex-1">`: 使用 Flex 布局，占据剩余空间的部分。
- `<div className="flex items-center  mb-6">`: Flex 布局，垂直居中并与下方元素有一定的外边距。
- `<div className="font-bold mr-4 w-32 text-center">`: 加粗字体，右边距 4 个单位，宽度 32 单位，并水平居中对齐。
- `<span className="text-[#ff4d4f] font-normal">*</span>&nbsp;{t('resource_name')}:`: 显示红色星号标记必填项，后跟资源名称。
- `<Input className="w-1/3" ... />`: 输入框组件，用于编辑资源名称，宽度占父元素的三分之一。
- `<div className="flex items-center">`: Flex 布局，垂直居中。
- `<div className="font-bold w-32 text-center">{t('resource_dynamic')}</div>`: 加粗字体，宽度 32 单位，水平居中显示“动态资源”文本。
- `<Switch defaultChecked={...} ... />`: 开关组件，用于切换资源是否为动态资源，默认状态和样式根据资源的动态属性变化。
- `<div className="font-bold mr-4 w-32  text-center">{t('resource_type')}: </div>`: 加粗字体，右边距 4 个单位，宽度 32 单位，水平居中显示“资源类型”文本。
- `<Select className="w-1/3" ... />`: 下拉选择框组件，用于选择资源类型，宽度占父元素的三分之一。
- `<div className="font-bold w-32 text-center">{t('resource_value')}:</div>`: 加粗字体，宽度 32 单位，水平居中显示“资源值”文本。
- `{resourceValueOptions?.length > 0 ? ... : ...}`: 根据资源值选项的长度决定显示下拉选择框或输入框。
- `<Select ... />`: 下拉选择框组件，用于选择资源值。
- `<Input ... />`: 输入框组件，用于编辑资源值。
}



# 这是一个代码块的结尾，用于结束某个代码段或函数的定义或逻辑结构。
```
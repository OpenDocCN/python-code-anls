# `.\DB-GPT-src\web\components\app\dag-layout.tsx`

```py
import React, { useEffect, useState } from 'react';
import PreviewFlow from '../flow/preview-flow';
import { apiInterceptors, getFlows } from '@/client/api';  // 导入 API 请求函数和流程获取函数
import { IFlow } from '@/types/flow';  // 导入流程类型接口
import { Select } from 'antd';  // 导入 Ant Design 的 Select 组件
import Link from 'next/link';  // 导入 Next.js 的 Link 组件
import { t } from 'i18next';  // 导入国际化的 t 函数

interface IProps {
  onFlowsChange: (data: any) => void;  // 定义回调函数类型，用于流程改变时的回调
  teamContext: any;  // 团队上下文信息
}

export default function DagLayout(props: IProps) {
  const { onFlowsChange, teamContext } = props;  // 解构 props 中的 onFlowsChange 和 teamContext
  const [flows, setFlows] = useState<IFlow[]>();  // 使用 useState 定义流程列表和其更新函数
  const [flowsOptions, setFlowsOptions] = useState<any>();  // 使用 useState 定义流程选项列表和其更新函数
  const [curFlow, setCurFlow] = useState<IFlow>();  // 使用 useState 定义当前流程和其更新函数

  // 异步函数，从后端获取流程数据
  const fetchFlows = async () => {
    const [_, data] = await apiInterceptors(getFlows());  // 使用 API 请求拦截器获取流程数据
    if (data) {
      // 设置流程选项列表为从数据中映射得到的流程名称和值的数组
      setFlowsOptions(data?.items?.map((item: IFlow) => ({ label: item.name, value: item.name })));
      setFlows(data.items);  // 设置流程列表为从数据中获取的流程数组
      onFlowsChange(data?.items[0]);  // 调用外部传入的回调函数，传递第一个流程作为参数
    }
  };

  // 处理流程选择变化的回调函数
  const handleFlowsChange = (value: string) => {
    // 根据选择的流程名称设置当前流程状态
    setCurFlow(flows?.find((item) => value === item.name));
    // 调用外部传入的回调函数，传递与选择名称匹配的流程作为参数
    onFlowsChange(flows?.find((item) => value === item.name));
  };

  // 当组件挂载时，执行一次获取流程数据的操作
  useEffect(() => {
    fetchFlows();
  }, []);

  // 当团队上下文或流程列表发生变化时，更新当前流程状态
  useEffect(() => {
    setCurFlow(flows?.find((item) => teamContext?.name === item.name) || flows?.[0]);
  }, [teamContext, flows]);

  // 返回布局组件的 JSX 结构
  return (
    <div className="w-full h-[300px]">
      <div className="mr-24 mb-4 mt-2">Flows:</div>  // 显示标题 "Flows:"
      <div className="flex items-center mb-6">
        <Select onChange={handleFlowsChange} value={curFlow?.name || flowsOptions?.[0]?.value} className="w-1/4" options={flowsOptions}></Select>  // 使用 Ant Design 的 Select 组件，处理流程选择
        <Link href="/flow/canvas/" className="ml-6">
          {t('edit_new_applications')}  // 显示链接文本，使用国际化函数 t 处理文本内容
        </Link>
        <div className="text-gray-500 ml-16">{curFlow?.description}</div>  // 显示当前流程的描述信息
      </div>
      {curFlow && (  // 如果有当前流程，则显示预览流程组件
        <div className="w-full h-full border-[0.5px] border-dark-gray">
          <PreviewFlow flowData={curFlow?.flow_data} />  // 向预览流程组件传递当前流程的流程数据
        </div>
      )}
    </div>
  );
}
```
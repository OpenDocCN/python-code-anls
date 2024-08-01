# `.\DB-GPT-src\web\components\app\agent-panel.tsx`

```py
import { apiInterceptors, getAppStrategy, getAppStrategyValues, getResource } from '@/client/api';
import { Button, Input, Select } from 'antd';
import React, { useEffect, useMemo, useState } from 'react';
import ResourceCard from './resource-card';
import { useTranslation } from 'react-i18next';

interface IProps {
  resourceTypes: any;
  updateDetailsByAgentKey: (key: string, data: any) => void;
  detail: any;
  editResources?: any;
}

export default function AgentPanel(props: IProps) {
  // 解构 props 对象
  const { resourceTypes, updateDetailsByAgentKey, detail, editResources } = props;
  // 获取国际化翻译函数
  const { t } = useTranslation();

  // 状态管理：资源列表，默认从 editResources 复制一份
  const [resources, setResources] = useState<any>([...(editResources ?? [])]);
  // 状态管理：代理信息，包括从 detail 复制的信息，resources 初始化为空数组
  const [agent, setAgent] = useState<any>({ ...detail, resources: [] });
  // 状态管理：策略选项
  const [strategyOptions, setStrategyOptions] = useState<any>([]);
  // 状态管理：策略数值选项
  const [strategyValueOptions, setStrategyValueOptions] = useState<any>([]);

  // 更新指定索引处的资源数据
  const updateResourcesByIndex = (data: any, index: number) => {
    setResources((resources: any) => {
      const tempResources = [...resources];
      if (!data) {
        // 如果 data 为空，过滤掉该索引处的资源
        return tempResources.filter((_: any, indey) => index !== indey);
      }

      // 否则更新指定索引处的资源数据
      return tempResources.map((item: any, indey) => {
        if (index === indey) {
          return data;
        } else {
          return item;
        }
      });
    });
  };

  // 异步获取策略列表
  const getStrategy = async () => {
    const [_, data] = await apiInterceptors(getAppStrategy());
    if (data) {
      // 设置策略选项
      setStrategyOptions(data?.map((item) => ({ label: item, value: item })));
    }
  };

  // 异步获取特定策略类型的策略数值列表
  const getStrategyValues = async (type: string) => {
    const [_, data] = await apiInterceptors(getAppStrategyValues(type));
    if (data) {
      // 设置策略数值选项，如果数据为空则设置为空数组
      setStrategyValueOptions(data.map((item) => ({ label: item, value: item })) ?? []);
    }
  };

  // 格式化策略数值，将字符串按逗号分隔为数组
  const formatStrategyValue = (value: string) => {
    return !value ? [] : value.split(',');
  };

  // 组件加载时执行，获取策略列表和特定策略类型的策略数值列表
  useEffect(() => {
    getStrategy();
    getStrategyValues(detail.llm_strategy);
  }, []);

  // 监听 resources 变化，更新代理信息
  useEffect(() => {
    updateAgent(resources, 'resources');
  }, [resources]);

  // 更新代理信息，根据 type 类型更新 data 数据
  const updateAgent = (data: any, type: string) => {
    const tempAgent = { ...agent };
    tempAgent[type] = data;

    // 设置代理信息
    setAgent(tempAgent);

    // 调用父组件传入的更新函数，更新代理信息
    updateDetailsByAgentKey(detail.key, tempAgent);
  };

  // 处理添加资源事件，向 resources 中添加新的空资源对象
  const handelAdd = () => {
    setResources([...resources, { name: '', type: '', introduce: '', value: '', is_dynamic: '' }]);
  };

  // 资源类型选项，使用 useMemo 缓存，当 resourceTypes 变化时重新计算
  const resourceTypeOptions = useMemo(() => {
    return resourceTypes?.map((item: string) => {
      return {
        label: item,
        value: item,
      };
    });
  }, [resourceTypes]);

  // 返回 JSX 结构
  return (
    <div>
      <div className="flex items-center mb-6 mt-6">
        {/* 显示提示信息 */}
        <div className="mr-2 w-16 text-center">{t('Prompt')}:</div>
        {/* 输入框组件 */}
        <Input
          required
          className="mr-6 w-1/4"
          value={agent.prompt_template}
          onChange={(e) => {
            updateAgent(e.target.value, 'prompt_template');
          }}
        />
        {/* 显示LLM策略 */}
        <div className="mr-2">{t('LLM_strategy')}:</div>
        {/* 下拉选择框组件 */}
        <Select
          value={agent.llm_strategy}
          options={strategyOptions}
          className="w-1/6 mr-6"
          onChange={(value) => {
            updateAgent(value, 'llm_strategy');
            getStrategyValues(value);
          }}
        />
        {/* 如果有策略值选项，则显示策略值下拉选择框 */}
        {strategyValueOptions && strategyValueOptions.length > 0 && (
          <>
            <div className="mr-2">{t('LLM_strategy_value')}:</div>
            <Select
              value={formatStrategyValue(agent.llm_strategy_value)}
              className="w-1/4"
              mode="multiple"
              options={strategyValueOptions}
              onChange={(value) => {
                if (!value || value?.length === 0) {
                  updateAgent(null, 'llm_strategy_value');
                  return null;
                }

                const curValue = value.reduce((pre: string, cur: string, index: number) => {
                  if (index === 0) {
                    return cur;
                  } else {
                    return `${pre},${cur}`;
                  }
                }, '');

                updateAgent(curValue, 'llm_strategy_value');
              }}
            />
          </>
        )}
      </div>
      {/* 显示可用资源标题 */}
      <div className="mb-3 text-lg font-bold">{t('available_resources')}</div>
      {/* 遍历资源列表，显示资源卡片 */}
      {resources.map((resource: any, index: number) => {
        return (
          <ResourceCard
            resource={resource}
            key={index}
            index={index}
            updateResourcesByIndex={updateResourcesByIndex}
            resourceTypeOptions={resourceTypeOptions}
          />
        );
      })}
      {/* 添加资源按钮 */}
      <Button type="primary" className="mt-2" size="middle" onClick={handelAdd}>
        {t('add_resource')}
      </Button>
    </div>
  );
}


注释：

# 这是一个代码块的结束标志，闭合了一个函数、循环、条件语句或其他代码块的开始标志（比如 `{`）。在这个例子中，它可能是一个函数的结束标志。
```
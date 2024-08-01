# `.\DB-GPT-src\web\components\chat\chat-content\chart-view.tsx`

```py
import { Datum } from '@antv/ava';  // 导入 '@antv/ava' 模块中的 Datum 类型

import { Table, Tabs, TabsProps } from 'antd';  // 导入 antd 库中的 Table、Tabs 和 TabsProps 组件

import React from 'react';  // 导入 React 库

import { AutoChart, BackEndChartType, getChartType } from '@/components/chart/autoChart';  // 导入 '@/components/chart/autoChart' 模块中的 AutoChart、BackEndChartType 和 getChartType 函数

import { CodePreview } from './code-preview';  // 导入当前目录下的 code-preview 模块中的 CodePreview 组件

import { formatSql } from '@/utils';  // 导入 '@/utils' 模块中的 formatSql 函数

function ChartView({ data, type, sql }: { data: Datum[]; type: BackEndChartType; sql: string }) {
  // 根据传入的 data 数据生成表格的列数组
  const columns = data?.[0]
    ? Object.keys(data?.[0])?.map((item) => {
        return {
          title: item,  // 列标题为 item
          dataIndex: item,  // 列数据索引为 item
          key: item,  // 列的 key 值为 item
        };
      })
    : [];  // 若 data 为空数组，则 columns 为空数组

  // ChartItem 对象，用于展示图表
  const ChartItem = {
    key: 'chart',  // 唯一标识符为 'chart'
    label: 'Chart',  // 标签显示名称为 'Chart'
    children: <AutoChart data={data} chartType={getChartType(type)} />,  // 使用 AutoChart 组件展示根据 type 类型和 data 数据生成的图表
  };

  // SqlItem 对象，用于展示 SQL 代码预览
  const SqlItem = {
    key: 'sql',  // 唯一标识符为 'sql'
    label: 'SQL',  // 标签显示名称为 'SQL'
    children: <CodePreview language="sql" code={formatSql(sql)} />,  // 使用 CodePreview 组件展示格式化后的 SQL 代码
  };

  // DataItem 对象，用于展示数据表格
  const DataItem = {
    key: 'data',  // 唯一标识符为 'data'
    label: 'Data',  // 标签显示名称为 'Data'
    children: <Table dataSource={data} columns={columns} scroll={{ x: 'auto' }} />,  // 使用 Table 组件展示数据表格，包括数据源和列信息，并支持横向滚动
  };

  // 根据 type 类型决定 TabItems 数组的内容
  const TabItems: TabsProps['items'] = type === 'response_table' ? [DataItem, SqlItem] : [ChartItem, SqlItem, DataItem];

  // 返回 Tabs 组件，根据 type 类型设置默认激活的标签页
  return <Tabs defaultActiveKey={type === 'response_table' ? 'data' : 'chart'} items={TabItems} size="small" />;
}

export default ChartView;  // 导出 ChartView 组件作为默认导出
```
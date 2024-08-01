# `.\DB-GPT-src\web\components\chat\chat-content\vis-chart.tsx`

```py
import { BackEndChartType } from '@/components/chart';
// 从组件库中导入后端图表类型定义

import ChartView from './chart-view';
// 导入图表视图组件

import { Datum } from '@antv/ava';
// 从 @antv/ava 库中导入 Datum 接口类型

interface Props {
  // 定义组件 Props 接口，包含以下字段
  data: {
    // 数据对象包含以下字段
    data: Datum[];
    // 数据数组，每个元素为 Datum 类型
    describe: string;
    // 描述字段，字符串类型
    title: string;
    // 标题字段，字符串类型
    type: BackEndChartType;
    // 图表类型，使用了从组件库中导入的 BackEndChartType 类型
    sql: string;
    // SQL 查询语句，字符串类型
  };
}

function VisChart({ data }: Props) {
  // 定义 VisChart 组件，接收 Props 对象作为参数
  return <ChartView data={data.data} type={data.type} sql={data.sql} />;
  // 渲染 ChartView 组件，并传入 data.data, data.type, data.sql 作为 props
}

export default VisChart;
// 导出 VisChart 组件作为默认导出
```
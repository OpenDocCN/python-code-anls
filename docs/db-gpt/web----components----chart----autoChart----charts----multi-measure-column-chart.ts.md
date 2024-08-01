# `.\DB-GPT-src\web\components\chart\autoChart\charts\multi-measure-column-chart.ts`

```py
import { hasSubset } from '../advisor/utils';  // 导入名为 hasSubset 的函数，来自上级目录中的 advisor/utils 模块

import type { ChartKnowledge, CustomChart, GetChartConfigProps, Specification } from '../types';  // 导入类型声明，包括 ChartKnowledge、CustomChart、GetChartConfigProps 和 Specification，来自上级目录中的 types 模块
import { findNominalField, findOrdinalField } from './util';  // 导入名为 findNominalField 和 findOrdinalField 的函数，来自当前目录中的 util 模块

const getChartSpec = (data: GetChartConfigProps['data'], dataProps: GetChartConfigProps['dataProps']) => {
  try {
    // @ts-ignore
    const field4Y = dataProps?.filter((field) => hasSubset(field.levelOfMeasurements, ['Interval']));  // 从 dataProps 中筛选出 levelOfMeasurements 包含 'Interval' 的字段数组，存储在 field4Y 中
    const nominalField = findNominalField(dataProps);  // 使用 findNominalField 函数从 dataProps 中查找名义字段，存储在 nominalField 中
    const ordinalField = findOrdinalField(dataProps);  // 使用 findOrdinalField 函数从 dataProps 中查找序数字段，存储在 ordinalField 中
    const field4X = nominalField ?? ordinalField;  // 如果 nominalField 存在，则将其赋给 field4X；否则将 ordinalField 赋给 field4X
    if (!field4X || !field4Y) return null;  // 如果 field4X 或 field4Y 不存在，则返回 null

    const spec: Specification = {
      type: 'view',  // 规范类型为 'view'
      data,  // 规范的数据字段为传入的 data
      children: [],  // 规范的子规范列表为空数组
    };

    field4Y?.forEach((field) => {
      const singleLine: Specification = {
        type: 'interval',  // 子规范类型为 'interval'
        encode: {
          x: field4X.name,  // x 轴编码为 field4X 的名称
          y: field.name,  // y 轴编码为当前 field 的名称
          color: () => field.name,  // 颜色编码为返回当前 field 名称的函数
          series: () => field.name,  // 系列编码为返回当前 field 名称的函数
        },
      };
      spec.children.push(singleLine);  // 将当前子规范 singleLine 添加到 spec 的子规范列表中
    });
    return spec;  // 返回构建好的规范对象
  } catch (err) {
    console.log(err);  // 捕获异常并打印到控制台
    return null;  // 返回 null
  }
};

const ckb: ChartKnowledge = {
  id: 'multi_measure_column_chart',  // 图表知识的唯一 ID 为 'multi_measure_column_chart'
  name: 'multi_measure_column_chart',  // 图表知识的名称为 'multi_measure_column_chart'
  alias: ['multi_measure_column_chart'],  // 别名为 'multi_measure_column_chart'
  family: ['ColumnCharts'],  // 所属家族为 'ColumnCharts'
  def: 'multi_measure_column_chart uses lines with segments to show changes in data in a ordinal dimension',  // 定义描述了多度量柱状图如何使用线段显示序数维度数据变化
  purpose: ['Comparison', 'Distribution'],  // 图表用途为比较和分布
  coord: ['Cartesian2D'],  // 坐标系类型为 'Cartesian2D'
  category: ['Statistic'],  // 分类为 'Statistic'
  shape: ['Lines'],  // 形状为 'Lines'
  dataPres: [
    { minQty: 1, maxQty: '*', fieldConditions: ['Interval'] },  // 数据展示要求：至少包含一个 'Interval' 类型字段
    { minQty: 1, maxQty: 1, fieldConditions: ['Nominal'] },  // 数据展示要求：恰好包含一个名义类型字段
  ],
  channel: ['Color', 'Direction', 'Position'],  // 可用的通道包括颜色、方向和位置
  recRate: 'Recommended',  // 推荐使用
  toSpec: getChartSpec,  // 转换为规范的函数为 getChartSpec
};

/* 订制一个图表需要的所有参数 */
export const multi_measure_column_chart: CustomChart = {
  /* 图表唯一 Id */
  chartType: 'multi_measure_column_chart',  // 图表类型为 'multi_measure_column_chart'
  /* 图表知识 */
  chartKnowledge: ckb as ChartKnowledge,  // 图表知识为先前定义的 ckb 对象，类型为 ChartKnowledge
  /** 图表中文名 */
  chineseName: '折线图',  // 图表的中文名称为 '折线图'
};

export default multi_measure_column_chart;  // 默认导出 multi_measure_column_chart 对象
```
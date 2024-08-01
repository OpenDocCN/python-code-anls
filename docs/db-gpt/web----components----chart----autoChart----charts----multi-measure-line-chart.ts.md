# `.\DB-GPT-src\web\components\chart\autoChart\charts\multi-measure-line-chart.ts`

```py
import { hasSubset } from '../advisor/utils';
import { findNominalField, findOrdinalField, getLineSize, processDateEncode, sortData } from './util';
import type { ChartKnowledge, CustomChart, GetChartConfigProps, Specification } from '../types';
import { Datum } from '@antv/ava';

// 定义常量，表示多指标折线图的类型
const MULTI_MEASURE_LINE_CHART = 'multi_measure_line_chart'

// 定义一个函数，用于生成图表的规格配置
const getChartSpec = (data: GetChartConfigProps['data'], dataProps: GetChartConfigProps['dataProps']) => {
  try {
    // 优先确认 x 轴，如果没有枚举类型字段，取第一个字段为 x 轴
    const field4Nominal = findNominalField(dataProps) ?? findOrdinalField(dataProps) ?? dataProps[0];
    
    // @ts-ignore
    // 筛选出符合条件的 Y 轴字段，即不是 x 轴字段且测量级别是 Interval 类型的字段数组
    const field4Y = dataProps?.filter((field) => field.name !== field4Nominal?.name && hasSubset(field.levelOfMeasurements, ['Interval']));
    
    // 如果 x 轴字段或者 Y 轴字段不存在，则返回 null
    if (!field4Nominal || !field4Y) return null;

    // 创建图表的规格对象
    const spec: Specification = {
      type: 'view',
      // 对数据进行排序处理，并指定 x 轴字段
      data: sortData({ data, chartType: MULTI_MEASURE_LINE_CHART, xField: field4Nominal }),
      children: [],
    };

    // 遍历 Y 轴字段数组，为每个字段创建单独的折线图规格对象，并添加到 spec 的 children 中
    field4Y?.forEach((field) => {
      const singleLine: Specification = {
        type: 'line',
        encode: {
          // 处理日期编码，将 x 轴字段名转为字符串，并设置数据属性
          x: processDateEncode(field4Nominal.name as string, dataProps),
          y: field.name, // 设置折线图的 y 轴字段
          color: () => field.name, // 颜色编码函数，返回字段名
          series: () => field.name, // 系列编码函数，返回字段名
          size: (datum: Datum) => getLineSize(datum, data, { field4X: field4Nominal }), // 设置线条大小的函数
        },
        legend: {
          size: false, // 禁用图例中的大小显示
        },
      };
      spec.children.push(singleLine); // 将单独的折线图规格对象添加到 spec 的 children 中
    });

    return spec; // 返回生成的图表规格对象
  } catch (err) {
    console.log(err); // 捕获并打印错误信息
    return null; // 返回 null 表示处理失败
  }
};

// 定义一个图表知识对象，描述多指标折线图的特征和用途
const ckb: ChartKnowledge = {
  id: MULTI_MEASURE_LINE_CHART,
  name: 'multi_measure_line_chart',
  alias: ['multi_measure_line_chart'],
  family: ['LineCharts'],
  def: 'multi_measure_line_chart uses lines with segments to show changes in data in a ordinal dimension',
  purpose: ['Comparison', 'Distribution'],
  coord: ['Cartesian2D'],
  category: ['Statistic'],
  shape: ['Lines'],
  dataPres: [
    { minQty: 1, maxQty: '*', fieldConditions: ['Interval'] }, // 数据展示要求：至少包含一个 Interval 类型的字段
    { minQty: 1, maxQty: 1, fieldConditions: ['Nominal'] }, // 数据展示要求：恰好包含一个 Nominal 类型的字段
  ],
  channel: ['Color', 'Direction', 'Position'], // 图表使用的通道：颜色、方向、位置
  recRate: 'Recommended', // 推荐使用
  toSpec: getChartSpec, // 将图表配置函数指定为转换规格的方法
};

/* 订制一个图表需要的所有参数 */
export const multi_measure_line_chart: CustomChart = {
  /* 图表唯一 Id */
  chartType: 'multi_measure_line_chart',
  /* 图表知识 */
  chartKnowledge: ckb as ChartKnowledge,
  /** 图表中文名 */
  chineseName: '折线图',
};

export default multi_measure_line_chart;
```
# `.\DB-GPT-src\web\components\chart\autoChart\charts\multi-line-chart.ts`

```py
import { hasSubset } from '../advisor/utils';
import { findOrdinalField, processDateEncode, findNominalField, getLineSize, sortData } from './util';
import type { ChartKnowledge, CustomChart, GetChartConfigProps, Specification } from '../types';
import type { Datum } from '@antv/ava';

const MULTI_LINE_CHART = 'multi_line_chart';

// 定义一个函数 getChartSpec，用于生成图表的配置规范
const getChartSpec = (data: GetChartConfigProps['data'], dataProps: GetChartConfigProps['dataProps']) => {
  // 查找数据中的序数字段和名义字段
  const ordinalField = findOrdinalField(dataProps);
  const nominalField = findNominalField(dataProps);

  // 选择用于 x 轴的字段，优先级为时间类型 > 序数类型 > 名义类型 > 第一个字段
  const field4X = ordinalField ?? nominalField ?? dataProps[0];

  // 过滤掉用于 x 轴后剩余的字段
  const remainFields = dataProps.filter((field) => field.name !== field4X?.name);

  // 从剩余字段中筛选出用于 y 轴的字段，首选度量水平为 Interval 的字段，否则选择第一个字段
  const field4Y = remainFields.filter((field) =>
    field.levelOfMeasurements && hasSubset(field.levelOfMeasurements, ['Interval'])
  ) ?? [remainFields[0]];

  // 从剩余字段中找出用于图例的名义字段
  const field4Nominal = remainFields
    .filter(field => !field4Y.find(y => y.name === field.name))
    .find(field => field.levelOfMeasurements && hasSubset(field.levelOfMeasurements, ['Nominal']));

  // 如果未能正确选取 x 或 y 轴字段，则返回空
  if (!field4X || !field4Y) return null;

  // 构建图表的规范对象
  const spec: Specification = {
    type: 'view',
    autoFit: true,
    // 对数据进行排序，并指定 x 轴字段
    data: sortData({ data, chartType: MULTI_LINE_CHART, xField: field4X }),
    children: [], // 存放每条线的配置
  };

  // 针对每个 y 轴字段生成单条线的配置
  field4Y.forEach((field) => {
    const singleLine: Specification = {
      type: 'line',
      encode: {
        // 设置 x 轴的编码规则
        x: processDateEncode(field4X.name as string, dataProps),
        // 设置 y 轴的字段名
        y: field.name,
        // 设置线条的大小，根据数据和指定的字段进行计算
        size: (datum: Datum) => getLineSize(datum, data, { field4Split: field4Nominal, field4X }),
      },
      legend: {
        size: false, // 禁用图例中的大小
      },
    };

    // 如果存在名义字段，设置颜色编码
    if (field4Nominal) {
      singleLine.encode.color = field4Nominal.name;
    }

    // 将单条线的配置添加到图表规范的 children 数组中
    spec.children.push(singleLine);
  });

  // 返回构建好的图表规范对象
  return spec;
};

// 定义图表知识对象 ckb，包含关于折线图的各种元数据信息
const ckb: ChartKnowledge = {
  id: MULTI_LINE_CHART,
  name: 'multi_line_chart',
  alias: ['multi_line_chart'],
  family: ['LineCharts'],
  def: 'multi_line_chart uses lines with segments to show changes in data in a ordinal dimension',
  purpose: ['Comparison', 'Trend'],
  coord: ['Cartesian2D'],
  category: ['Statistic'],
  shape: ['Lines'],
  dataPres: [
    { minQty: 1, maxQty: 1, fieldConditions: ['Time', 'Ordinal'] },
    { minQty: 1, maxQty: '*', fieldConditions: ['Interval'] },
    { minQty: 0, maxQty: 1, fieldConditions: ['Nominal'] },
  ],
  channel: ['Color', 'Direction', 'Position'],
  recRate: 'Recommended',
  toSpec: getChartSpec, // 指定生成图表规范的函数
};

// 定义一个自定义图表对象 multi_line_chart，包含有关折线图的配置和中文名
export const multi_line_chart: CustomChart = {
  chartType: 'multi_line_chart', // 图表类型
  chartKnowledge: ckb as ChartKnowledge, // 图表知识
  chineseName: '折线图', // 图表中文名
};

// 导出自定义图表对象
export default multi_line_chart;
```
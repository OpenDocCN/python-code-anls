# `.\DB-GPT-src\web\components\chart\autoChart\charts\util.ts`

```py
/**
 * 导入特定类型 Datum 和 FieldInfo，来自 "@antv/ava" 模块
 * 导入两个函数 hasSubset 和 intersects，来自 "../advisor/utils" 模块
 * 导入 lodash 中的 cloneDeep 和 uniq 函数
 */
import type { Datum, FieldInfo } from "@antv/ava";
import { hasSubset, intersects } from "../advisor/utils";
import { cloneDeep, uniq } from "lodash";

/**
 * 处理日期列，将其转换为 Date 对象。
 * @param field - 要处理的字段名
 * @param dataProps - 包含字段信息的数组
 * @returns 返回一个处理函数，用于将字段值转换为 Date 对象，或返回原始字段名（如果字段不是日期类型）
 */
export function processDateEncode(field: string, dataProps: FieldInfo[]) {
  // 查找指定字段名的字段信息
  const dp = dataProps.find((dataProp) => dataProp.name === field);

  // 如果字段为日期类型，则返回一个处理函数，将字段值转换为 Date 对象
  if (dp?.recommendation === 'date') {
    return (d: any) => new Date(d[field]);
  }
  // 如果字段不是日期类型，则直接返回原始字段名
  return field;
}

/**
 * 查找包含序数级别的字段。
 * @param fields - 包含字段信息的数组
 * @returns 第一个匹配到的包含 'Time' 或 'Ordinal' 级别的字段，或 undefined 如果找不到匹配字段
 */
export function findOrdinalField(fields: FieldInfo[]) {
  return fields.find((field) => field.levelOfMeasurements && intersects(field.levelOfMeasurements, ['Time', 'Ordinal']));
}

/**
 * 查找包含名义级别的字段。
 * @param fields - 包含字段信息的数组
 * @returns 第一个匹配到的包含 'Nominal' 级别的字段，或 undefined 如果找不到匹配字段
 */
export function findNominalField(fields: FieldInfo[]) {
  return fields.find((field) => field.levelOfMeasurements && hasSubset(field.levelOfMeasurements, ['Nominal']));
}

/**
 * 判断 x 轴上的数据是否唯一（用于判断折线图是否只有一个点）。
 * @param xField - x 轴字段名
 * @param data - 数据数组
 * @returns 如果 x 轴上的数据不唯一返回 true，否则返回 false
 */
export const isUniqueXValue = ({ data, xField }: { xField: string; data: Datum[] }): boolean => {
  // 获取数据中 xField 列的唯一值数组
  const uniqXValues = uniq(data.map((datum) => datum[xField]));
  // 判断唯一值数组的长度是否小于等于 1
  return uniqXValues.length <= 1;
};

/**
 * 获取线条宽度。
 * @param datum - 单个数据项
 * @param allData - 所有数据数组
 * @param fields - 包含字段信息的对象
 * @returns 如果需要特殊设置线宽返回 5，否则返回 undefined
 */
export const getLineSize = (
  datum: Datum,
  allData: Datum[],
  fields: {
    field4Split?: FieldInfo;
    field4X?: FieldInfo;
  },
) => {
  const { field4Split, field4X } = fields;
  // 如果分割字段和 x 轴字段都存在
  if (field4Split?.name && field4X?.name) {
    const seriesValue = datum[field4Split.name];
    // 过滤出分割字段值相同的数据
    const splitData = allData.filter((item) => field4Split.name && item[field4Split.name] === seriesValue);
    // 检查分割数据中 x 轴的唯一性
    return isUniqueXValue({ data: splitData, xField: field4X.name }) ? 5 : undefined;
  }
  // 如果只有 x 轴字段，并且 x 轴数据唯一
  return field4X?.name && isUniqueXValue({ data: allData, xField: field4X.name }) ? 5 : undefined;
};

/**
 * 对数据进行排序处理。
 * @param data - 要排序的数据数组
 * @param chartType - 图表类型
 * @param xField - x 轴字段信息
 * @returns 排序后的数据数组
 */
export const sortData = ({ data, chartType, xField }: {
  data: Datum[];
  xField?: FieldInfo;
  chartType: string;
}) => {
  // 克隆数据，防止原始数据修改
  const sortedData = cloneDeep(data);
  try {
    // 如果图表类型包含 'line' 并且 x 轴字段存在且推荐为日期类型
    if (chartType.includes('line') && xField?.name && xField.recommendation === 'date') {
      // 按照日期从小到大排序数据
      sortedData.sort((datum1, datum2) => new Date(datum1[xField.name as string]).getTime() - new Date(datum2[xField.name as string]).getTime());
      return sortedData;
    }
    // 如果图表类型包含 'line' 并且 x 轴字段存在且推荐为浮点数或整数类型
    if (chartType.includes('line') && xField?.name && ['float', 'integer'].includes(xField.recommendation)) {
      // 按照数值大小排序数据
      sortedData.sort((datum1, datum2) => (datum1[xField.name as string] as number) - (datum2[xField.name as string] as number));
      return sortedData;
    }
  } catch (err) {
    // 捕获并打印排序过程中的错误
    console.error(err);
  }
  // 返回排序后的数据数组
  return sortedData;
};

/**
 * 数据空值处理：将后端返回的空数据 '-' 修改为 null。
 * @param data - 要处理的数据数组
 * @param emptyValue - 空数据占位符，默认为 '-'
 * @returns 处理后的数据数组，空数据修改为 null
 */
export const processNilData = (data: Datum[], emptyValue = '-') => data.map((datum) => {
  const processedDatum: Record<string, string | number | null> = {};
  // 遍历数据对象的每个属性
  Object.keys(datum).forEach((key) => {
    // 将值为 emptyValue 的属性值修改为 null
    processedDatum[key] = datum[key] === emptyValue ? null : datum[key];
  });
  // 返回处理后的数据对象
  return processedDatum;
});
```
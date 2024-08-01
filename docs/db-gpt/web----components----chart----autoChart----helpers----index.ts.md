# `.\DB-GPT-src\web\components\chart\autoChart\helpers\index.ts`

```py
# 导入所需的 ChartId 和 CustomChartsType 类型定义
import { ChartId } from '@antv/ava';
import { CustomChartsType } from '../charts';

# 定义后端图表类型，包括线图、柱状图、饼图、散点图、面积图、热力图和表格
export type BackEndChartType =
  | 'response_line_chart'
  | 'response_bar_chart'
  | 'response_pie_chart'
  | 'response_scatter_chart'
  | 'response_area_chart'
  | 'response_heatmap_chart'
  | 'response_table';

# 定义图表类型为 ChartId 或 CustomChartsType 类型
type ChartType = ChartId | CustomChartsType;

# 根据后端图表类型返回对应的前端图表类型数组
export const getChartType = (backendChartType: BackEndChartType): ChartType[] => {
  # 如果后端图表类型为 'response_line_chart'，返回线图的前端类型数组
  if (backendChartType === 'response_line_chart') {
    return ['multi_line_chart', 'multi_measure_line_chart'];
  }
  # 如果后端图表类型为 'response_bar_chart'，返回柱状图的前端类型数组
  if (backendChartType === 'response_bar_chart') {
    return ['multi_measure_column_chart'];
  }
  # 如果后端图表类型为 'response_pie_chart'，返回饼图的前端类型数组
  if (backendChartType === 'response_pie_chart') {
    return ['pie_chart'];
  }
  # 如果后端图表类型为 'response_scatter_chart'，返回散点图的前端类型数组
  if (backendChartType === 'response_scatter_chart') {
    return ['scatter_plot'];
  }
  # 如果后端图表类型为 'response_area_chart'，返回面积图的前端类型数组
  if (backendChartType === 'response_area_chart') {
    return ['area_chart'];
  }
  # 如果后端图表类型为 'response_heatmap_chart'，返回热力图的前端类型数组
  if (backendChartType === 'response_heatmap_chart') {
    return ['heatmap'];
  }
  # 默认情况下，返回空数组
  return [];
};
```
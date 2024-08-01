# `.\DB-GPT-src\web\components\chart\autoChart\charts\index.ts`

```py
# 导入多行图表组件
import multi_line_chart from './multi-line-chart';
# 导入多度量柱状图组件
import multi_measure_column_chart from './multi-measure-column-chart';
# 导入多度量折线图组件
import multi_measure_line_chart from './multi-measure-line-chart';
# 导入自定义图表类型
import type { CustomChart } from '../types';

# 定义自定义图表数组，包含多行图表、多度量柱状图和多度量折线图
export const customCharts: CustomChart[] = [multi_line_chart, multi_measure_column_chart, multi_measure_line_chart];

# 定义自定义图表类型，包括多行图表、多度量柱状图和多度量折线图
export type CustomChartsType = 'multi_line_chart' | 'multi_measure_column_chart' | 'multi_measure_line_chart';
```
# `.\DB-GPT-src\web\components\chat\chat-content\vis-dashboard.tsx`

```py
import { AutoChart, BackEndChartType, getChartType } from '@/components/chart';
import { Datum } from '@antv/ava';
import { useMemo } from 'react';

interface Props {
  data: {
    data: {
      data: Datum[];                              // 定义一个数组，包含了 Datum 类型的数据
      describe: string;                           // 描述字段，字符串类型
      title: string;                              // 标题字段，字符串类型
      type: BackEndChartType;                     // 后端图表类型
      sql: string;                                // SQL 查询字符串
    }[];
    title: string | null;                         // 标题，可以为空字符串
    display_strategy: string;                     // 显示策略，字符串类型
    chart_count: number;                          // 图表数量，数字类型
  };
}

const chartLayout = [[2], [1, 2], [1, 3], [2, 1, 2], [2, 1, 3], [3, 1, 3], [3, 2, 3]];  // 定义了几种图表布局方式的数组

function VisDashboard({ data }: Props) {
  const charts = useMemo(() => {                   // 使用 useMemo 进行性能优化，避免不必要的重新计算
    if (data.chart_count > 1) {                    // 如果图表数量大于 1
      const layout = chartLayout[data.chart_count - 2];  // 根据图表数量选择合适的布局方式
      let prevIndex = 0;                          // 初始化前一个索引为 0
      return layout.map((item) => {                // 遍历布局数组，生成对应的图表数据
        const items = data.data.slice(prevIndex, prevIndex + item);  // 根据布局要求切分数据
        prevIndex = item;                          // 更新前一个索引
        return items;                              // 返回切分后的图表数据
      });
    }
    return [data.data];                            // 如果只有一个图表，则直接返回数据
  }, [data.data, data.chart_count]);               // 依赖项为 data.data 和 data.chart_count，当这些变化时重新计算

  return (
    <div className="flex flex-col gap-3">          // 使用 flex 布局，垂直排列，间距为 3
      {charts.map((row, index) => (                // 遍历每一行的图表数据
        <div key={`row-${index}`} className="flex gap-3">  // 每一行的容器，水平排列，间距为 3
          {row.map((chart, subIndex) => (          // 遍历每个图表数据
            <div
              key={`chart-${subIndex}`}
              className="flex flex-1 flex-col justify-between p-4 rounded border border-gray-200 dark:border-gray-500 whitespace-normal"
            >                                       // 每个图表的容器，包含样式类名和自动换行
              <div>
                {chart.title && <div className="mb-2 text-lg">{chart.title}</div>}  // 如果有标题，则显示标题
                {chart.describe && <div className="mb-4 text-sm text-gray-500">{chart.describe}</div>}  // 如果有描述，则显示描述
              </div>
              <AutoChart data={chart.data} chartType={getChartType(chart.type)} />  // 使用 AutoChart 组件显示图表，传入数据和图表类型
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

export default VisDashboard;                      // 导出 VisDashboard 组件
```
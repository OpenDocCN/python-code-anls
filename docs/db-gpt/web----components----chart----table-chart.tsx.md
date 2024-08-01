# `.\DB-GPT-src\web\components\chart\table-chart.tsx`

```py
import { ChartData } from '@/types/chat'; 
// 从指定路径导入 ChartData 类型

import { Table } from '@mui/joy'; 
// 从 '@mui/joy' 库中导入 Table 组件

import { groupBy } from 'lodash'; 
// 从 'lodash' 库中导入 groupBy 函数

export default function TableChart({ chart }: { key: string; chart: ChartData }) { 
// 导出默认的 TableChart 组件，接受一个对象参数，包含 key 和 chart 属性，chart 属性的类型为 ChartData

  const data = groupBy(chart.values, 'type'); 
  // 使用 groupBy 函数将 chart.values 按照 'type' 属性进行分组，结果存储在 data 中

  return (
    <div className="flex-1 min-w-0 p-4 bg-white dark:bg-theme-dark-container rounded">
      // 返回一个 div 元素，设置样式类名为 'flex-1 min-w-0 p-4 bg-white dark:bg-theme-dark-container rounded'
      <div className="h-full">
        // 内部 div 元素设置样式类名为 'h-full'
        <div className="mb-2">{chart.chart_name}</div>
        // 内部 div 元素显示 chart.chart_name 属性的值
        <div className="opacity-80 text-sm mb-2">{chart.chart_desc}</div>
        // 内部 div 元素显示 chart.chart_desc 属性的值
        <div className="flex-1">
          // 内部 div 元素设置样式类名为 'flex-1'
          <Table aria-label="basic table" stripe="odd" hoverRow borderAxis="bothBetween">
          // 使用 Table 组件，设置属性 aria-label 为 'basic table'，stripe 为 'odd'，hoverRow 为 true，borderAxis 为 'bothBetween'
            <thead>
              // 表头部分
              <tr>
                // 表头行
                {Object.keys(data).map((key) => (
                  // 遍历 data 对象的键，生成对应的表头单元格
                  <th key={key}>{key}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              // 表格主体部分
              {Object.values(data)?.[0]?.map((value, i) => (
                // 遍历第一个分组的值，生成表格行
                <tr key={i}>
                  // 表格行
                  {Object.keys(data)?.map((k) => (
                    // 遍历 data 对象的键，生成对应的表格单元格
                    <td key={k}>{data?.[k]?.[i].value || ''}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </Table>
        </div>
      </div>
    </div>
  );
}
```
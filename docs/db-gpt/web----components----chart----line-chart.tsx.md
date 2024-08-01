# `.\DB-GPT-src\web\components\chart\line-chart.tsx`

```py
import { ChartData } from '@/types/chat';  // 导入 ChartData 类型定义，来自 '@/types/chat'
import { Chart } from '@berryv/g2-react';  // 导入 Chart 组件，来自 '@berryv/g2-react'
import { useContext } from 'react';  // 导入 useContext 钩子，用于从 React 上下文中获取数据
import { ChatContext } from '@/app/chat-context';  // 导入 ChatContext，来自 '@/app/chat-context'

export default function LineChart({ chart }: { chart: ChartData }) {
  const { mode } = useContext(ChatContext);  // 使用 useContext 获取 ChatContext 中的 mode 数据

  return (
    <div className="flex-1 min-w-0 p-4 bg-white dark:bg-theme-dark-container rounded">
      <div className="h-full">
        <div className="mb-2">{chart.chart_name}</div>  {/* 显示图表名称 */}
        <div className="opacity-80 text-sm mb-2">{chart.chart_desc}</div>  {/* 显示图表描述，文字大小为小号 */}
        <div className="h-[300px]">
          <Chart
            style={{ height: '100%' }}  // 设置图表容器的高度为100%
            options={{
              autoFit: true,  // 启用自适应布局
              theme: mode,  // 使用从 ChatContext 中获取的主题模式
              type: 'view',  // 设置图表类型为视图
              data: chart.values,  // 使用传入的图表数据作为图表的数据源
              children: [  // 子元素配置，包括线图和面积图
                {
                  type: 'line',  // 子图表类型为线图
                  encode: {
                    x: 'name',  // x 轴绑定数据字段为 'name'
                    y: 'value',  // y 轴绑定数据字段为 'value'
                    color: 'type',  // 颜色映射字段为 'type'
                    shape: 'smooth',  // 使用平滑曲线形状
                  },
                },
                {
                  type: 'area',  // 子图表类型为面积图
                  encode: {
                    x: 'name',  // x 轴绑定数据字段为 'name'
                    y: 'value',  // y 轴绑定数据字段为 'value'
                    color: 'type',  // 颜色映射字段为 'type'
                    shape: 'smooth',  // 使用平滑曲线形状
                  },
                  legend: false,  // 隐藏图例
                  style: {
                    fillOpacity: 0.15,  // 设置填充透明度为 0.15
                  },
                },
              ],
              axis: {
                x: {
                  labelAutoRotate: false,  // 禁用 x 轴标签自动旋转
                },
              },
            }}
          />
        </div>
      </div>
    </div>
  );
}
```
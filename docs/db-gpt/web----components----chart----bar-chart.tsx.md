# `.\DB-GPT-src\web\components\chart\bar-chart.tsx`

```py
import { ChatContext } from '@/app/chat-context';  // 导入聊天上下文组件
import { ChartData } from '@/types/chat';  // 导入图表数据类型定义
import { Chart } from '@berryv/g2-react';  // 导入图表组件
import { useContext } from 'react';  // 导入 React 的 useContext 钩子

export default function BarChart({ chart }: { key: string; chart: ChartData }) {
  const { mode } = useContext(ChatContext);  // 使用 useContext 获取聊天上下文中的模式信息（暗/亮模式）

  return (
    <div className="flex-1 min-w-0 p-4 bg-white dark:bg-theme-dark-container rounded">
      <div className="h-full">
        <div className="mb-2">{chart.chart_name}</div>  {/* 显示图表名称 */}
        <div className="opacity-80 text-sm mb-2">{chart.chart_desc}</div>  {/* 显示图表描述 */}
        <div className="h-[300px]">
          <Chart
            style={{ height: '100%' }}  // 设置图表组件的样式，使其高度铺满父容器
            options={{
              autoFit: true,  // 自适应宽度
              theme: mode,  // 根据上下文中的模式设置图表主题（暗/亮模式）
              type: 'interval',  // 图表类型为柱状图
              data: chart.values,  // 图表显示的数据
              encode: { x: 'name', y: 'value', color: 'type' },  // 指定数据的编码方式，对应 x 轴、y 轴和颜色
              axis: {
                x: {
                  labelAutoRotate: false,  // 禁止 x 轴标签自动旋转
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
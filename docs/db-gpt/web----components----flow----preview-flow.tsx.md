# `.\DB-GPT-src\web\components\flow\preview-flow.tsx`

```py
import { IFlowData } from '@/types/flow';
import React from 'react';
import ReactFlow, { Background } from 'reactflow';
import ButtonEdge from './button-edge';
import { mapUnderlineToHump } from '@/utils/flow';
import 'reactflow/dist/style.css';

// 定义 PreviewFlow 组件，接收 flowData 和可选的最小缩放比例 minZoom
const PreviewFlow: React.FC<{ flowData: IFlowData; minZoom?: number }> = ({ flowData, minZoom }) => {
  // 将 flowData 中的下划线转换为驼峰命名
  const data = mapUnderlineToHump(flowData);

  // 渲染 ReactFlow 组件，传入转换后的节点和边数据，及自定义的边类型 ButtonEdge
  return (
    <ReactFlow nodes={data.nodes} edges={data.edges} edgeTypes={{ buttonedge: ButtonEdge }} fitView minZoom={minZoom || 0.1}>
      {/* 设置流程图的背景色和节点之间的间隔 */}
      <Background color="#aaa" gap={16} />
    </ReactFlow>
  );
};

export default PreviewFlow;
```
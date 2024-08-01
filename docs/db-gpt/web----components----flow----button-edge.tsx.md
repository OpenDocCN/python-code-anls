# `.\DB-GPT-src\web\components\flow\button-edge.tsx`

```py
import React from 'react';
import { getBezierPath, EdgeProps, BaseEdge, useReactFlow } from 'reactflow';

const ButtonEdge: React.FC<EdgeProps> = ({ id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, style = {}, data, markerEnd }) => {
    // 根据提供的位置信息计算贝塞尔曲线路径、曲线中心点的坐标
    const [edgePath, edgeCenterX, edgeCenterY] = getBezierPath({
        sourceX,
        sourceY,
        sourcePosition,
        targetX,
        targetY,
        targetPosition,
    });
    
    // 获取当前 React Flow 实例
    const reactFlow = useReactFlow();

    // 处理边线点击事件的回调函数
    function onEdgeClick(event: React.MouseEvent, id: string) {
        // 阻止事件冒泡，确保不影响其他元素的事件处理
        event.stopPropagation();
        // 过滤掉当前点击的边线，更新 React Flow 中的边线列表
        reactFlow.setEdges(reactFlow.getEdges().filter((edge) => edge.id !== id));
    }

    return (
        <>
            {/* 基础边线组件，用于渲染连接两个节点之间的边线 */}
            <BaseEdge id={id} style={style} path={edgePath} markerEnd={markerEnd} />
            
            {/* 外部容器，包含一个用于点击删除边线的按钮 */}
            <foreignObject
                width={40}
                height={40}
                x={edgeCenterX - 40 / 2}
                y={edgeCenterY - 40 / 2}
                className="bg-transparent w-10 h-10 relative"
                requiredExtensions="http://www.w3.org/1999/xhtml"
            >
                {/* 删除按钮，点击时调用 onEdgeClick 处理删除边线的逻辑 */}
                <button
                    className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-5 h-5 rounded-full bg-stone-400 dark:bg-zinc-700 cursor-pointer text-sm"
                    onClick={(event) => onEdgeClick(event, id)}
                >
                    ×
                </button>
            </foreignObject>
        </>
    );
};

export default ButtonEdge;
```
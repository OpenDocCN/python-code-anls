# `.\DB-GPT-src\web\components\flow\static-nodes.tsx`

```py
import { IFlowNode } from '@/types/flow';
import { Avatar, Empty, List } from 'antd';
import React, { DragEvent } from 'react';
import { useTranslation } from 'react-i18next';

// 定义 Props 类型，包含节点数组
const StaticNodes: React.FC<{ nodes: IFlowNode[] }> = ({ nodes }) => {
  // 使用 i18n 国际化钩子
  const { t } = useTranslation();

  // 拖拽开始事件处理函数，传递节点数据为 JSON 字符串
  function onDragStart(event: DragEvent, node: IFlowNode) {
    event.dataTransfer.setData('application/reactflow', JSON.stringify(node));
    event.dataTransfer.effectAllowed = 'move';
  }

  // 如果节点数组不为空，则渲染节点列表
  if (nodes?.length > 0) {
    return (
      // 使用 Ant Design List 组件展示节点列表
      <List
        className="overflow-hidden overflow-y-auto w-full"
        itemLayout="horizontal"
        dataSource={nodes}
        // 遍历每个节点，渲染列表项
        renderItem={(node) => (
          <List.Item
            className="cursor-move hover:bg-[#F1F5F9] dark:hover:bg-theme-dark p-0 py-2"
            draggable
            // 当拖拽开始时调用 onDragStart 处理函数
            onDragStart={(event) => onDragStart(event, node)}
          >
            {/* 列表项内容 */}
            <List.Item.Meta
              className="flex items-center justify-center"
              // 节点图标
              avatar={<Avatar src={'/icons/node/vis.png'} size={'large'} />}
              // 节点标题
              title={<p className="line-clamp-1 font-medium">{node.label}</p>}
              // 节点描述
              description={<p className="line-clamp-2">{node.description}</p>}
            />
          </List.Item>
        )}
      />
    );
  } else {
    // 如果节点数组为空，则显示空状态组件，并使用国际化描述文字
    return <Empty className="px-2" description={t('no_node')} />;
  }
};

// 导出静态节点组件
export default StaticNodes;
```
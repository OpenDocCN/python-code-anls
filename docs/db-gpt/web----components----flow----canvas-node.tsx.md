# `.\DB-GPT-src\web\components\flow\canvas-node.tsx`

```py
// 导入模块，包括接口和组件
import { IFlowNode } from '@/types/flow';
import Image from 'next/image';
import NodeParamHandler from './node-param-handler';
import classNames from 'classnames';
import { useState } from 'react';
import NodeHandler from './node-handler';
import { Popover, Tooltip } from 'antd';
import { CopyOutlined, DeleteOutlined, InfoCircleOutlined } from '@ant-design/icons';
import { useReactFlow } from 'reactflow';
import IconWrapper from '../common/icon-wrapper';
import { getUniqueNodeId } from '@/utils/flow';
import { cloneDeep } from 'lodash';

// 定义属性类型
type CanvasNodeProps = {
  data: IFlowNode;
};

// 图标路径前缀常量
const ICON_PATH_PREFIX = '/icons/node/';

// 类型标签组件，用于显示节点标签
function TypeLabel({ label }: { label: string }) {
  return <div className="w-full h-8 bg-stone-100 dark:bg-zinc-700 px-2 flex items-center justify-center">{label}</div>;
}

// CanvasNode 组件定义
const CanvasNode: React.FC<CanvasNodeProps> = ({ data }) => {
  // 解构传入的数据对象
  const node = data;
  const { inputs, outputs, parameters, flow_type: flowType } = node;
  // 设置状态钩子
  const [isHovered, setIsHovered] = useState(false);
  // 获取 reactflow 实例
  const reactFlow = useReactFlow();

  // 鼠标移入事件处理函数
  function onHover() {
    setIsHovered(true);
  }

  // 鼠标移出事件处理函数
  function onLeave() {
    setIsHovered(false);
  }

  // 复制节点操作
  function copyNode(e: React.MouseEvent<HTMLButtonElement, MouseEvent>) {
    e.preventDefault();
    e.stopPropagation();
    // 获取所有节点
    const nodes = reactFlow.getNodes();
    // 查找原始节点
    const originalNode = nodes.find((item) => item.id === node.id);
    if (originalNode) {
      // 获取唯一节点 ID
      const newNodeId = getUniqueNodeId(originalNode as IFlowNode, nodes);
      // 深拷贝原始节点
      const cloneNode = cloneDeep(originalNode);
      // 创建副本节点对象
      const duplicatedNode = {
        ...cloneNode,
        id: newNodeId,
        position: {
          x: cloneNode.position.x + 400,
          y: cloneNode.position.y,
        },
        positionAbsolute: {
          x: cloneNode.positionAbsolute!.x + 400,
          y: cloneNode.positionAbsolute!.y,
        },
        data: {
          ...cloneNode.data,
          id: newNodeId,
        },
        selected: false,
      };
      // 更新节点列表
      reactFlow.setNodes((nodes) => [...nodes, duplicatedNode]);
    }
  }

  // 删除节点操作
  function deleteNode(e: React.MouseEvent<HTMLButtonElement, MouseEvent>) {
    e.preventDefault();
    e.stopPropagation();
    // 过滤删除节点及其相关边
    reactFlow.setNodes((nodes) => nodes.filter((item) => item.id !== node.id));
    reactFlow.setEdges((edges) => edges.filter((edge) => edge.source !== node.id && edge.target !== node.id));
  }

  // 渲染输出端口
  function renderOutput(data: IFlowNode) {
    // 如果节点类型为操作符且有输出端口
    if (flowType === 'operator' && outputs?.length > 0) {
      return (
        <>
          <TypeLabel label="Outputs" />
          {/* 遍历输出端口并渲染节点处理器 */}
          {(outputs || []).map((output, index) => (
            <NodeHandler key={`${data.id}_input_${index}`} node={data} data={output} type="source" label="outputs" index={index} />
          ))}
        </>
      );
    }
  }
    } else if (flowType === 'resource') {
      // 如果流程类型是 'resource'，则显示默认的输出节点
      return (
        <>
          <TypeLabel label="Outputs" />
          {/* 创建一个资源节点，显示输出 */}
          <NodeHandler key={`${data.id}_input_0`} node={data} data={data} type="source" label="outputs" index={0} />
        </>
      );
    }
  }

  // 默认情况下，显示一个弹出框，包含图标和标签信息
  return (
    <Popover
      placement="rightTop"
      trigger={['hover']}
      content={
        <>
          {/* 复制节点的图标 */}
          <IconWrapper className="hover:text-blue-500">
            <CopyOutlined className="h-full text-lg cursor-pointer" onClick={copyNode} />
          </IconWrapper>
          {/* 删除节点的图标 */}
          <IconWrapper className="mt-2 hover:text-red-500">
            <DeleteOutlined className="h-full text-lg cursor-pointer" onClick={deleteNode} />
          </IconWrapper>
          {/* 节点信息的图标 */}
          <IconWrapper className="mt-2">
            <Tooltip title={<><p className="font-bold">{node.label}</p><p>{node.description}</p></>} placement="right">
              <InfoCircleOutlined className="h-full text-lg cursor-pointer" />
            </Tooltip>
          </IconWrapper>
        </>
      }
    >
      {/* 节点的主体部分，包括图标、标签和各种状态的样式 */}
      <div
        className={classNames('w-72 h-auto rounded-xl shadow-md p-0 border bg-white dark:bg-zinc-800 cursor-grab', {
          'border-blue-500': node.selected || isHovered,
          'border-stone-400 dark:border-white': !node.selected && !isHovered,
          'border-dashed': flowType !== 'operator',
          'border-red-600': node.invalid,
        })}
        onMouseEnter={onHover}
        onMouseLeave={onLeave}
      >
        {/* 图标和标签部分 */}
        <div className="flex flex-row items-center p-2">
          <Image src={'/icons/node/vis.png'} width={24} height={24} alt="" />
          {/* 节点的标签 */}
          <p className="ml-2 text-lg font-bold text-ellipsis overflow-hidden whitespace-nowrap">{node.label}</p>
        </div>
        {/* 显示节点的输入部分 */}
        {inputs && inputs.length > 0 && (
          <>
            <TypeLabel label="Inputs" />
            {/* 显示每个输入节点处理器 */}
            {(inputs || []).map((input, index) => (
              <NodeHandler key={`${node.id}_input_${index}`} node={node} data={input} type="target" label="inputs" index={index} />
            ))}
          </>
        )}
        {/* 显示节点的参数部分 */}
        {parameters && parameters.length > 0 && (
          <>
            <TypeLabel label="Parameters" />
            {/* 显示每个参数节点处理器 */}
            {(parameters || []).map((parameter, index) => (
              <NodeParamHandler key={`${node.id}_param_${index}`} node={node} data={parameter} label="parameters" index={index} />
            ))}
          </>
        )}
        {/* 渲染节点的输出部分 */}
        {renderOutput(node)}
      </div>
    </Popover>
  );
};

export default CanvasNode;
```
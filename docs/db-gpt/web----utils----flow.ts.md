# `.\DB-GPT-src\web\utils\flow.ts`

```py
import { IFlowData, IFlowDataNode, IFlowNode } from '@/types/flow';
import { Node } from 'reactflow';

// 寻找唯一节点ID，基于节点数据和节点数组
export const getUniqueNodeId = (nodeData: IFlowNode, nodes: Node[]) => {
  let count = 0; // 初始化计数器为0
  nodes.forEach((node) => { // 遍历节点数组
    if (node.data.name === nodeData.name) { // 如果节点名称匹配
      count++; // 计数器加一
    }
  });
  return `${nodeData.id}_${count}`; // 返回拼接的唯一节点ID
};

// 驼峰转下划线，映射接口数据字段命名规范
export const mapHumpToUnderline = (flowData: IFlowData) => {
  /**
   * sourceHandle -> source_handle,
   * targetHandle -> target_handle,
   * positionAbsolute -> position_absolute
   */
  const { nodes, edges, ...rest } = flowData; // 解构流数据对象
  const newNodes = nodes.map((node) => { // 映射新的节点数组
    const { positionAbsolute, ...rest } = node; // 解构节点属性
    return {
      position_absolute: positionAbsolute, // 映射驼峰命名为下划线命名
      ...rest,
    };
  });
  const newEdges = edges.map((edge) => { // 映射新的边缘数组
    const { sourceHandle, targetHandle, ...rest } = edge; // 解构边缘属性
    return {
      source_handle: sourceHandle, // 映射驼峰命名为下划线命名
      target_handle: targetHandle,
      ...rest,
    };
  });
  return {
    nodes: newNodes,
    edges: newEdges,
    ...rest,
  };
};

// 下划线转驼峰，映射接口数据字段命名规范
export const mapUnderlineToHump = (flowData: IFlowData) => {
  /**
   * source_handle -> sourceHandle,
   * target_handle -> targetHandle,
   * position_absolute -> positionAbsolute
   */
  const { nodes, edges, ...rest } = flowData; // 解构流数据对象
  const newNodes = nodes.map((node) => { // 映射新的节点数组
    const { position_absolute, ...rest } = node; // 解构节点属性
    return {
      positionAbsolute: position_absolute, // 映射下划线命名为驼峰命名
      ...rest,
    };
  });
  const newEdges = edges.map((edge) => { // 映射新的边缘数组
    const { source_handle, target_handle, ...rest } = edge; // 解构边缘属性
    return {
      sourceHandle: source_handle, // 映射下划线命名为驼峰命名
      targetHandle: target_handle,
      ...rest,
    };
  });
  return {
    nodes: newNodes,
    edges: newEdges,
    ...rest,
  };
};

// 检查流数据的必填项
export const checkFlowDataRequied = (flowData: IFlowData) => {
  const { nodes, edges } = flowData; // 解构流数据中的节点和边缘
  let result: [boolean, IFlowDataNode, string] = [true, nodes[0], '']; // 初始化结果数组
  outerLoop: for (let i = 0; i < nodes.length; i++) { // 外部循环遍历节点
    const node = nodes[i].data; // 获取节点的数据
    const { inputs = [], parameters = [] } = node; // 解构节点的输入和参数
    // 检查输入项
    for (let j = 0; j < inputs.length; j++) { // 内部循环遍历节点的输入项
      if (!edges.some((edge) => edge.targetHandle === `${nodes[i].id}|inputs|${j}`)) { // 如果边缘中没有匹配的目标句柄
        result = [false, nodes[i], `The input ${inputs[j].type_name} of node ${node.label} is required`]; // 更新结果数组
        break outerLoop; // 跳出外部循环
      }
    }
    // 检查参数项
    // (此处省略了未完整的代码)
    // 遍历 parameters 数组，检查每个参数的必需性和类型
    for (let k = 0; k < parameters.length; k++) {
      // 获取当前循环的参数对象
      const parameter = parameters[k];
      // 检查参数是否非可选、属于资源类别，并且在 edges 数组中不存在指向该参数的边
      if (!parameter.optional && parameter.category === 'resource' && !edges.some((edge) => edge.targetHandle === `${nodes[i].id}|parameters|${k}`)) {
        // 设置返回结果为包含错误信息的数组，表示缺少必需参数
        result = [false, nodes[i], `The parameter ${parameter.type_name} of node ${node.label} is required`];
        // 跳出外层循环标签，结束整体循环
        break outerLoop;
      // 检查参数是否非可选、属于普通类别，并且其值为 undefined 或 null
      } else if (!parameter.optional && parameter.category === 'common' && (parameter.value === undefined || parameter.value === null)) {
        // 设置返回结果为包含错误信息的数组，表示缺少必需参数
        result = [false, nodes[i], `The parameter ${parameter.type_name} of node ${node.label} is required`];
        // 跳出外层循环标签，结束整体循环
        break outerLoop;
      }
    }
  }
  // 返回最终的结果数组，包含验证结果和相关信息
  return result;
};


注释：


# 这是一个单独的分号，用于结束某个语句或表达式。在大多数编程语言中，分号用来分隔语句或表达式，表示语句的结束。
```
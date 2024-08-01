# `.\DB-GPT-src\web\types\flow.ts`

```py
// 导入 Node 类型来自 'reactflow' 库
import { Node } from 'reactflow';

// 定义流程状态的枚举类型
export type FlowState = 'deployed' | 'developing' | 'initializing' | 'testing' | 'disabled' | 'running' | 'load_failed';

// 定义更新流程参数的接口
export type IFlowUpdateParam = {
  name: string;             // 流程名称
  label: string;            // 显示标签
  editable: boolean;        // 是否可编辑
  description: string;      // 描述信息
  uid?: string;             // 可选的唯一标识
  flow_data?: IFlowData;    // 可选的流程数据
  state?: FlowState;        // 可选的流程状态
};

// 定义流程的接口
export type IFlow = {
  dag_id: string;           // DAG ID
  gmt_created: string;      // 创建时间
  gmt_modified: string;     // 修改时间
  uid: string;              // 唯一标识
  name: string;             // 流程名称
  label: string;            // 显示标签
  editable: boolean;        // 是否可编辑
  description: string;      // 描述信息
  flow_data: IFlowData;     // 流程数据
  source: string;           // 来源
  state?: FlowState;        // 可选的流程状态
  error_message?: string;   // 可选的错误信息
};

// 定义流程响应的接口
export type IFlowResponse = {
  items: Array<IFlow>;      // 流程项数组
  total_count: number;      // 总数
  total_pages: number;      // 总页数
  page: number;             // 当前页码
  page_size: number;        // 每页大小
};

// 定义流程节点参数的接口
export type IFlowNodeParameter = {
  id: string;               // 参数ID
  type_name: string;        // 类型名称
  type_cls: string;         // 类型类别
  label: string;            // 显示标签
  name: string;             // 参数名称
  category: string;         // 类别
  optional: boolean;        // 是否可选
  default?: any;            // 可选的默认值
  placeholder?: any;        // 可选的占位符
  description: string;      // 描述信息
  options?: any;            // 可选的选项
  value: any;               // 参数值
  is_list?: boolean;        // 是否为列表
};

// 定义流程节点输入的接口
export type IFlowNodeInput = {
  type_name: string;        // 类型名称
  type_cls: string;         // 类型类别
  label: string;            // 显示标签
  name: string;             // 输入名称
  description: string;      // 描述信息
  id: string;               // 输入ID
  optional?: boolean | undefined; // 可选的是否可选
  value: any;               // 输入值
  is_list?: boolean;        // 是否为列表
};

// 定义流程节点输出的接口
export type IFlowNodeOutput = {
  type_name: string;        // 类型名称
  type_cls: string;         // 类型类别
  label: string;            // 显示标签
  name: string;             // 输出名称
  description: string;      // 描述信息
  id: string;               // 输出ID
  optional?: boolean | undefined; // 可选的是否可选
  is_list?: boolean;        // 是否为列表
};

// 定义流程节点的接口，继承自 Node 类型
export type IFlowNode = Node & {
  type_name: string;              // 类型名称
  type_cls: string;               // 类型类别
  parent_cls?: string;            // 父类类别（仅资源类节点有）
  label: string;                  // 显示标签
  name: string;                   // 节点名称
  description: string;            // 描述信息
  category: string;               // 类别
  category_label: string;         // 类别标签
  flow_type: 'resource' | 'operator'; // 流程类型：资源或操作符
  icon?: string;                  // 图标
  documentation_url?: null;       // 文档 URL（可能为 null）
  id: string;                     // 节点ID
  tags?: any;                     // 标签（任意类型）
  parameters: Array<IFlowNodeParameter>; // 参数数组
  inputs: Array<IFlowNodeInput>;   // 输入数组
  outputs: Array<IFlowNodeOutput>; // 输出数组
  version: string;                // 版本号
  invalid?: boolean;              // 是否无效
};

// 定义位置接口
interface Position {
  x: number;    // x 坐标
  y: number;    // y 坐标
  zoom: number; // 缩放比例
}

// 定义流程数据节点的接口
export type IFlowDataNode = {
  width: number;                  // 宽度
  height: number;                 // 高度
  id: string;                     // ID
  position: Position;             // 位置信息
  position_absolute?: Position;   // 绝对位置信息（可选）
  positionAbsolute?: Position;    // 绝对位置信息（可选）
  data: IFlowNode;                // 节点数据
  type: string;                   // 类型
};

// 定义流程数据边的接口
export type IFlowDataEdge = {
  source: string;                 // 源节点ID
  target: string;                 // 目标节点ID
  source_handle?: string;         // 源节点处理程序（可选）
  sourceHandle?: string;          // 源节点处理程序（可选）
  target_handle?: string;         // 目标节点处理程序（可选）
  targetHandle?: string;          // 目标节点处理程序（可选）
  id: string;                     // 边的ID
  type: string;                   // 类型
};

// 定义流程数据视口的接口
export type IFlowDataViewport = {
  x: number;                      // x 坐标
  y: number;                      // y 坐标
  zoom: number;                   // 缩放比例
};

// 定义流程数据的接口
export type IFlowData = {
  nodes: Array<IFlowDataNode>;    // 节点数组
  edges: Array<IFlowDataEdge>;    // 边数组
  viewport: IFlowDataViewport;    // 视口信息
};
```
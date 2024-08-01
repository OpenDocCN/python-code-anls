# `.\DB-GPT-src\web\types\knowledge.ts`

```py
// 引入 Node.js 中的操作系统模块中的类型定义
import { type } from 'os';

// 表示空间的接口，具有上下文、描述、文档、创建时间、修改时间、标识、名称、所有者、向量类型和域类型等属性
export interface ISpace {
  context?: any; // 可选的上下文信息
  desc: string; // 空间描述
  docs: string | number; // 文档数量或者其他相关文档信息
  gmt_created: string; // 创建时间
  gmt_modified: string; // 修改时间
  id: string | number; // 空间标识
  name: string; // 空间名称
  owner: string; // 所有者名称
  vector_type: string; // 向量类型
  domain_type: string; // 域类型
}

// 添加知识的参数类型定义，包括名称、向量类型、所有者、描述和域类型
export type AddKnowledgeParams = {
  name: string;
  vector_type: string;
  owner: string;
  desc: string;
  domain_type: string;
};

// 基础文档参数类型定义，包括文档名称、内容、文档类型
export type BaseDocumentParams = {
  doc_name: string;
  content: string;
  doc_type: string;
};

// 嵌入（Embedding）对象类型定义，包括块重叠、块大小、模型、召回分数类型和数量、以及 topk 值等
export type Embedding = {
  chunk_overlap: string | number;
  chunk_size: string | number;
  model: string;
  recall_score: string | number;
  recall_type: string;
  topk: string;
};

// 提示（Prompt）对象类型定义，包括最大令牌数、场景和模板
export type Prompt = {
  max_token: string | number;
  scene: string;
  template: string;
};

// 摘要（Summary）对象类型定义，包括最大迭代次数和并发限制
export type Summary = {
  max_iteration: number;
  concurrency_limit: number;
};

// 参数（IArguments）对象类型定义，包括嵌入、提示和摘要等对象
export type IArguments = {
  embedding: Embedding;
  prompt: Prompt;
  summary: Summary;
};

// 文档参数（DocumentParams）类型定义，包括文档名称、来源、内容和文档类型等
export type DocumentParams = {
  doc_name: string;
  source?: string;
  content: string;
  doc_type: string;
};

// 文档（IDocument）类型定义，包括文档名称、来源、内容、文档类型、块大小、创建时间、修改时间、标识、上次同步时间、结果、空间和状态等
export type IDocument = {
  doc_name: string;
  source?: string;
  content: string;
  doc_type: string;
  chunk_size: string | number;
  gmt_created: string;
  gmt_modified: string;
  id: number;
  last_sync: string;
  result: string;
  space: string;
  status: string;
  vector_ids: string;
};

// 文档响应（IDocumentResponse）类型定义，包括文档数据数组、页数和总数
export type IDocumentResponse = {
  data: Array<IDocument>;
  page: number;
  total: number;
};

// 策略参数（IStrategyParameter）类型定义，包括参数名称、类型、默认值、描述等
export type IStrategyParameter = {
  param_name: string;
  param_type: string;
  default_value?: string | number;
  description: string;
};

// 块策略响应（IChunkStrategyResponse）类型定义，包括策略、名称、参数数组、后缀数组和类型数组
export type IChunkStrategyResponse = {
  strategy: string;
  name: string;
  parameters: Array<IStrategyParameter>;
  suffix: Array<string>;
  type: Array<string>;
};

// 策略属性（IStrategyProps）类型定义，包括块策略、块大小和块重叠等
export type IStrategyProps = {
  chunk_strategy: string;
  chunk_size?: number;
  chunk_overlap?: number;
};

// 同步批处理参数（ISyncBatchParameter）类型定义，包括文档标识、名称、块参数等
export type ISyncBatchParameter = {
  doc_id: number;
  name?: string;
  chunk_parameters: IStrategyProps;
};

// 同步批处理响应（ISyncBatchResponse）类型定义，包括任务数组
export type ISyncBatchResponse = {
  tasks: Array<number>;
};

// 块列表参数（ChunkListParams）类型定义，包括文档标识、页数和页大小
export type ChunkListParams = {
  document_id?: string | number;
  page: number;
  page_size: number;
};

// 块（IChunk）类型定义，包括内容、文档名称、文档类型、文档标识、创建时间、修改时间、标识、元信息、召回分数等
export type IChunk = {
  content: string;
  doc_name: string;
  doc_type: string;
  document_id: string | number;
  gmt_created: string;
  gmt_modified: string;
  id: string | number;
  meta_info: string;
  recall_score?: string | number;
};

// 块列表（IChunkList）类型定义，包括块数据数组、页数和总数
export type IChunkList = {
  data: Array<IChunk>;
  page: number;
  total: number;
};

// 图形可视化结果（GraphVisResult）类型定义，包括节点数组和边数组
export type GraphVisResult = {
  nodes: Array<any>;
  edges: Array<any>;
};

// 参数参数（ArgumentsParams）类型定义，包括参数
export type ArgumentsParams = {
  argument: string;
};

// 步骤变更参数（StepChangeParams）类型定义，包括标签、空间名称、文档类型、文件数组和速度等
export type StepChangeParams = {
  label: 'forward' | 'back' | 'finish';
  spaceName?: string;
  docType?: string;
  files?: Array<File>;
  pace?: number;
};

// 文件类型（File）类型定义，包括名称、文档标识和状态等
export type File = {
  name: string;
  doc_id: number;
  status?: string;
};

// 摘要参数（SummaryParams）类型定义，包括文档标识、模型名称和会话标识
export type SummaryParams = {
  doc_id: number;
  model_name: string;
  conv_uid: string;
};

// 空间配置（SpaceConfig）类型定义，包括存储信息
export type SpaceConfig = {
  storage: IStorage;
};
# 定义了一个类型 IStorage，表示一个数组，数组的每个元素是一个对象，对象包含以下字段：
# - name: 字符串类型，表示存储项的名称
# - desc: 字符串类型，描述存储项的详细信息
# - domain_types: 数组类型，包含多个对象，每个对象有两个字段：
#   - name: 字符串类型，表示域类型的名称
#   - desc: 字符串类型，描述域类型的详细信息
export type IStorage = Array<{
  name: string;
  desc: string;
  domain_types: Array<{ name: string; desc: string }>;
}>;
```
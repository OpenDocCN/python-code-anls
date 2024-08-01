# `.\DB-GPT-src\web\types\chat.ts`

```py
/**
 * Represents the structure of a single value in a chart.
 */
type ChartValue = {
  name: string;         // 名称字段，用于标识数据点的名称
  type: string;         // 类型字段，描述数据点的类型
  value: number;        // 数值字段，存储具体的数值信息
};

/**
 * Represents the data structure of a chart used in a dashboard.
 */
export type ChartData = {
  chart_desc: string;                 // 图表描述信息
  chart_name: string;                 // 图表名称
  chart_sql: string;                  // 与图表相关的 SQL 查询语句
  chart_type: string;                 // 图表类型
  chart_uid: string;                  // 图表唯一标识符
  column_name: Array<string>;         // 列名数组，描述图表数据列的名称
  values: Array<ChartValue>;          // ChartValue 类型的数组，存储图表数据
  type?: string;                      // 可选的类型字段
};

/**
 * Represents the response structure of a scene in a dashboard.
 */
export type SceneResponse = {
  chat_scene: string;                 // 对话场景名称
  param_title: string;                // 参数标题
  scene_describe: string;             // 场景描述
  scene_name: string;                 // 场景名称
  show_disable: boolean;              // 是否禁用显示
};

/**
 * Represents the parameters for a new dialogue.
 */
export type NewDialogueParam = {
  chat_mode: string;                  // 对话模式类型
  model?: string;                     // 可选的模型名称
};

/**
 * Represents the response structure of chat history messages.
 */
export type ChatHistoryResponse = IChatDialogueMessageSchema[];

/**
 * Represents the structure of a single dialogue in the chat system.
 */
export type IChatDialogueSchema = {
  conv_uid: string;                   // 会话唯一标识符
  user_input: string;                 // 用户输入内容
  user_name: string;                  // 用户名称
  chat_mode:                           // 对话模式类型，使用联合类型
    | 'chat_with_db_execute'
    | 'chat_excel'
    | 'chat_with_db_qa'
    | 'chat_knowledge'
    | 'chat_dashboard'
    | 'chat_execution'
    | 'chat_agent'
    | 'chat_flow'
    | (string & {});
  select_param: string;               // 选择参数
};

/**
 * Represents the response structure of a list of dialogues.
 */
export type DialogueListResponse = IChatDialogueSchema[];

/**
 * Represents the structure of a single message in a chat dialogue.
 */
export type IChatDialogueMessageSchema = {
  role: 'human' | 'view' | 'system' | 'ai';  // 角色类型，指示消息是人类、视图、系统还是 AI
  context: string;                          // 消息内容
  order: number;                            // 消息排序顺序
  time_stamp: number | string | null;        // 时间戳，可为数字或字符串，或为空
  model_name: string;                       // 模型名称
  retry?: boolean;                          // 是否是重试消息，可选字段
};

/**
 * Represents the types of models used in the system.
 */
export type ModelType =
  | 'proxyllm'
  | 'flan-t5-base'
  | 'vicuna-13b'
  | 'vicuna-7b'
  | 'vicuna-13b-v1.5'
  | 'vicuna-7b-v1.5'
  | 'codegen2-1b'
  | 'codet5p-2b'
  | 'chatglm-6b-int4'
  | 'chatglm-6b'
  | 'chatglm2-6b'
  | 'chatglm2-6b-int4'
  | 'guanaco-33b-merged'
  | 'falcon-40b'
  | 'gorilla-7b'
  | 'gptj-6b'
  | 'proxyllm'
  | 'chatgpt_proxyllm'
  | 'bard_proxyllm'
  | 'claude_proxyllm'
  | 'wenxin_proxyllm'
  | 'tongyi_proxyllm'
  | 'zhipu_proxyllm'
  | 'llama-2-7b'
  | 'llama-2-13b'
  | 'llama-2-70b'
  | 'baichuan-7b'
  | 'baichuan-13b'
  | 'baichuan2-7b'
  | 'baichuan2-13b'
  | 'wizardlm-13b'
  | 'llama-cpp'
  | (string & {});

/**
 * Represents options for a label with an icon.
 */
export type LLMOption = {
  label: string;   // 标签文本
  icon: string;    // 图标名称或路径
};

/**
 * Represents feedback options for various purposes.
 */
export type FeedBack = {
  information?: string;   // 信息反馈
  just_fun?: string;      // 娱乐性反馈
  others?: string;        // 其他类型反馈
  work_study?: string;    // 工作或学习相关反馈
};

/**
 * Represents a reference with named chunks.
 */
export type Reference = {
  name: string;          // 引用名称
  chunks: Array<number>; // 数组，包含引用的块编号
};

/**
 * Represents the structure of a database configuration.
 */
export type IDB = {
  param: string;   // 参数名称
  type: string;    // 参数类型
};
```
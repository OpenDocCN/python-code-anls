# `.\DB-GPT-src\web\types\model.ts`

```py
// 定义接口 IModelData，表示模型数据结构
export type IModelData = {
  chat_scene: string;                   // 聊天场景
  model_name: string;                   // 模型名称
  model_type: string;                   // 模型类型
  host: string;                         // 主机地址
  port: number;                         // 端口号
  manager_host: string;                 // 管理员主机地址
  manager_port: number;                 // 管理员端口号
  healthy: boolean;                     // 健康状态
  check_healthy: boolean;               // 检查健康状态
  prompt_template: string;              // 提示模板
  last_heartbeat: string;               // 上次心跳时间
  stream_api: string;                   // 流式 API 地址
  nostream_api: string;                 // 非流式 API 地址
};

// 定义接口 BaseModelParams，表示基础模型参数
export type BaseModelParams = {
  host: string;                         // 主机地址
  port: number;                         // 端口号
  model: string;                        // 模型名称
  worker_type: string;                  // 工作类型
  params: any;                          // 其他参数
};

// 定义接口 ModelParams，表示模型参数
export type ModelParams = {
  model_name: string;                   // 模型名称
  model_path: string;                   // 模型路径
  proxy_api_key: string;                // 代理 API 密钥
  proxy_server_url: string;             // 代理服务器 URL
  model_type: string;                   // 模型类型
  max_context_size: number;             // 最大上下文大小
};

// 定义接口 StartModelParams，表示启动模型参数
export type StartModelParams = {
  host: string;                         // 主机地址
  port: number;                         // 端口号
  model: string;                        // 模型名称
  worker_type: string;                  // 工作类型
  params: ModelParams;                  // 模型参数
};

// 定义接口 ExtMetadata，表示扩展元数据
interface ExtMetadata {
  tags: string;                         // 标签
}

// 定义接口 SupportModelParams，表示支持模型参数
export type SupportModelParams = {
  param_class: string;                  // 参数类别
  param_name: string;                   // 参数名称
  param_type: string;                   // 参数类型
  default_value: string | boolean | number;  // 默认值（可以是字符串、布尔值或数字）
  description: string;                  // 描述
  required: boolean;                    // 是否必需
  valid_values: null;                   // 有效值（目前为 null）
  ext_metadata: ExtMetadata;            // 扩展元数据
};

// 定义接口 SupportModel，表示支持的模型
export type SupportModel = {
  model: string;                        // 模型名称
  path: string;                         // 路径
  worker_type: string;                  // 工作类型
  path_exist: boolean;                  // 路径是否存在
  proxy: boolean;                       // 是否使用代理
  enabled: boolean;                     // 是否启用
  host: string;                         // 主机地址
  port: number;                         // 端口号
  params: SupportModelParams;           // 支持模型参数
};
```
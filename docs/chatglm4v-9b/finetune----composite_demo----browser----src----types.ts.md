# `.\chatglm4-finetune\composite_demo\browser\src\types.ts`

```
# 定义文件接口，包含文件的 ID、名称和大小
export interface File {
  # 文件的唯一标识符
  id: string;
  # 文件的名称
  name: string;
  # 文件的大小，以字节为单位
  size: number;
}

# 定义元数据接口，包含文件列表和引用字符串（可选）
export interface Metadata {
  # 文件列表，类型为 File 数组（可选）
  files?: File[];
  # 引用字符串（可选）
  reference?: string;
}

# 定义消息接口，表示用户、助手、系统或观察者的消息
export interface Message {
  # 消息角色，限定为特定的字符串类型
  role: 'user' | 'assistant' | 'system' | 'observation';
  # 消息的元数据，类型为字符串
  metadata: string;
  # 消息的内容，类型为字符串
  content: string;
  # 请求元数据（可选）
  request_metadata?: Metadata;
}

# 定义工具观察接口，描述工具执行结果
export interface ToolObservation {
  # 内容类型，表示结果的 MIME 类型
  contentType: string;
  # 工具的执行结果
  result: string;
  # 可能的文本内容（可选）
  text?: string;
  # 观察者角色的元数据（可选）
  roleMetadata?: string; // metadata for <|observation|>${metadata}
  # 响应的元数据，类型为任意
  metadata: any; // metadata for response
}
```
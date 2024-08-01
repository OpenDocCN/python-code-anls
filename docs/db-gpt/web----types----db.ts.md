# `.\DB-GPT-src\web\types\db.ts`

```py
// 定义一个类型，表示数据库选项的对象结构
export type DBOption = {
  label: string;             // 选项显示的文本标签
  value: DBType;             // 选项对应的数据库类型
  disabled?: boolean;        // 是否禁用该选项的标志，可选
  isFileDb?: boolean;        // 是否为文件型数据库的标志，可选
  icon: string;              // 选项显示的图标路径
  desc?: string;             // 选项的描述信息，可选
};

// 定义一个联合类型，表示支持的数据库类型
export type DBType =
  | 'mysql'
  | 'duckdb'
  | 'sqlite'
  | 'mssql'
  | 'clickhouse'
  | 'oracle'
  | 'postgresql'
  | 'vertica'
  | 'db2'
  | 'access'
  | 'mongodb'
  | 'starrocks'
  | 'hbase'
  | 'redis'
  | 'cassandra'
  | 'couchbase'
  | (string & {});           // 允许未知的数据库类型，以字符串形式表示

// 定义一个接口，表示数据库模式的结构
export type IChatDbSchema = {
  comment: string;           // 数据库的描述信息
  db_host: string;           // 数据库主机地址
  db_name: string;           // 数据库名称
  db_path: string;           // 数据库文件路径或其他特定路径
  db_port: number;           // 数据库连接端口
  db_pwd: string;            // 数据库连接密码
  db_type: DBType;           // 数据库类型
  db_user: string;           // 数据库连接用户名
};

// 定义一个类型，表示数据库列表的响应数据结构
export type DbListResponse = IChatDbSchema[];

// 定义一个接口，表示数据库支持的类型的结构
export type IChatDbSupportTypeSchema = {
  db_type: DBType;           // 数据库类型
  is_file_db: boolean;       // 是否为文件型数据库
};

// 定义一个类型，表示数据库支持类型列表的响应数据结构
export type DbSupportTypeResponse = IChatDbSupportTypeSchema[];

// 定义一个类型，表示向服务器发送的更新数据库请求的参数结构
export type PostDbParams = Partial<DbListResponse[0] & { file_path: string }>;

// 定义一个类型，表示聊天反馈数据的结构
export type ChatFeedBackSchema = {
  conv_uid: string;          // 会话唯一标识
  conv_index: number;        // 会话索引号
  question: string;          // 提出的问题
  knowledge_space: string;   // 关联的知识空间
  score: number;             // 问题的得分
  ques_type: string;         // 问题的类型
  messages: string;          // 相关的消息内容
};

// 定义一个类型，表示提示组件的属性
export type PromptProps = {
  id: number;                // 提示的唯一标识
  chat_scene: string;        // 聊天场景
  sub_chat_scene: string;    // 子聊天场景
  prompt_type: string;       // 提示类型
  content: string;           // 提示的内容
  user_name: string;         // 用户名
  prompt_name: string;       // 提示的名称
  gmt_created: string;       // 提示创建时间（GMT）
  gmt_modified: string;      // 提示修改时间（GMT）
};

// 定义一个类型，表示向服务器发送刷新数据库请求的参数结构
export type PostDbRefreshParams = {
  db_name: string;           // 需要刷新的数据库名称
  db_type: DBType;           // 需要刷新的数据库类型
};
```
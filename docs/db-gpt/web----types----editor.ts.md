# `.\DB-GPT-src\web\types\editor.ts`

```py
# 定义一个类型 IEditorSQLRound，表示编辑器中的 SQL 回合信息
export type IEditorSQLRound = {
  db_name: string;         # 数据库名称，字符串类型
  round: number;           # 回合数，数字类型
  round_name: string;      # 回合名称，字符串类型
};

# 定义一个类型 GetEditorSQLRoundRequest，表示获取编辑器 SQL 回合的请求
export type GetEditorSQLRoundRequest = IEditorSQLRound[];

# 定义一个类型 PostEditorSQLRunParams，表示提交编辑器中 SQL 运行的参数
export type PostEditorSQLRunParams = {
  db_name: string;         # 数据库名称，字符串类型
  sql: string;             # SQL 查询语句，字符串类型
};

# 定义一个类型 PostEditorChartRunParams，表示提交编辑器中图表运行的参数
export type PostEditorChartRunParams = {
  db_name: string;         # 数据库名称，字符串类型
  sql?: string;            # 可选的 SQL 查询语句，字符串类型
  chart_type?: string;     # 可选的图表类型，字符串类型
};

# 定义一个类型 PostEditorChartRunResponse，表示编辑器中图表运行的响应
export type PostEditorChartRunResponse = {
  sql_data: {               # SQL 查询数据对象
    result_info: string;    # 结果信息，字符串类型
    run_cost: string;       # 运行成本，字符串类型
    colunms: string[];      # 列名数组，字符串数组类型
    values: Record<string, any>[];  # 值数组，每个元素为包含列名和值的对象的数组
  };
  chart_values: Record<string, any>[];  # 图表数值数组，每个元素为包含图表值的对象的数组
  chart_type: string;       # 图表类型，字符串类型
};

# 定义一个类型 PostSQLEditorSubmitParams，表示提交 SQL 编辑器内容的参数
export type PostSQLEditorSubmitParams = {
  conv_uid: string;        # 会话唯一标识符，字符串类型
  db_name: string;         # 数据库名称，字符串类型
  conv_round?: string | number | null;  # 可选的会话回合，字符串、数字或 null 类型
  old_sql?: string;        # 可选的旧 SQL 查询语句，字符串类型
  old_speak?: string;      # 可选的旧说话内容，字符串类型
  new_sql?: string;        # 可选的新 SQL 查询语句，字符串类型
  new_speak?: string;      # 可选的新说话内容，字符串类型
};

# 定义一个类型 PostEditorSqlParams，表示提交编辑器 SQL 内容的参数
export type PostEditorSqlParams = {
  con_uid: string;         # 连接唯一标识符，字符串类型
  round: string | number;  # 回合数，字符串或数字类型
};

# 定义一个类型 PostEditorSqlRequest，表示提交编辑器 SQL 请求的空参数
export type PostEditorSqlRequest = {};

# 定义一个类型 GetEditorySqlParams，表示获取编辑器 SQL 参数的参数
export type GetEditorySqlParams = { 
  con_uid: string;         # 连接唯一标识符，字符串类型
  round: string | number   # 回合数，字符串或数字类型
};
```
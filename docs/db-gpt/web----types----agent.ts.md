# `.\DB-GPT-src\web\types\agent.ts`

```py
# 定义一个类型，用于表示提交到代理人中心的更新参数
export type PostAgentHubUpdateParams = {
  channel: string;          # 更新的渠道
  url: string;              # 更新的 URL 地址
  branch: string;           # 更新的分支
  authorization: string;    # 授权信息
};

# 定义一个类型，用于表示向代理人查询发送的参数
export type PostAgentQueryParams = {
  page_index: number;       # 查询的页码索引
  page_size: number;        # 查询的每页大小
  filter?: {                # 可选的查询过滤条件
    name?: string;          # 名称过滤
    description?: string;   # 描述过滤
    author?: string;        # 作者过滤
    email?: string;         # 邮箱过滤
    type?: string;          # 类型过滤
    version?: string;       # 版本过滤
    storage_channel?: string;  # 存储渠道过滤
    storage_url?: string;   # 存储 URL 过滤
  };
};

# 定义一个类型，表示代理插件的信息
export type IAgentPlugin = {
  name: string;             # 插件名称
  description: string;      # 插件描述
  email: string;            # 插件作者邮箱
  version: string;          # 插件版本
  storage_url: string;      # 存储插件的 URL 地址
  download_param: string;   # 插件下载参数
  installed: number;        # 已安装次数
  id: number;               # 插件 ID
  author: string;           # 插件作者
  type: string;             # 插件类型
  storage_channel: string;  # 存储插件的渠道
  created_at: string;       # 插件创建时间
};

# 定义一个类型，表示代理插件查询的返回结果
export type PostAgentPluginResponse = {
  page_index: number;       # 返回结果的页码索引
  page_size: number;        # 返回结果的每页大小
  total_page: number;       # 总页数
  total_row_count: number;  # 总行数
  datas: IAgentPlugin[];    # 插件信息数组
};

# 定义一个类型，表示我的插件信息
export type IMyPlugin = {
  user_name: null | string; # 用户名，可以为空
  id: number;               # 插件 ID
  file_name: string;        # 插件文件名
  version: string;          # 插件版本
  succ_count: number;       # 成功次数
  name: string;             # 插件名称
  tenant: null | string;    # 租户，可以为空
  user_code: string;        # 用户代码
  type: string;             # 插件类型
  use_count: number;        # 使用次数
  created_at: string;       # 插件创建时间
  description: string;      # 插件描述
};

# 定义一个类型，表示我的插件查询的返回结果
export type PostAgentMyPluginResponse = IMyPlugin[];

# 定义一个类型，表示获取 DBGPTs 列表的返回结果
export type GetDBGPTsListResponse = {
  app_code: string;         # 应用代码
  app_describe: string;     # 应用描述
  app_name: string;         # 应用名称
  language: string;         # 编程语言
  sys_code: string;         # 系统代码
  updated_at: string;       # 更新时间
  team_mode: string;        # 团队模式
  id: number;               # DBGPT ID
  user_code: string;        # 用户代码
  created_at: string;       # 创建时间
}[];
```
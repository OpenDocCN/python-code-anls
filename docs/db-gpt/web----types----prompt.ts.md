# `.\DB-GPT-src\web\types\prompt.ts`

```py
export type PromptParams = {
  prompt_type: string;  // 定义 PromptParams 类型，包含 prompt_type 字段，表示提示类型
  current: number;  // 当前页数
  pageSize: number;  // 每页数据量
  hideOnSinglePage: boolean;  // 是否在单页时隐藏
  showQuickJumper: boolean;  // 是否显示快速跳转器
};

export interface UpdatePromptParams extends IPrompt {
  prompt_type: string;  // 定义 UpdatePromptParams 接口，扩展自 IPrompt 接口，包含 prompt_type 字段
}

export interface IPrompt {
  chat_scene: string;  // 聊天场景
  content: string;  // 内容
  gmt_created: string;  // 创建时间
  gmt_modified: string;  // 修改时间
  id: number;  // ID
  prompt_name: string;  // 提示名称
  prompt_type: string;  // 提示类型
  sub_chat_scene: string;  // 子聊天场景
  user_name?: string;  // 用户名（可选）
}
```
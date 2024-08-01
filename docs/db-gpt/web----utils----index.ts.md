# `.\DB-GPT-src\web\utils\index.ts`

```py
import { format } from 'sql-formatter';

/** Theme */
// 存储主题设置的键名
export const STORAGE_THEME_KEY = '__db_gpt_theme_key';
/** Language */
// 存储语言设置的键名
export const STORAGE_LANG_KEY = '__db_gpt_lng_key';
/** Init Message */
// 存储初始化消息的键名
export const STORAGE_INIT_MESSAGE_KEY = '__db_gpt_im_key';
/** Flow nodes */
// 存储流程节点的静态键名
export const FLOW_NODES_KEY = '__db_gpt_static_flow_nodes_key';

// 格式化 SQL 语句的函数，支持指定语言
export function formatSql(sql: string, lang?: string) {
  // 如果 SQL 语句为空，则返回空字符串
  if (!sql) return '';
  try {
    // 使用 sql-formatter 库格式化 SQL 语句，可指定语言
    return format(sql, { language: lang });
  } catch (e) {
    // 格式化失败则返回原始 SQL 语句
    return sql;
  }
}

// 导出所有与存储相关的函数和常量
export * from './storage';
// 导出所有常量
export * from './constants';
```
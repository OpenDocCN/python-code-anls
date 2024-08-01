# `.\DB-GPT-src\web\utils\storage.ts`

```py
import { STORAGE_INIT_MESSAGE_KET } from '@/utils';
从工具库中导入 STORAGE_INIT_MESSAGE_KET 常量，用于访问本地存储中初始化消息的键名

export function getInitMessage() {
  // 获取本地存储中指定键名的值，如果不存在则返回空字符串
  const value = localStorage.getItem(STORAGE_INIT_MESSAGE_KET) ?? '';

  try {
    // 尝试将存储的值解析为 JSON 对象，结构必须符合 { id: string; message: string } 类型
    const initData = JSON.parse(value) as { id: string; message: string };
    // 返回解析后的初始化数据对象
    return initData;
  } catch (e) {
    // 如果解析失败，返回 null 表示无法获取有效的初始化数据
    return null;
  }
}
```
# `.\DB-GPT-src\web\components\chat\ob-editor\service.ts`

```py
import type { IModelOptions } from '@oceanbase-odc/monaco-plugin-ob/dist/type';
import { ISession } from '../monaco-editor';

// 定义函数 getModelService，返回类型为 IModelOptions 的对象
export function getModelService(
  // 参数包括 modelId 和 delimiter，类型为字符串，以及可选的 session 函数
  { modelId, delimiter }: { modelId: string; delimiter: string },
  session?: () => ISession | null
): IModelOptions {
  // 返回一个对象，其中包含 delimiter 属性和三个异步方法
  return {
    delimiter,
    // 异步方法：根据给定的 schemaName 获取表格列表，如果 session 存在则调用其 getTableList 方法，否则返回空数组
    async getTableList(schemaName?: string) {
      return session?.()?.getTableList(schemaName) || [];
    },
    // 异步方法：根据给定的 tableName 和 dbName 获取表格列信息，如果 session 存在则调用其 getTableColumns 方法，否则返回空数组
    async getTableColumns(tableName: string, dbName?: string) {
      return session?.()?.getTableColumns(tableName) || [];
    },
    // 异步方法：获取数据库的 schema 列表，如果 session 存在则调用其 getSchemaList 方法，否则返回空数组
    async getSchemaList() {
      return session?.()?.getSchemaList() || [];
    },
  };
}
```
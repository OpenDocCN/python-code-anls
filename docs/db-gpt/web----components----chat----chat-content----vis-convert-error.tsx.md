# `.\DB-GPT-src\web\components\chat\chat-content\vis-convert-error.tsx`

```py
# 导入代码预览组件和 SQL 格式化工具函数
import { CodePreview } from './code-preview';
import { formatSql } from '@/utils';

# 定义 Props 接口，包含数据对象的结构
interface Props {
  data: {
    display_type: string;  # 显示类型
    sql: string;  # SQL 查询语句
    thought: string;  # 错误信息
  };
}

# 可视化转换错误组件，接收数据对象作为参数
function VisConvertError({ data }: Props) {
  # 返回一个包含错误信息和 SQL 代码的 div 元素
  return (
    <div className="rounded overflow-hidden">
      # 显示错误类型
      <div className="p-3 text-white bg-red-500 whitespace-normal">{data.display_type}</div>
      <div className="p-3 bg-red-50">
        <div className="mb-2 whitespace-normal">{data.thought}</div>  # 显示错误信息
        <CodePreview code={formatSql(data.sql)} language="sql" />  # 显示格式化后的 SQL 代码
      </div>
    </div>
  );
}

# 导出可视化转换错误组件
export default VisConvertError;
```
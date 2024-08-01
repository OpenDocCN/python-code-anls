# `.\DB-GPT-src\web\components\flow\required-icon.tsx`

```py
// 导入 React 库，用于支持 React 组件的创建与使用
import React from 'react';

// 定义 RequiredIcon 组件，类型为 React 函数组件，接收一个名为 optional 的可选布尔类型属性
const RequiredIcon: React.FC<{ optional?: boolean | undefined }> = ({ optional }) => {
  // 如果 optional 属性为 true 或未定义，返回 null，表示不显示任何内容
  if (optional) {
    return null;
  }
  // 如果 optional 属性为 false，返回一个 span 元素，显示一个红色星号作为必填标志
  return <span className="text-red-600 align-middle inline-block">&nbsp;*</span>;
};

// 导出 RequiredIcon 组件，使其可以在其他文件中使用
export default RequiredIcon;
```
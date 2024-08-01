# `.\DB-GPT-src\web\components\common\icon-wrapper.tsx`

```py
# 导入 React 库中的 classNames 和 React
import classNames from 'classnames';
import React from 'react';

# 定义 IconWrapperProps 接口，指定 children 属性为 React 的节点，className 可选为字符串
interface IconWrapperProps {
  children: React.ReactNode;
  className?: string;
}

# IconWrapper 是一个 React 函数组件，用于包装图标，并设置背景颜色和悬停颜色，同时固定宽度和高度
const IconWrapper: React.FC<IconWrapperProps> = ({ children, className }) => {
  return (
    # 返回一个 div 元素作为图标的容器，设置其类名根据条件动态生成
    <div
      className={classNames(
        'flex justify-center items-center w-8 h-8 rounded-full dark:bg-zinc-700 hover:bg-stone-200 dark:hover:bg-zinc-900',
        className,
      )}
    >
      {children}  # 将传入的 children 属性渲染在容器内部
    </div>
  );
};

# 导出 IconWrapper 组件作为默认导出
export default IconWrapper;
```
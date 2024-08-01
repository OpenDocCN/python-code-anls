# `.\DB-GPT-src\web\components\common\db-icon.tsx`

```py
import Image from 'next/image';  // 导入 next/image 模块

interface IProps {  // 定义接口 IProps，描述组件的属性
  width?: number;  // 可选属性：图像宽度
  height?: number;  // 可选属性：图像高度
  src: string;  // 必需属性：图像的源地址
  label: string;  // 必需属性：图像的描述标签
  className?: string;  // 可选属性：自定义样式类名
}

function DBIcon({ src, label, width, height, className }: IProps) {  // 定义函数组件 DBIcon，接收 IProps 中定义的属性
  return (  // 返回一个组件
    <Image
      className={`w-11 h-11 rounded-full mr-4 border border-gray-200 object-contain bg-white ${className}`}  // 图像组件的样式类名，包括预设样式和传入的自定义样式
      width={width || 44}  // 图像宽度，默认为 44，或者根据传入的 width 属性
      height={height || 44}  // 图像高度，默认为 44，或者根据传入的 height 属性
      src={src}  // 图像的源地址，由传入的 src 属性决定
      alt={label || 'db-icon'}  // 图像的替代文本，如果未提供 label 属性则默认为 'db-icon'
    />
  );
}

export default DBIcon;  // 导出 DBIcon 组件作为默认导出
```
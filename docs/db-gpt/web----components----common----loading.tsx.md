# `.\DB-GPT-src\web\components\common\loading.tsx`

```py
import { LoadingOutlined } from '@ant-design/icons';

// 从 ant-design/icons 中导入 LoadingOutlined 组件


function MuiLoading({ visible }: { visible: boolean }) {

// 定义名为 MuiLoading 的 React 组件，接受一个布尔类型的 visible 属性作为参数


  if (!visible) return null;

// 如果 visible 属性为假（即 false 或 undefined），则返回 null，不显示组件内容


  return (
    <div className="absolute w-full h-full top-0 left-0 flex justify-center items-center z-10 bg-white dark:bg-black bg-opacity-50 dark:bg-opacity-50 backdrop-blur-sm text-3xl animate-fade animate-duration-200">

// 如果 visible 为真，则渲染以下内容：
// - 一个绝对定位的 div 元素，覆盖整个父容器，内容水平和垂直居中对齐
// - 其背景颜色为白色或黑色（取决于当前主题），并使用透明度为 50%
// - 应用 backdrop 模糊效果 backdrop-blur-sm
// - 文本大小为 3xl，应用淡入动画效果 animate-fade，持续时间为 200 毫秒


      <LoadingOutlined />
    </div>
  );
}

// 在上述 div 内部渲染 LoadingOutlined 组件，用于显示加载动画


export default MuiLoading;

// 导出 MuiLoading 组件，使其可以在其他文件中导入和使用
```
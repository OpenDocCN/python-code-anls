# `.\DB-GPT-src\web\components\icons\pending-icon.tsx`

```py
// 定义一个 React 组件 PendingIcon，用于显示一个带有动态加载状态的 SVG 图标
export default function PendingIcon() {
  // 返回一个 SVG 元素，指定视图框和版本信息，设置大小为 1.5em × 1.5em
  return (
    <svg viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="4260" width="1.5em" height="1.5em">
      {/* 第一个路径：绘制一个矩形区域，表示加载中的背景 */}
      <path d="M114.5856 951.04h298.24v-71.68H186.2656v-747.52h593.92v271.36h71.68v-343.04h-737.28v890.88z" fill="#747690" p-id="4261"></path>
      {/* 第二个路径：绘制加载条 */}
      <path
        d="M662.4256 311.04h-358.4v-71.68h358.4v71.68zM508.8256 490.24h-204.8v-71.68h204.8v71.68zM668.8256 554.24a168.96 168.96 0 1 0 0 337.92 168.96 168.96 0 0 0 0-337.92z m-240.64 168.96a240.64 240.64 0 1 1 481.28 0 240.64 240.64 0 0 1-481.28 0z"
        fill="#747690"
        p-id="4262"
      ></path>
      {/* 第三个路径：绘制加载指示器 */}
      <path d="M629.76 588.8h71.68v131.4304l82.5856 41.3184-32.0512 64.1024-122.2144-61.0816V588.8z" fill="#747690" p-id="4263"></path>
    </svg>
  );
}
```
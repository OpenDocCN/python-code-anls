# `basic-computer-games\86_Target\csharp\Offset.cs`

```
# 引入 System 命名空间
using System;

# 定义 Target 命名空间
namespace Target
{
    # 定义 Offset 类
    internal class Offset
    {
        # 定义 Offset 类的构造函数，接受三个参数 deltaX、deltaY、deltaZ
        public Offset(float deltaX, float deltaY, float deltaZ)
        {
            # 将参数赋值给对应的属性
            DeltaX = deltaX;
            DeltaY = deltaY;
            DeltaZ = deltaZ;

            # 计算三维坐标的距离并赋值给 Distance 属性
            Distance = (float)Math.Sqrt(deltaX * deltaX + deltaY * deltaY + deltaZ + deltaZ);
        }

        # 定义 DeltaX 属性，只读
        public float DeltaX { get; }
        # 定义 DeltaY 属性，只读
        public float DeltaY { get; }
        # 定义 DeltaZ 属性，只读
        public float DeltaZ { get; }
        # 定义 Distance 属性，只读
        public float Distance { get; }
    }
}
```
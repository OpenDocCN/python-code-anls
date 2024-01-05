# `62_Mugwump\csharp\IRandomExtensions.cs`

```
# 定义命名空间 Mugwump
namespace Mugwump;

# 定义一个内部静态类 IRandomExtensions
internal static class IRandomExtensions
{
    # 定义一个内部静态方法 NextPosition，接收一个 IRandom 对象和两个整数参数，返回一个 Position 对象
    internal static Position NextPosition(this IRandom random, int maxX, int maxY) =>
        # 创建一个新的 Position 对象，其 x 坐标为 random.Next(maxX)，y 坐标为 random.Next(maxY)
        new(random.Next(maxX), random.Next(maxY));
}
```
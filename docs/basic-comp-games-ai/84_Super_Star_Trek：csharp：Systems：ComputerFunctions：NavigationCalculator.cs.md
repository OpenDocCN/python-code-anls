# `d:/src/tocomm/basic-computer-games\84_Super_Star_Trek\csharp\Systems\ComputerFunctions\NavigationCalculator.cs`

```
using Games.Common.IO;  # 导入 Games.Common.IO 模块
using SuperStarTrek.Space;  # 导入 SuperStarTrek.Space 模块
using SuperStarTrek.Utils;  # 导入 SuperStarTrek.Utils 模块

namespace SuperStarTrek.Systems.ComputerFunctions;  # 定义 SuperStarTrek.Systems.ComputerFunctions 命名空间

internal abstract class NavigationCalculator : ComputerFunction  # 定义一个抽象类 NavigationCalculator，继承自 ComputerFunction 类
{
    protected NavigationCalculator(string description, IReadWrite io)  # 定义 NavigationCalculator 类的构造函数，接受描述和 IReadWrite 对象作为参数
        : base(description, io)  # 调用父类的构造函数，传入描述和 IReadWrite 对象
    {
    }

    protected void WriteDirectionAndDistance(Coordinates from, Coordinates to)  # 定义一个方法 WriteDirectionAndDistance，接受起始坐标和目标坐标作为参数
    {
        var (direction, distance) = from.GetDirectionAndDistanceTo(to)  # 调用起始坐标的 GetDirectionAndDistanceTo 方法，获取方向和距离
        Write(direction, distance)  # 调用 Write 方法，传入方向和距离
    }

    protected void WriteDirectionAndDistance((float X, float Y) from, (float X, float Y) to)  # 定义一个方法 WriteDirectionAndDistance，接受起始坐标和目标坐标作为参数
    {
        // 从起点到终点计算方向和距离
        var (direction, distance) = DirectionAndDistance.From(from.X, from.Y).To(to.X, to.Y);
        // 调用 Write 方法，将方向和距离输出
        Write(direction, distance);
    }

    // 定义一个私有方法，用于输出方向和距离
    private void Write(float direction, float distance)
    {
        // 输出方向
        IO.WriteLine($"Direction = {direction}");
        // 输出距离
        IO.WriteLine($"Distance = {distance}");
    }
}
```
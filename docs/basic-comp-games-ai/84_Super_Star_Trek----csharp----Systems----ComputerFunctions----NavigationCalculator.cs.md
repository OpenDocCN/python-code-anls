# `basic-computer-games\84_Super_Star_Trek\csharp\Systems\ComputerFunctions\NavigationCalculator.cs`

```py
// 引入所需的命名空间
using Games.Common.IO;
using SuperStarTrek.Space;
using SuperStarTrek.Utils;

// 定义一个内部抽象类 NavigationCalculator，继承自 ComputerFunction
internal abstract class NavigationCalculator : ComputerFunction
{
    // 构造函数，接受描述和 IReadWrite 对象作为参数
    protected NavigationCalculator(string description, IReadWrite io)
        : base(description, io)
    {
    }

    // 写入起点和终点的方向和距离
    protected void WriteDirectionAndDistance(Coordinates from, Coordinates to)
    {
        // 获取起点到终点的方向和距离
        var (direction, distance) = from.GetDirectionAndDistanceTo(to);
        // 调用 Write 方法写入方向和距离
        Write(direction, distance);
    }

    // 写入起点和终点的方向和距离
    protected void WriteDirectionAndDistance((float X, float Y) from, (float X, float Y) to)
    {
        // 获取起点到终点的方向和距离
        var (direction, distance) = DirectionAndDistance.From(from.X, from.Y).To(to.X, to.Y);
        // 调用 Write 方法写入方向和距离
        Write(direction, distance);
    }

    // 写入方向和距离
    private void Write(float direction, float distance)
    {
        // 使用 IO 对象输出方向和距离
        IO.WriteLine($"Direction = {direction}");
        IO.WriteLine($"Distance = {distance}");
    }
}
```
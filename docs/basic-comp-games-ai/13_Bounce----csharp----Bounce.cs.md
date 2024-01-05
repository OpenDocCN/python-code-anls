# `13_Bounce\csharp\Bounce.cs`

```
namespace Bounce; // 命名空间声明

/// <summary>
/// Represents the bounce of the ball, calculating duration, height and position in time.
/// </summary>
/// <remarks>
/// All calculations are derived from the equation for projectile motion: s = vt + 0.5at^2
/// </remarks>
internal class Bounce // 内部类声明，用于表示球的弹跳，计算持续时间、高度和时间位置
{
    private const float _acceleration = -32; // feet/s^2，声明并初始化加速度常量

    private readonly float _velocity; // 声明只读的速度变量

    internal Bounce(float velocity) // 类的构造函数，接受速度参数
    {
        _velocity = velocity; // 初始化速度变量
    }

    public float Duration => -2 * _velocity / _acceleration; // 公共属性，返回持续时间的计算结果
    public float MaxHeight => 
        (float)Math.Round(-_velocity * _velocity / 2 / _acceleration, MidpointRounding.AwayFromZero);
    // 计算最大高度，根据公式：-v^2 / 2a，使用Math.Round进行四舍五入

    public float Plot(Graph graph, float startTime)
    {
        var time = 0f;
        for (; time <= Duration; time += graph.TimeIncrement)
        {
            var height = _velocity * time + _acceleration * time * time / 2;
            // 计算每个时间点的高度
            graph.Plot(startTime + time, height);
            // 在图表上绘制时间点和对应的高度
        }

        return startTime + time;
        // 返回结束时间
    }

    public Bounce Next(float elasticity) => new Bounce(_velocity * elasticity);
    // 计算下一个弹跳的速度，根据给定的弹性系数
}
```
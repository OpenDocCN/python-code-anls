# `basic-computer-games\13_Bounce\csharp\Bounce.cs`

```

namespace Bounce;

/// <summary>
/// Represents the bounce of the ball, calculating duration, height and position in time.
/// </summary>
/// <remarks>
/// All calculations are derived from the equation for projectile motion: s = vt + 0.5at^2
/// </remarks>
internal class Bounce
{
    private const float _acceleration = -32; // feet/s^2  // 设置加速度常量为-32英尺/秒^2

    private readonly float _velocity;  // 设置只读的速度变量

    internal Bounce(float velocity)  // 构造函数，初始化速度
    {
        _velocity = velocity;
    }

    public float Duration => -2 * _velocity / _acceleration;  // 计算弹跳持续时间

    public float MaxHeight =>  // 计算最大高度
        (float)Math.Round(-_velocity * _velocity / 2 / _acceleration, MidpointRounding.AwayFromZero);

    public float Plot(Graph graph, float startTime)  // 绘制图形
    {
        var time = 0f;  // 初始化时间
        for (; time <= Duration; time += graph.TimeIncrement)  // 循环计算每个时间点的高度并绘制
        {
            var height = _velocity * time + _acceleration * time * time / 2;  // 根据时间计算高度
            graph.Plot(startTime + time, height);  // 绘制图形
        }

        return startTime + time;  // 返回结束时间
    }

    public Bounce Next(float elasticity) => new Bounce(_velocity * elasticity);  // 计算下一次弹跳的速度
}

```
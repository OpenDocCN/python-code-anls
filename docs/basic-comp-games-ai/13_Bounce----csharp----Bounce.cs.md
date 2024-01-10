# `basic-computer-games\13_Bounce\csharp\Bounce.cs`

```
/// <summary>
/// 表示球的弹跳，计算持续时间、高度和时间内的位置。
/// </summary>
/// <remarks>
/// 所有计算都是从抛物线运动方程推导出来的：s = vt + 0.5at^2
/// </remarks>
internal class Bounce
{
    private const float _acceleration = -32; // feet/s^2  // 加速度常量，单位为英尺/秒^2

    private readonly float _velocity;  // 初始速度

    internal Bounce(float velocity)  // 构造函数，传入初始速度
    {
        _velocity = velocity;  // 初始化初始速度
    }

    public float Duration => -2 * _velocity / _acceleration;  // 计算弹跳持续时间

    public float MaxHeight =>
        (float)Math.Round(-_velocity * _velocity / 2 / _acceleration, MidpointRounding.AwayFromZero);  // 计算最大高度

    public float Plot(Graph graph, float startTime)  // 绘制弹跳轨迹
    {
        var time = 0f;  // 初始化时间
        for (; time <= Duration; time += graph.TimeIncrement)  // 循环计算每个时间点的高度
        {
            var height = _velocity * time + _acceleration * time * time / 2;  // 计算当前时间点的高度
            graph.Plot(startTime + time, height);  // 绘制当前时间点的位置
        }

        return startTime + time;  // 返回结束时间
    }

    public Bounce Next(float elasticity) => new Bounce(_velocity * elasticity);  // 计算下一次弹跳的速度
}
```
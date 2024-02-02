# `basic-computer-games\00_Common\dotnet\Games.Common\Randomness\RandomNumberGenerator.cs`

```py
using System;

namespace Games.Common.Randomness;

/// <inheritdoc />
// 实现了IRandom接口的RandomNumberGenerator类
public class RandomNumberGenerator : IRandom
{
    private Random _random; // 私有Random对象_random
    private float _previous; // 私有浮点型变量_previous

    public RandomNumberGenerator()
    {
        // 使用当前时间的秒数作为种子，初始化_random对象
        _random = new Random((int)(DateTime.UtcNow.Ticks / TimeSpan.TicksPerSecond));
    }

    // 生成下一个随机浮点数，并将其赋值给_previous
    public float NextFloat() => _previous = (float)_random.NextDouble();

    // 返回_previous的值
    public float PreviousFloat() => _previous;

    // 重新设置_random对象的种子
    public void Reseed(int seed) => _random = new Random(seed);
}
```
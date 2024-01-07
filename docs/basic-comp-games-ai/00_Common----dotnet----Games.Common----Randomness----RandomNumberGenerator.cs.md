# `basic-computer-games\00_Common\dotnet\Games.Common\Randomness\RandomNumberGenerator.cs`

```

// 引入 System 命名空间
using System;

// 声明 RandomNumberGenerator 类，实现 IRandom 接口
namespace Games.Common.Randomness;

/// <inheritdoc />
public class RandomNumberGenerator : IRandom
{
    private Random _random; // 声明 Random 对象
    private float _previous; // 声明 float 类型的变量 _previous

    // RandomNumberGenerator 类的构造函数
    public RandomNumberGenerator()
    {
        // 使用当前时间的秒数作为种子来初始化 Random 对象
        _random = new Random((int)(DateTime.UtcNow.Ticks / TimeSpan.TicksPerSecond));
    }

    // 生成下一个随机浮点数
    public float NextFloat() => _previous = (float)_random.NextDouble();

    // 返回上一个生成的随机浮点数
    public float PreviousFloat() => _previous;

    // 重新设置随机数生成器的种子
    public void Reseed(int seed) => _random = new Random(seed);
}

```
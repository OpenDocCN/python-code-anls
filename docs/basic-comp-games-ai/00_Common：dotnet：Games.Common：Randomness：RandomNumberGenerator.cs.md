# `d:/src/tocomm/basic-computer-games\00_Common\dotnet\Games.Common\Randomness\RandomNumberGenerator.cs`

```
using System;  # 导入 System 模块

namespace Games.Common.Randomness;  # 定义命名空间 Games.Common.Randomness

/// <inheritdoc />  # 继承父类的注释

public class RandomNumberGenerator : IRandom  # 定义 RandomNumberGenerator 类并实现 IRandom 接口
{
    private Random _random;  # 声明私有变量 _random，用于生成随机数
    private float _previous;  # 声明私有变量 _previous，用于存储上一个随机数值

    public RandomNumberGenerator()  # 定义 RandomNumberGenerator 类的构造函数
    {
        // The BASIC RNG is seeded based on time with a 1 second resolution  # 使用基本的随机数生成器，以1秒的分辨率基于时间进行种子生成
        _random = new Random((int)(DateTime.UtcNow.Ticks / TimeSpan.TicksPerSecond));  # 初始化 _random 变量，使用当前时间的 ticks 作为种子
    }

    public float NextFloat() => _previous = (float)_random.NextDouble();  # 定义 NextFloat 方法，返回下一个随机浮点数，并将其赋值给 _previous 变量

    public float PreviousFloat() => _previous;  # 定义 PreviousFloat 方法，返回上一个随机浮点数值
}
# 使用给定的种子值重新初始化随机数生成器
public void Reseed(int seed) => _random = new Random(seed);
# 参数：
# seed - 用于重新初始化随机数生成器的种子值
# 返回值：无
```
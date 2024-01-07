# `basic-computer-games\86_Target\csharp\RandomExtensions.cs`

```

# 导入 Games.Common.Randomness 包
using Games.Common.Randomness;

# 定义名为 Target 的命名空间
namespace Target
{
    # 定义名为 RandomExtensions 的静态类
    internal static class RandomExtensions
    {
        # 定义名为 NextPosition 的扩展方法，接收一个实现了 IRandom 接口的对象，并返回一个 Point 对象
        public static Point NextPosition(this IRandom rnd) => new (
            # 使用随机数生成器生成一个浮点数，并将其转换为角度
            Angle.InRotations(rnd.NextFloat()),
            # 使用随机数生成器生成一个浮点数，并将其转换为角度
            Angle.InRotations(rnd.NextFloat()),
            # 使用随机数生成器生成两个浮点数，计算它们的和并乘以 100000，然后返回结果
            100000 * rnd.NextFloat() + rnd.NextFloat());
    }
}

```
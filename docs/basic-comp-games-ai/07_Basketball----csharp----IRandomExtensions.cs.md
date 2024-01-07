# `basic-computer-games\07_Basketball\csharp\IRandomExtensions.cs`

```

# 使用 Games.Common.Randomness 命名空间中的随机数生成器
using Games.Common.Randomness;

# 声明 Basketball 命名空间
namespace Basketball;

# 声明一个内部的静态类 IRandomExtensions
internal static class IRandomExtensions
{
    # 创建一个扩展方法，用于生成下一个投篮动作
    internal static Shot NextShot(this IRandom random) => Shot.Get(random.NextFloat(1, 3.5f));
}

```
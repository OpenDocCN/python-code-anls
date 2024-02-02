# `basic-computer-games\07_Basketball\csharp\IRandomExtensions.cs`

```py
# 使用 Games.Common.Randomness 命名空间
using Games.Common.Randomness;

# 声明 Basketball 命名空间
namespace Basketball;

# 声明一个内部的静态类 IRandomExtensions
internal static class IRandomExtensions
{
    # 声明一个内部的静态方法 NextShot，接收一个 IRandom 类型的参数并返回 Shot 类型的对象
    internal static Shot NextShot(this IRandom random) => Shot.Get(random.NextFloat(1, 3.5f));
}
```
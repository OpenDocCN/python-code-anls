# `d:/src/tocomm/basic-computer-games\07_Basketball\csharp\IRandomExtensions.cs`

```
using Games.Common.Randomness;  # 导入 Games.Common.Randomness 命名空间，以便使用其中的类和方法

namespace Basketball;  # 声明 Basketball 命名空间，将下面的类放入该命名空间中

internal static class IRandomExtensions  # 声明一个内部的静态类 IRandomExtensions
{
    internal static Shot NextShot(this IRandom random) => Shot.Get(random.NextFloat(1, 3.5f));  # 声明一个内部的静态方法 NextShot，接受一个 IRandom 类型的参数 random，返回一个 Shot 类型的对象，通过调用 Shot 类的 Get 方法并传入 random.NextFloat(1, 3.5f) 的结果来创建 Shot 对象
}
```
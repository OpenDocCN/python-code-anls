# `d:/src/tocomm/basic-computer-games\86_Target\csharp\RandomExtensions.cs`

```
using Games.Common.Randomness;  // 导入 Games.Common.Randomness 命名空间，以便使用其中的类和接口

namespace Target  // 声明一个名为 Target 的命名空间
{
    internal static class RandomExtensions  // 声明一个名为 RandomExtensions 的静态类
    {
        public static Point NextPosition(this IRandom rnd) => new (  // 声明一个名为 NextPosition 的扩展方法，接收一个实现了 IRandom 接口的对象作为参数，并返回一个 Point 对象
            Angle.InRotations(rnd.NextFloat()),  // 调用 Angle.InRotations 方法，传入 rnd.NextFloat() 的返回值作为参数
            Angle.InRotations(rnd.NextFloat()),  // 调用 Angle.InRotations 方法，传入 rnd.NextFloat() 的返回值作为参数
            100000 * rnd.NextFloat() + rnd.NextFloat());  // 计算 100000 * rnd.NextFloat() + rnd.NextFloat() 的值，并作为参数创建一个新的 Point 对象
    }
}
```
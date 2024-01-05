# `30_Cube\csharp\ZerosGenerator.cs`

```
namespace Cube;  // 命名空间声明，定义了代码所在的命名空间

internal class ZerosGenerator : IRandom  // 定义了一个内部类 ZerosGenerator，实现了接口 IRandom
{
    public float NextFloat() => 0;  // 实现了接口方法 NextFloat，返回值为 0

    public float PreviousFloat() => 0;  // 实现了接口方法 PreviousFloat，返回值为 0

    public void Reseed(int seed) { }  // 实现了接口方法 Reseed，但是方法体为空
}
```
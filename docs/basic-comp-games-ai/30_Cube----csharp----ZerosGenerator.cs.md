# `basic-computer-games\30_Cube\csharp\ZerosGenerator.cs`

```
# 在 Cube 命名空间下定义一个内部类 ZerosGenerator，实现了 IRandom 接口
internal class ZerosGenerator : IRandom
{
    # 实现接口方法，返回固定值 0
    public float NextFloat() => 0;

    # 实现接口方法，返回固定值 0
    public float PreviousFloat() => 0;

    # 实现接口方法，重新设置种子，但不做任何操作
    public void Reseed(int seed) { }
}
```
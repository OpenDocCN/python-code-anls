# `basic-computer-games\77_Salvo\csharp\Ships\Destroyer.cs`

```
# 在 Salvo.Ships 命名空间下定义 Destroyer 类，继承自 Ship 类
internal sealed class Destroyer : Ship
{
    # Destroyer 类的构造函数，接受 nameIndex 和 io 作为参数，调用基类 Ship 的构造函数
    internal Destroyer(string nameIndex, IReadWrite io)
        : base(io, $"<{nameIndex}>")
    {
    }

    # Destroyer 类的构造函数，接受 nameIndex 和 random 作为参数，调用基类 Ship 的构造函数
    internal Destroyer(string nameIndex, IRandom random)
        : base(random, $"<{nameIndex}>")
    {
    }

    # 重写基类的 Shots 属性，返回值为 1
    internal override int Shots => 1;
    # 重写基类的 Size 属性，返回值为 2
    internal override int Size => 2;
}
```
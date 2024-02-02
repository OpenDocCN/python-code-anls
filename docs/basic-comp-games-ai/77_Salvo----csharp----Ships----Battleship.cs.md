# `basic-computer-games\77_Salvo\csharp\Ships\Battleship.cs`

```py
# 在 Salvo.Ships 命名空间下定义一个内部密封类 Battleship，继承自 Ship 类
internal sealed class Battleship : Ship
{
    # Battleship 类的构造函数，接受一个 IReadWrite 接口类型的参数 io
    internal Battleship(IReadWrite io) 
        # 调用基类 Ship 的构造函数，传入参数 io
        : base(io) 
    { 
    }

    # Battleship 类的构造函数，接受一个 IRandom 接口类型的参数 random
    internal Battleship(IRandom random)
        # 调用基类 Ship 的构造函数，传入参数 random
        : base(random)
    {
    }

    # 重写基类的 Shots 属性，返回值为 3
    internal override int Shots => 3;
    # 重写基类的 Size 属性，返回值为 5
    internal override int Size => 5;
}
```
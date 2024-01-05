# `77_Salvo\csharp\Ships\Battleship.cs`

```
namespace Salvo.Ships;  // 声明命名空间 Salvo.Ships

internal sealed class Battleship : Ship  // 声明一个内部密封类 Battleship，继承自 Ship 类
{
    internal Battleship(IReadWrite io)  // 声明一个内部方法 Battleship，接受一个 IReadWrite 类型的参数 io
        : base(io)  // 调用基类 Ship 的构造函数，传入参数 io
    { 
    }

    internal Battleship(IRandom random)  // 声明一个内部方法 Battleship，接受一个 IRandom 类型的参数 random
        : base(random)  // 调用基类 Ship 的构造函数，传入参数 random
    {
    }

    internal override int Shots => 3;  // 声明一个内部方法 Shots，返回值为整数类型，重写基类的 Shots 属性，返回值为 3
    internal override int Size => 5;  // 声明一个内部方法 Size，返回值为整数类型，重写基类的 Size 属性，返回值为 5
}
```
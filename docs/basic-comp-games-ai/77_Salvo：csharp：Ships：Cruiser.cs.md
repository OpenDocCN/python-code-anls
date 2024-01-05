# `d:/src/tocomm/basic-computer-games\77_Salvo\csharp\Ships\Cruiser.cs`

```
namespace Salvo.Ships;  // 声明命名空间 Salvo.Ships

internal sealed class Cruiser : Ship  // 声明一个名为 Cruiser 的类，继承自 Ship 类，并且是内部类
{
    internal Cruiser(IReadWrite io)  // 声明一个接受 IReadWrite 类型参数的构造函数
        : base(io)  // 调用基类 Ship 的构造函数，传入参数 io
    { 
    }
    
    internal Cruiser(IRandom random)  // 声明一个接受 IRandom 类型参数的构造函数
        : base(random)  // 调用基类 Ship 的构造函数，传入参数 random
    {
    }

    internal override int Shots => 2;  // 声明 Shots 属性，返回值为 2
    internal override int Size => 3;  // 声明 Size 属性，返回值为 3
}
```
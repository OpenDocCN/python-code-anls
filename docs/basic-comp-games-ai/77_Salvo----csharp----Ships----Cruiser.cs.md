# `basic-computer-games\77_Salvo\csharp\Ships\Cruiser.cs`

```
# 在 Salvo.Ships 命名空间下定义一个内部密封类 Cruiser，继承自 Ship 类
internal sealed class Cruiser : Ship
{
    # Cruiser 类的构造函数，接受一个 IReadWrite 接口类型的参数 io，并调用基类 Ship 的构造函数
    internal Cruiser(IReadWrite io) 
        : base(io) 
    { 
    }
    
    # Cruiser 类的构造函数，接受一个 IRandom 接口类型的参数 random，并调用基类 Ship 的构造函数
    internal Cruiser(IRandom random)
        : base(random)
    {
    }

    # 重写基类的 Shots 属性，返回值为 2
    internal override int Shots => 2;
    
    # 重写基类的 Size 属性，返回值为 3
    internal override int Size => 3;
}
```
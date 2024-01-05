# `d:/src/tocomm/basic-computer-games\77_Salvo\csharp\Ships\Destroyer.cs`

```
namespace Salvo.Ships;  // 命名空间声明，表示该类属于Salvo.Ships命名空间

internal sealed class Destroyer : Ship  // 声明一个名为Destroyer的类，它是Ship类的子类，并且只能在当前程序集内部访问
{
    internal Destroyer(string nameIndex, IReadWrite io)  // Destroyer类的构造函数，接受nameIndex和IReadWrite类型的参数
        : base(io, $"<{nameIndex}>")  // 调用基类Ship的构造函数，传入io和格式化后的nameIndex参数
    {
    }

    internal Destroyer(string nameIndex, IRandom random)  // Destroyer类的另一个构造函数，接受nameIndex和IRandom类型的参数
        : base(random, $"<{nameIndex}>")  // 调用基类Ship的构造函数，传入random和格式化后的nameIndex参数
    {
    }

    internal override int Shots => 1;  // 重写基类Ship的Shots属性，返回值为1
    internal override int Size => 2;  // 重写基类Ship的Size属性，返回值为2
}
```
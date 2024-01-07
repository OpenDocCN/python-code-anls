# `basic-computer-games\84_Super_Star_Trek\csharp\Systems\ComputerFunctions\ComputerFunction.cs`

```

# 导入所需的命名空间
using Games.Common.IO;
using SuperStarTrek.Space;

# 定义一个抽象类 ComputerFunction
namespace SuperStarTrek.Systems.ComputerFunctions;

internal abstract class ComputerFunction
{
    # 定义构造函数，接受描述和 IReadWrite 接口实例作为参数
    protected ComputerFunction(string description, IReadWrite io)
    {
        Description = description;
        IO = io;
    }

    # 定义描述属性
    internal string Description { get; }

    # 定义 IReadWrite 接口实例属性
    protected IReadWrite IO { get; }

    # 定义抽象方法，接受 Quadrant 对象作为参数
    internal abstract void Execute(Quadrant quadrant);
}

```
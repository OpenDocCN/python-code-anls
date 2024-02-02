# `basic-computer-games\84_Super_Star_Trek\csharp\Systems\ComputerFunctions\ComputerFunction.cs`

```py
# 导入 Games.Common.IO 和 SuperStarTrek.Space 模块
using Games.Common.IO;
using SuperStarTrek.Space;

# 定义 SuperStarTrek.Systems.ComputerFunctions 命名空间下的抽象类 ComputerFunction
internal abstract class ComputerFunction
{
    # 定义受保护的构造函数，接受描述和 IReadWrite 接口实例作为参数
    protected ComputerFunction(string description, IReadWrite io)
    {
        # 将描述赋给 Description 属性
        Description = description;
        # 将 IReadWrite 实例赋给 IO 属性
        IO = io;
    }

    # 定义只读的 Description 属性
    internal string Description { get; }

    # 定义受保护的 IO 属性，类型为 IReadWrite 接口
    protected IReadWrite IO { get; }

    # 定义抽象的 Execute 方法，接受 Quadrant 对象作为参数
    internal abstract void Execute(Quadrant quadrant);
}
```
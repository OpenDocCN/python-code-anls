# `d:/src/tocomm/basic-computer-games\84_Super_Star_Trek\csharp\Systems\ComputerFunctions\ComputerFunction.cs`

```
# 导入 Games.Common.IO 和 SuperStarTrek.Space 模块
using Games.Common.IO;
using SuperStarTrek.Space;

# 定义 SuperStarTrek.Systems.ComputerFunctions 命名空间下的抽象类 ComputerFunction
namespace SuperStarTrek.Systems.ComputerFunctions;

internal abstract class ComputerFunction
{
    # 定义抽象类的构造函数，接受描述和 IReadWrite 对象作为参数
    protected ComputerFunction(string description, IReadWrite io)
    {
        Description = description;  # 设置描述属性
        IO = io;  # 设置 IReadWrite 对象属性
    }

    internal string Description { get; }  # 描述属性的 getter 方法

    protected IReadWrite IO { get; }  # IReadWrite 对象属性的 getter 方法

    internal abstract void Execute(Quadrant quadrant);  # 抽象方法，用于执行特定的计算机功能
}
```
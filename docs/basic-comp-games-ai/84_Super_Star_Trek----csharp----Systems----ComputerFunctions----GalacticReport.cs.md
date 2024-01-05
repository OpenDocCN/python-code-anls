# `84_Super_Star_Trek\csharp\Systems\ComputerFunctions\GalacticReport.cs`

```
using System.Collections.Generic; // 导入 System.Collections.Generic 命名空间，用于使用泛型集合类
using System.Linq; // 导入 System.Linq 命名空间，用于使用 LINQ 查询
using Games.Common.IO; // 导入 Games.Common.IO 命名空间，用于使用游戏通用的输入输出功能
using SuperStarTrek.Space; // 导入 SuperStarTrek.Space 命名空间，用于使用星际飞船相关的功能

namespace SuperStarTrek.Systems.ComputerFunctions; // 定义 SuperStarTrek.Systems.ComputerFunctions 命名空间
internal abstract class GalacticReport : ComputerFunction // 定义 GalacticReport 抽象类，继承自 ComputerFunction 类
{
    internal GalacticReport(string description, IReadWrite io, Galaxy galaxy) // 定义 GalacticReport 类的构造函数，接受描述、输入输出对象和星系对象作为参数
        : base(description, io) // 调用基类的构造函数，传入描述和输入输出对象
    {
        Galaxy = galaxy; // 将传入的星系对象赋值给类的 Galaxy 属性
    }

    protected Galaxy Galaxy { get; } // 定义受保护的 Galaxy 属性，用于存储星系对象

    protected abstract void WriteHeader(Quadrant quadrant); // 定义受保护的抽象方法 WriteHeader，用于输出表头信息，接受象限对象作为参数

    protected abstract IEnumerable<string> GetRowData(); // 定义受保护的抽象方法 GetRowData，用于获取行数据，返回字符串集合
}
    # 重写父类的 Execute 方法，接受一个 Quadrant 参数
    internal sealed override void Execute(Quadrant quadrant)
    {
        # 调用 WriteHeader 方法，传入 quadrant 参数
        WriteHeader(quadrant);
        # 输出表头
        IO.WriteLine("       1     2     3     4     5     6     7     8");
        IO.WriteLine("     ----- ----- ----- ----- ----- ----- ----- -----");

        # 遍历 GetRowData() 方法返回的数据，同时获取行数据和索引
        foreach (var (row, index) in GetRowData().Select((r, i) => (r, i)))
        {
            # 输出行号和行数据
            IO.WriteLine($" {index+1}   {row}");
            IO.WriteLine("     ----- ----- ----- ----- ----- ----- ----- -----");
        }
    }
}
```
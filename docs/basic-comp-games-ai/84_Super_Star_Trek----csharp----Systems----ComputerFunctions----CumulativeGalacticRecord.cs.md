# `84_Super_Star_Trek\csharp\Systems\ComputerFunctions\CumulativeGalacticRecord.cs`

```
using System.Collections.Generic; // 导入 System.Collections.Generic 命名空间，用于使用泛型集合类
using System.Linq; // 导入 System.Linq 命名空间，用于使用 LINQ 查询
using Games.Common.IO; // 导入 Games.Common.IO 命名空间，用于使用游戏通用的输入输出功能
using SuperStarTrek.Space; // 导入 SuperStarTrek.Space 命名空间，用于使用游戏中的太空相关功能

namespace SuperStarTrek.Systems.ComputerFunctions; // 定义 SuperStarTrek.Systems.ComputerFunctions 命名空间

internal class CumulativeGalacticRecord : GalacticReport // 定义 CumulativeGalacticRecord 类，继承自 GalacticReport 类
{
    internal CumulativeGalacticRecord(IReadWrite io, Galaxy galaxy) // 定义 CumulativeGalacticRecord 类的构造函数，接受 IReadWrite 和 Galaxy 类型的参数
        : base("Cumulative galactic record", io, galaxy) // 调用基类 GalacticReport 的构造函数，传入指定的参数
    {
    }

    protected override void WriteHeader(Quadrant quadrant) // 重写基类 GalacticReport 的 WriteHeader 方法，接受 Quadrant 类型的参数
    {
        IO.WriteLine(); // 调用 IO 对象的 WriteLine 方法，输出空行
        IO.WriteLine($"Computer record of galaxy for quadrant {quadrant.Coordinates}"); // 调用 IO 对象的 WriteLine 方法，输出指定格式的字符串
        IO.WriteLine(); // 调用 IO 对象的 WriteLine 方法，输出空行
    }
}
    # 重写父类方法，获取行数据
    protected override IEnumerable<string> GetRowData() =>
        # 从Galaxy.Quadrants中选择每一行数据，并用空格连接成字符串
        Galaxy.Quadrants.Select(row => " " + string.Join("   ", row));
}
```

这段代码是C#语言的代码，重写了父类的方法GetRowData()，使用Galaxy.Quadrants中的数据生成行数据。在这段代码中，使用了Lambda表达式和LINQ语句。
```
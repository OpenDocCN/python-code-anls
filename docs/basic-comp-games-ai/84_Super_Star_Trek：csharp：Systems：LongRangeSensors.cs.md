# `d:/src/tocomm/basic-computer-games\84_Super_Star_Trek\csharp\Systems\LongRangeSensors.cs`

```
using System.Linq; // 导入 System.Linq 命名空间，用于使用 LINQ 查询
using Games.Common.IO; // 导入 Games.Common.IO 命名空间，用于处理输入输出
using SuperStarTrek.Commands; // 导入 SuperStarTrek.Commands 命名空间，用于处理游戏命令
using SuperStarTrek.Space; // 导入 SuperStarTrek.Space 命名空间，用于处理游戏空间相关的操作

namespace SuperStarTrek.Systems; // 声明 SuperStarTrek.Systems 命名空间

internal class LongRangeSensors : Subsystem // 声明 LongRangeSensors 类，继承自 Subsystem 类
{
    private readonly Galaxy _galaxy; // 声明私有成员变量 _galaxy，用于存储 Galaxy 对象
    private readonly IReadWrite _io; // 声明私有成员变量 _io，用于存储 IReadWrite 对象

    internal LongRangeSensors(Galaxy galaxy, IReadWrite io) // 声明 LongRangeSensors 类的构造函数，接受 Galaxy 和 IReadWrite 对象作为参数
        : base("Long Range Sensors", Command.LRS, io) // 调用父类 Subsystem 的构造函数，传入指定的参数
    {
        _galaxy = galaxy; // 将传入的 galaxy 参数赋值给 _galaxy 成员变量
        _io = io; // 将传入的 io 参数赋值给 _io 成员变量
    }

    protected override bool CanExecuteCommand() => IsOperational("{name} are inoperable"); // 重写父类的 CanExecuteCommand 方法，返回 IsOperational 方法的结果
}
    // 重写执行命令核心方法，接收一个象限参数
    protected override CommandResult ExecuteCommandCore(Quadrant quadrant)
    {
        // 输出长距离扫描的象限坐标
        _io.WriteLine($"Long range scan for quadrant {quadrant.Coordinates}");
        // 输出分隔线
        _io.WriteLine("-------------------");
        // 遍历获取到的象限邻居
        foreach (var quadrants in _galaxy.GetNeighborhood(quadrant))
        {
            // 输出每个邻居象限的扫描结果，如果为空则输出 ***
            _io.WriteLine(": " + string.Join(" : ", quadrants.Select(q => q?.Scan() ?? "***")) + " :");
            // 输出分隔线
            _io.WriteLine("-------------------");
        }

        // 返回命令执行结果为成功
        return CommandResult.Ok;
    }
```
# `basic-computer-games\84_Super_Star_Trek\csharp\Systems\LongRangeSensors.cs`

```

using System.Linq; // 导入 LINQ 扩展方法
using Games.Common.IO; // 导入通用游戏输入输出命名空间
using SuperStarTrek.Commands; // 导入超级星际远航命令类
using SuperStarTrek.Space; // 导入超级星际远航空间类

namespace SuperStarTrek.Systems; // 定义超级星际远航系统命名空间

internal class LongRangeSensors : Subsystem // 定义长程传感器子系统类，继承自 Subsystem 类
{
    private readonly Galaxy _galaxy; // 声明私有 Galaxy 类型的字段 _galaxy
    private readonly IReadWrite _io; // 声明私有 IReadWrite 类型的字段 _io

    internal LongRangeSensors(Galaxy galaxy, IReadWrite io) // 定义 LongRangeSensors 类的构造函数，接受 Galaxy 和 IReadWrite 类型的参数
        : base("Long Range Sensors", Command.LRS, io) // 调用基类 Subsystem 的构造函数，传入子系统名称、命令和 io 参数
    {
        _galaxy = galaxy; // 将传入的 galaxy 参数赋值给 _galaxy 字段
        _io = io; // 将传入的 io 参数赋值给 _io 字段
    }

    protected override bool CanExecuteCommand() => IsOperational("{name} are inoperable"); // 重写基类的 CanExecuteCommand 方法，判断子系统是否可执行命令

    protected override CommandResult ExecuteCommandCore(Quadrant quadrant) // 重写基类的 ExecuteCommandCore 方法，执行子系统核心命令
    {
        _io.WriteLine($"Long range scan for quadrant {quadrant.Coordinates}"); // 输出长程扫描的象限坐标
        _io.WriteLine("-------------------"); // 输出分隔线
        foreach (var quadrants in _galaxy.GetNeighborhood(quadrant)) // 遍历获取指定象限的邻近象限
        {
            _io.WriteLine(": " + string.Join(" : ", quadrants.Select(q => q?.Scan() ?? "***")) + " :"); // 输出邻近象限的扫描结果
            _io.WriteLine("-------------------"); // 输出分隔线
        }

        return CommandResult.Ok; // 返回命令执行结果为成功
    }
}

```
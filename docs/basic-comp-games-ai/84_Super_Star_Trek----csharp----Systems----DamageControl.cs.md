# `basic-computer-games\84_Super_Star_Trek\csharp\Systems\DamageControl.cs`

```
using Games.Common.IO;  // 导入 Games.Common.IO 命名空间
using SuperStarTrek.Commands;  // 导入 SuperStarTrek.Commands 命名空间
using SuperStarTrek.Objects;  // 导入 SuperStarTrek.Objects 命名空间
using SuperStarTrek.Space;  // 导入 SuperStarTrek.Space 命名空间

namespace SuperStarTrek.Systems;  // 定义 SuperStarTrek.Systems 命名空间

internal class DamageControl : Subsystem  // 定义 DamageControl 类，继承自 Subsystem 类
{
    private readonly Enterprise _enterprise;  // 声明私有的 Enterprise 类型字段 _enterprise
    private readonly IReadWrite _io;  // 声明私有的 IReadWrite 类型字段 _io

    internal DamageControl(Enterprise enterprise, IReadWrite io)  // DamageControl 类的构造函数，接受 Enterprise 和 IReadWrite 类型的参数
        : base("Damage Control", Command.DAM, io)  // 调用基类 Subsystem 的构造函数，传入字符串 "Damage Control"、Command.DAM 和 io
    {
        _enterprise = enterprise;  // 将参数 enterprise 的值赋给 _enterprise 字段
        _io = io;  // 将参数 io 的值赋给 _io 字段
    }

    protected override CommandResult ExecuteCommandCore(Quadrant quadrant)  // 重写基类 Subsystem 的 ExecuteCommandCore 方法，接受 Quadrant 类型的参数 quadrant
    {
        if (IsDamaged)  // 如果当前子系统已经受损
        {
            _io.WriteLine("Damage Control report not available");  // 输出信息："Damage Control report not available"
        }
        else  // 如果当前子系统未受损
        {
            _io.WriteLine();  // 输出空行
            WriteDamageReport();  // 调用 WriteDamageReport 方法
        }

        if (_enterprise.DamagedSystemCount > 0 && _enterprise.IsDocked)  // 如果 Enterprise 的受损系统数量大于 0 且已停靠
        {
            if (quadrant.Starbase.TryRepair(_enterprise, out var repairTime))  // 尝试修复 Starbase，将修复时间保存在 repairTime 变量中
            {
                WriteDamageReport();  // 调用 WriteDamageReport 方法
                return CommandResult.Elapsed(repairTime);  // 返回修复时间
            }
        }

        return CommandResult.Ok;  // 返回 CommandResult.Ok
    }

    internal void WriteDamageReport()  // 定义 WriteDamageReport 方法
    {
        _io.WriteLine();  // 输出空行
        _io.WriteLine("Device             State of Repair");  // 输出信息："Device             State of Repair"
        foreach (var system in _enterprise.Systems)  // 遍历 Enterprise 的 Systems
        {
            _io.Write(system.Name.PadRight(25));  // 输出系统名称并右对齐，总长度为 25
            _io.WriteLine((int)(system.Condition * 100) * 0.01F);  // 输出系统修复状态
        }
        _io.WriteLine();  // 输出空行
    }
}
```
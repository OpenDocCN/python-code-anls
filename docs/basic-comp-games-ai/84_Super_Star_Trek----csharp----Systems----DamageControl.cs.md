# `basic-computer-games\84_Super_Star_Trek\csharp\Systems\DamageControl.cs`

```

// 引入所需的命名空间
using Games.Common.IO;
using SuperStarTrek.Commands;
using SuperStarTrek.Objects;
using SuperStarTrek.Space;

// 定义 DamageControl 类，继承自 Subsystem 类
namespace SuperStarTrek.Systems;
internal class DamageControl : Subsystem
{
    // 声明私有变量 _enterprise 和 _io
    private readonly Enterprise _enterprise;
    private readonly IReadWrite _io;

    // DamageControl 类的构造函数，初始化 _enterprise 和 _io
    internal DamageControl(Enterprise enterprise, IReadWrite io)
        : base("Damage Control", Command.DAM, io)
    {
        _enterprise = enterprise;
        _io = io;
    }

    // 重写 ExecuteCommandCore 方法
    protected override CommandResult ExecuteCommandCore(Quadrant quadrant)
    {
        // 如果系统受损，输出提示信息
        if (IsDamaged)
        {
            _io.WriteLine("Damage Control report not available");
        }
        // 否则输出空行，并调用 WriteDamageReport 方法
        else
        {
            _io.WriteLine();
            WriteDamageReport();
        }

        // 如果 Enterprise 受损系统数量大于 0 且已对接
        if (_enterprise.DamagedSystemCount > 0 && _enterprise.IsDocked)
        {
            // 尝试修复受损系统，并返回修复时间
            if (quadrant.Starbase.TryRepair(_enterprise, out var repairTime))
            {
                // 调用 WriteDamageReport 方法，并返回修复时间
                WriteDamageReport();
                return CommandResult.Elapsed(repairTime);
            }
        }

        // 返回命令执行结果为正常
        return CommandResult.Ok;
    }

    // 定义 WriteDamageReport 方法
    internal void WriteDamageReport()
    {
        // 输出空行
        _io.WriteLine();
        // 输出设备状态报告表头
        _io.WriteLine("Device             State of Repair");
        // 遍历 Enterprise 的系统列表，输出系统名称和修复状态
        foreach (var system in _enterprise.Systems)
        {
            _io.Write(system.Name.PadRight(25));
            _io.WriteLine((int)(system.Condition * 100) * 0.01F);
        }
        // 输出空行
        _io.WriteLine();
    }
}

```
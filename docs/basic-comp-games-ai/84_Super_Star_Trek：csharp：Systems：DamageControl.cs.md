# `d:/src/tocomm/basic-computer-games\84_Super_Star_Trek\csharp\Systems\DamageControl.cs`

```
using Games.Common.IO;  # 导入 Games.Common.IO 模块
using SuperStarTrek.Commands;  # 导入 SuperStarTrek.Commands 模块
using SuperStarTrek.Objects;  # 导入 SuperStarTrek.Objects 模块
using SuperStarTrek.Space;  # 导入 SuperStarTrek.Space 模块

namespace SuperStarTrek.Systems;  # 定义 SuperStarTrek.Systems 命名空间

internal class DamageControl : Subsystem  # 定义 DamageControl 类，继承自 Subsystem 类
{
    private readonly Enterprise _enterprise;  # 声明私有属性 _enterprise，类型为 Enterprise
    private readonly IReadWrite _io;  # 声明私有属性 _io，类型为 IReadWrite

    internal DamageControl(Enterprise enterprise, IReadWrite io)  # 定义 DamageControl 类的构造函数，接受 Enterprise 和 IReadWrite 类型的参数
        : base("Damage Control", Command.DAM, io)  # 调用父类 Subsystem 的构造函数，传入字符串 "Damage Control"、Command.DAM 和 io 参数
    {
        _enterprise = enterprise;  # 将传入的 enterprise 参数赋值给 _enterprise 属性
        _io = io;  # 将传入的 io 参数赋值给 _io 属性
    }

    protected override CommandResult ExecuteCommandCore(Quadrant quadrant)  # 重写父类 Subsystem 的 ExecuteCommandCore 方法，接受 Quadrant 类型的参数
        # 如果飞船受损，则输出“Damage Control report not available”
        if (IsDamaged)
        {
            _io.WriteLine("Damage Control report not available");
        }
        # 如果飞船没有受损
        else
        {
            # 输出空行
            _io.WriteLine();
            # 调用WriteDamageReport()方法，输出损坏报告
            WriteDamageReport();
        }

        # 如果企业号受损系统数量大于0且正在停靠
        if (_enterprise.DamagedSystemCount > 0 && _enterprise.IsDocked)
        {
            # 如果象限星站尝试修复企业号，并返回修复时间
            if (quadrant.Starbase.TryRepair(_enterprise, out var repairTime))
            {
                # 输出损坏报告
                WriteDamageReport();
                # 返回修复时间
                return CommandResult.Elapsed(repairTime);
            }
        }
        return CommandResult.Ok;  # 返回一个表示命令执行成功的结果

    }

    internal void WriteDamageReport()  # 内部方法，用于输出损坏报告
    {
        _io.WriteLine();  # 输出空行
        _io.WriteLine("Device             State of Repair");  # 输出标题行
        foreach (var system in _enterprise.Systems)  # 遍历企业系统列表
        {
            _io.Write(system.Name.PadRight(25));  # 输出系统名称并右对齐到25个字符
            _io.WriteLine((int)(system.Condition * 100) * 0.01F);  # 输出系统状态的修复程度
        }
        _io.WriteLine();  # 输出空行
    }
}
```
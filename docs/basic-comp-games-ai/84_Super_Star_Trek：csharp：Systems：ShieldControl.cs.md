# `d:/src/tocomm/basic-computer-games\84_Super_Star_Trek\csharp\Systems\ShieldControl.cs`

```
using Games.Common.IO;  # 导入 Games.Common.IO 模块
using SuperStarTrek.Commands;  # 导入 SuperStarTrek.Commands 模块
using SuperStarTrek.Objects;  # 导入 SuperStarTrek.Objects 模块
using SuperStarTrek.Resources;  # 导入 SuperStarTrek.Resources 模块
using SuperStarTrek.Space;  # 导入 SuperStarTrek.Space 模块

namespace SuperStarTrek.Systems;  # 定义 SuperStarTrek.Systems 命名空间

internal class ShieldControl : Subsystem  # 定义 ShieldControl 类，继承自 Subsystem 类
{
    private readonly Enterprise _enterprise;  # 声明私有属性 _enterprise，类型为 Enterprise
    private readonly IReadWrite _io;  # 声明私有属性 _io，类型为 IReadWrite

    internal ShieldControl(Enterprise enterprise, IReadWrite io)  # 定义 ShieldControl 类的构造函数，接受 Enterprise 和 IReadWrite 类型的参数
        : base("Shield Control", Command.SHE, io)  # 调用父类 Subsystem 的构造函数，传入字符串 "Shield Control"、Command.SHE 和 io 参数
    {
        _enterprise = enterprise;  # 将传入的 enterprise 参数赋值给 _enterprise 属性
        _io = io;  # 将传入的 io 参数赋值给 _io 属性
    }
    internal float ShieldEnergy { get; set; }  # 定义一个内部的浮点型属性 ShieldEnergy，用于存储护盾能量值

    protected override bool CanExecuteCommand() => IsOperational("{name} inoperable");  # 重写 CanExecuteCommand 方法，判断是否可以执行命令，调用 IsOperational 方法来判断是否可操作

    protected override CommandResult ExecuteCommandCore(Quadrant quadrant)  # 重写 ExecuteCommandCore 方法，执行核心命令
    {
        _io.WriteLine($"Energy available = {_enterprise.TotalEnergy}");  # 在控制台输出可用能量值
        var requested = _io.ReadNumber($"Number of units to shields");  # 从控制台读取用户输入的护盾能量值

        if (Validate(requested))  # 调用 Validate 方法验证用户输入的护盾能量值是否有效
        {
            ShieldEnergy = requested;  # 将用户输入的护盾能量值赋给 ShieldEnergy 属性
            _io.Write(Strings.ShieldsSet, requested);  # 在控制台输出设置了多少能量值到护盾
        }
        else
        {
            _io.WriteLine("<SHIELDS UNCHANGED>");  # 在控制台输出护盾未改变的提示
        }

        return CommandResult.Ok;  # 返回命令执行结果为成功
    }
    }  # 结束 Validate 方法的定义

    private bool Validate(float requested)
    {
        if (requested > _enterprise.TotalEnergy)  # 如果请求的能量大于企业的总能量
        {
            _io.WriteLine("Shield Control reports, 'This is not the Federation Treasury.'");  # 输出错误信息
            return false  # 返回 false
        }

        return requested >= 0 && requested != ShieldEnergy  # 返回请求的能量大于等于0且不等于当前护盾能量
    }

    internal void AbsorbHit(int hitStrength) => ShieldEnergy -= hitStrength;  # 护盾吸收攻击，减去攻击力

    internal void DropShields() => ShieldEnergy = 0;  # 关闭护盾，护盾能量设为0
}
```
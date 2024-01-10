# `basic-computer-games\84_Super_Star_Trek\csharp\Systems\ShieldControl.cs`

```
// 引入所需的命名空间
using Games.Common.IO;
using SuperStarTrek.Commands;
using SuperStarTrek.Objects;
using SuperStarTrek.Resources;
using SuperStarTrek.Space;

// 定义 ShieldControl 类，继承自 Subsystem 类
namespace SuperStarTrek.Systems
{
    internal class ShieldControl : Subsystem
    {
        // 声明私有字段 _enterprise 和 _io
        private readonly Enterprise _enterprise;
        private readonly IReadWrite _io;

        // ShieldControl 类的构造函数，接受 Enterprise 和 IReadWrite 对象作为参数
        internal ShieldControl(Enterprise enterprise, IReadWrite io)
            : base("Shield Control", Command.SHE, io)
        {
            _enterprise = enterprise;
            _io = io;
        }

        // 声明 ShieldEnergy 属性，用于设置和获取护盾能量值
        internal float ShieldEnergy { get; set; }

        // 重写父类的 CanExecuteCommand 方法，判断是否可以执行命令
        protected override bool CanExecuteCommand() => IsOperational("{name} inoperable");

        // 重写父类的 ExecuteCommandCore 方法，执行核心命令逻辑
        protected override CommandResult ExecuteCommandCore(Quadrant quadrant)
        {
            // 输出可用能量值
            _io.WriteLine($"Energy available = {_enterprise.TotalEnergy}");
            // 从用户输入读取护盾能量值
            var requested = _io.ReadNumber($"Number of units to shields");

            // 验证用户输入的护盾能量值，并作出相应处理
            if (Validate(requested))
            {
                ShieldEnergy = requested;
                _io.Write(Strings.ShieldsSet, requested);
            }
            else
            {
                _io.WriteLine("<SHIELDS UNCHANGED>");
            }

            return CommandResult.Ok;
        }

        // 验证用户输入的护盾能量值是否有效
        private bool Validate(float requested)
        {
            if (requested > _enterprise.TotalEnergy)
            {
                _io.WriteLine("Shield Control reports, 'This is not the Federation Treasury.'");
                return false;
            }

            return requested >= 0 && requested != ShieldEnergy;
        }

        // 用于处理护盾吸收攻击的方法
        internal void AbsorbHit(int hitStrength) => ShieldEnergy -= hitStrength;

        // 用于关闭护盾的方法
        internal void DropShields() => ShieldEnergy = 0;
    }
}
```
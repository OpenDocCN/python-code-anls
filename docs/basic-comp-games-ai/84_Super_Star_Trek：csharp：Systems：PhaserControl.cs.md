# `d:/src/tocomm/basic-computer-games\84_Super_Star_Trek\csharp\Systems\PhaserControl.cs`

```
using System.Linq;  // 导入 System.Linq 命名空间，用于 LINQ 查询
using Games.Common.IO;  // 导入 Games.Common.IO 命名空间
using Games.Common.Randomness;  // 导入 Games.Common.Randomness 命名空间
using SuperStarTrek.Commands;  // 导入 SuperStarTrek.Commands 命名空间
using SuperStarTrek.Objects;  // 导入 SuperStarTrek.Objects 命名空间
using SuperStarTrek.Resources;  // 导入 SuperStarTrek.Resources 命名空间
using SuperStarTrek.Space;  // 导入 SuperStarTrek.Space 命名空间

namespace SuperStarTrek.Systems  // 声明 SuperStarTrek.Systems 命名空间
{
    internal class PhaserControl : Subsystem  // 声明 PhaserControl 类，继承自 Subsystem 类
    {
        private readonly Enterprise _enterprise;  // 声明私有字段 _enterprise，存储 Enterprise 对象
        private readonly IReadWrite _io;  // 声明私有字段 _io，存储 IReadWrite 对象
        private readonly IRandom _random;  // 声明私有字段 _random，存储 IRandom 对象

        internal PhaserControl(Enterprise enterprise, IReadWrite io, IRandom random)  // 声明 PhaserControl 类的构造函数，接受 Enterprise、IReadWrite 和 IRandom 对象作为参数
            : base("Phaser Control", Command.PHA, io)  // 调用基类 Subsystem 的构造函数，传入字符串 "Phaser Control"、Command.PHA 和 io 对象
        {
            _enterprise = enterprise;  // 将参数 enterprise 赋值给 _enterprise 字段
        _io = io;  # 将io模块赋值给_io变量
        _random = random;  # 将random模块赋值给_random变量
    }

    protected override bool CanExecuteCommand() => IsOperational("Phasers inoperative");  # 重写CanExecuteCommand方法，判断是否可以执行命令，根据"Phasers inoperative"来判断

    protected override CommandResult ExecuteCommandCore(Quadrant quadrant)
    {
        if (!quadrant.HasKlingons)  # 如果象限中没有克林贡人
        {
            _io.WriteLine(Strings.NoEnemyShips);  # 输出"NoEnemyShips"字符串
            return CommandResult.Ok;  # 返回命令执行结果为Ok
        }

        if (_enterprise.Computer.IsDamaged)  # 如果企业号的电脑系统受损
        {
            _io.WriteLine("Computer failure hampers accuracy");  # 输出"Computer failure hampers accuracy"字符串
        }

        _io.Write($"Phasers locked on target;  ");  # 输出"Phasers locked on target;"字符串
        // 获取相位器强度
        var phaserStrength = GetPhaserStrength();
        // 如果相位器强度小于0，则返回命令结果为Ok
        if (phaserStrength < 0) { return CommandResult.Ok; }
        // 使用相位器强度消耗能量
        _enterprise.UseEnergy(phaserStrength);
        // 获取每个敌人的相位器强度
        var perEnemyStrength = GetPerTargetPhaserStrength(phaserStrength, quadrant.KlingonCount);
        // 遍历象限中的克林贡人员，并解决对其的攻击
        foreach (var klingon in quadrant.Klingons.ToList())
        {
            ResolveHitOn(klingon, perEnemyStrength, quadrant);
        }
        // 克林贡人员对企业号进行攻击
        return quadrant.KlingonsFireOnEnterprise();
    }

    // 获取相位器强度
    private float GetPhaserStrength()
    {
        while (true)
        {
            _io.WriteLine($"Energy available = {_enterprise.Energy} units");  // 输出企业号飞船的能量单位数量
            var phaserStrength = _io.ReadNumber("Number of units to fire");  // 从输入中读取要发射的相位炮强度

            if (phaserStrength <= _enterprise.Energy) { return phaserStrength; }  // 如果相位炮强度小于等于企业号飞船的能量单位数量，则返回相位炮强度
        }
    }

    private float GetPerTargetPhaserStrength(float phaserStrength, int targetCount)
    {
        if (_enterprise.Computer.IsDamaged)  // 如果企业号飞船的电脑系统受损
        {
            phaserStrength *= _random.NextFloat();  // 相位炮强度乘以一个随机浮点数
        }

        return phaserStrength / targetCount;  // 返回每个目标的相位炮强度
    }

    private void ResolveHitOn(Klingon klingon, float perEnemyStrength, Quadrant quadrant)
    {
        var distance = _enterprise.SectorCoordinates.GetDistanceTo(klingon.Sector);  // 计算企业号飞船与克林贡战舰的距离
        # 计算每次攻击的力量，根据敌人的强度和距离来计算，加上一个随机浮点数
        var hitStrength = (int)(perEnemyStrength / distance * (2 + _random.NextFloat()));

        # 如果克林贡受到攻击并且能量小于等于0，则从象限中移除克林贡
        # 否则，显示克林贡的剩余能量
        if (klingon.TakeHit(hitStrength))
        {
            _io.WriteLine($"{hitStrength} unit hit on Klingon at sector {klingon.Sector}");
            _io.WriteLine(
                klingon.Energy <= 0
                    ? quadrant.Remove(klingon)
                    : $"   (sensors show {klingon.Energy} units remaining)");
        }
        # 如果克林贡没有受到伤害，则显示无伤害的信息
        else
        {
            _io.WriteLine($"Sensors show no damage to enemy at {klingon.Sector}");
        }
    }
}
```
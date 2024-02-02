# `basic-computer-games\84_Super_Star_Trek\csharp\Systems\PhaserControl.cs`

```py
using System.Linq; // 导入 LINQ 扩展方法
using Games.Common.IO; // 导入通用输入输出类
using Games.Common.Randomness; // 导入通用随机数生成类
using SuperStarTrek.Commands; // 导入星际迷航游戏命令类
using SuperStarTrek.Objects; // 导入星际迷航游戏对象类
using SuperStarTrek.Resources; // 导入星际迷航游戏资源类
using SuperStarTrek.Space; // 导入星际迷航游戏空间类

namespace SuperStarTrek.Systems; // 定义 SuperStarTrek.Systems 命名空间

internal class PhaserControl : Subsystem // 定义 PhaserControl 类，继承自 Subsystem 类
{
    private readonly Enterprise _enterprise; // 声明私有 Enterprise 对象 _enterprise
    private readonly IReadWrite _io; // 声明私有 IReadWrite 对象 _io
    private readonly IRandom _random; // 声明私有 IRandom 对象 _random

    internal PhaserControl(Enterprise enterprise, IReadWrite io, IRandom random) // 定义 PhaserControl 类的构造函数
        : base("Phaser Control", Command.PHA, io) // 调用基类 Subsystem 的构造函数
    {
        _enterprise = enterprise; // 初始化 _enterprise
        _io = io; // 初始化 _io
        _random = random; // 初始化 _random
    }

    protected override bool CanExecuteCommand() => IsOperational("Phasers inoperative"); // 重写基类的 CanExecuteCommand 方法

    protected override CommandResult ExecuteCommandCore(Quadrant quadrant) // 重写基类的 ExecuteCommandCore 方法
    {
        if (!quadrant.HasKlingons) // 如果象限中没有克林贡战舰
        {
            _io.WriteLine(Strings.NoEnemyShips); // 输出 "No enemy ships in this quadrant."
            return CommandResult.Ok; // 返回命令执行结果为 Ok
        }

        if (_enterprise.Computer.IsDamaged) // 如果企业号的电脑系统受损
        {
            _io.WriteLine("Computer failure hampers accuracy"); // 输出 "Computer failure hampers accuracy."
        }

        _io.Write($"Phasers locked on target;  "); // 输出 "Phasers locked on target;"

        var phaserStrength = GetPhaserStrength(); // 调用 GetPhaserStrength 方法获取激光器强度
        if (phaserStrength < 0) { return CommandResult.Ok; } // 如果激光器强度小于 0，则返回命令执行结果为 Ok

        _enterprise.UseEnergy(phaserStrength); // 使用相应能量

        var perEnemyStrength = GetPerTargetPhaserStrength(phaserStrength, quadrant.KlingonCount); // 调用 GetPerTargetPhaserStrength 方法获取每个目标的激光器强度

        foreach (var klingon in quadrant.Klingons.ToList()) // 遍历象限中的克林贡战舰列表
        {
            ResolveHitOn(klingon, perEnemyStrength, quadrant); // 对每个克林贡战舰进行攻击
        }

        return quadrant.KlingonsFireOnEnterprise(); // 克林贡战舰对企业号进行攻击
    }

    private float GetPhaserStrength() // 定义私有方法 GetPhaserStrength
    {
        while (true) // 无限循环
        {
            _io.WriteLine($"Energy available = {_enterprise.Energy} units"); // 输出可用能量
            var phaserStrength = _io.ReadNumber("Number of units to fire"); // 从输入中读取激光器强度

            if (phaserStrength <= _enterprise.Energy) { return phaserStrength; } // 如果激光器强度小于等于可用能量，则返回激光器强度
        }
    }

    private float GetPerTargetPhaserStrength(float phaserStrength, int targetCount) // 定义私有方法 GetPerTargetPhaserStrength
    {
        // 如果企业号受损，那么相位器强度会减弱
        if (_enterprise.Computer.IsDamaged)
        {
            phaserStrength *= _random.NextFloat();
        }

        // 返回每个敌人的相位器强度除以目标数量的结果
        return phaserStrength / targetCount;
    }

    // 处理对克林贡的命中
    private void ResolveHitOn(Klingon klingon, float perEnemyStrength, Quadrant quadrant)
    {
        // 计算企业号与克林贡的距离
        var distance = _enterprise.SectorCoordinates.GetDistanceTo(klingon.Sector);
        // 计算命中强度，根据敌人的强度、距离和随机因素计算
        var hitStrength = (int)(perEnemyStrength / distance * (2 + _random.NextFloat()));

        // 如果克林贡受到攻击
        if (klingon.TakeHit(hitStrength))
        {
            // 输出对克林贡的命中信息
            _io.WriteLine($"{hitStrength} unit hit on Klingon at sector {klingon.Sector}");
            // 如果克林贡能量小于等于0，从象限中移除克林贡，否则输出剩余能量信息
            _io.WriteLine(
                klingon.Energy <= 0
                    ? quadrant.Remove(klingon)
                    : $"   (sensors show {klingon.Energy} units remaining)");
        }
        else
        {
            // 输出传感器显示克林贡未受到伤害的信息
            _io.WriteLine($"Sensors show no damage to enemy at {klingon.Sector}");
        }
    }
# 闭合前面的函数定义
```
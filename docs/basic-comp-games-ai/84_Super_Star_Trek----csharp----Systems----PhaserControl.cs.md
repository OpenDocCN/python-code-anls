# `basic-computer-games\84_Super_Star_Trek\csharp\Systems\PhaserControl.cs`

```

// 引入所需的命名空间
using System.Linq;
using Games.Common.IO;
using Games.Common.Randomness;
using SuperStarTrek.Commands;
using SuperStarTrek.Objects;
using SuperStarTrek.Resources;
using SuperStarTrek.Space;

namespace SuperStarTrek.Systems;

// 定义 PhaserControl 类，继承自 Subsystem 类
internal class PhaserControl : Subsystem
{
    // 声明私有字段
    private readonly Enterprise _enterprise;
    private readonly IReadWrite _io;
    private readonly IRandom _random;

    // 定义构造函数
    internal PhaserControl(Enterprise enterprise, IReadWrite io, IRandom random)
        : base("Phaser Control", Command.PHA, io)
    {
        _enterprise = enterprise;
        _io = io;
        _random = random;
    }

    // 重写父类的 CanExecuteCommand 方法
    protected override bool CanExecuteCommand() => IsOperational("Phasers inoperative");

    // 重写父类的 ExecuteCommandCore 方法
    protected override CommandResult ExecuteCommandCore(Quadrant quadrant)
    {
        // 检查象限中是否有克林贡战舰
        if (!quadrant.HasKlingons)
        {
            _io.WriteLine(Strings.NoEnemyShips);
            return CommandResult.Ok;
        }

        // 检查企业号的电脑是否受损
        if (_enterprise.Computer.IsDamaged)
        {
            _io.WriteLine("Computer failure hampers accuracy");
        }

        _io.Write($"Phasers locked on target;  ");

        // 获取相应的相位炮强度
        var phaserStrength = GetPhaserStrength();
        if (phaserStrength < 0) { return CommandResult.Ok; }

        // 使用相应的能量
        _enterprise.UseEnergy(phaserStrength);

        // 获取每个目标的相位炮强度
        var perEnemyStrength = GetPerTargetPhaserStrength(phaserStrength, quadrant.KlingonCount);

        // 遍历象限中的克林贡战舰，解决对其的攻击
        foreach (var klingon in quadrant.Klingons.ToList())
        {
            ResolveHitOn(klingon, perEnemyStrength, quadrant);
        }

        // 克林贡战舰对企业号进行攻击
        return quadrant.KlingonsFireOnEnterprise();
    }

    // 获取相位炮强度
    private float GetPhaserStrength()
    {
        while (true)
        {
            _io.WriteLine($"Energy available = {_enterprise.Energy} units");
            var phaserStrength = _io.ReadNumber("Number of units to fire");

            if (phaserStrength <= _enterprise.Energy) { return phaserStrength; }
        }
    }

    // 获取每个目标的相位炮强度
    private float GetPerTargetPhaserStrength(float phaserStrength, int targetCount)
    {
        if (_enterprise.Computer.IsDamaged)
        {
            phaserStrength *= _random.NextFloat();
        }

        return phaserStrength / targetCount;
    }

    // 解决对克林贡战舰的攻击
    private void ResolveHitOn(Klingon klingon, float perEnemyStrength, Quadrant quadrant)
    {
        var distance = _enterprise.SectorCoordinates.GetDistanceTo(klingon.Sector);
        var hitStrength = (int)(perEnemyStrength / distance * (2 + _random.NextFloat()));

        if (klingon.TakeHit(hitStrength))
        {
            _io.WriteLine($"{hitStrength} unit hit on Klingon at sector {klingon.Sector}");
            _io.WriteLine(
                klingon.Energy <= 0
                    ? quadrant.Remove(klingon)
                    : $"   (sensors show {klingon.Energy} units remaining)");
        }
        else
        {
            _io.WriteLine($"Sensors show no damage to enemy at {klingon.Sector}");
        }
    }
}

```
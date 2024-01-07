# `basic-computer-games\84_Super_Star_Trek\csharp\Systems\WarpEngines.cs`

```

// 引入命名空间
using System;
using Games.Common.IO;
using SuperStarTrek.Commands;
using SuperStarTrek.Objects;
using SuperStarTrek.Resources;
using SuperStarTrek.Space;

// 定义 WarpEngines 类，继承自 Subsystem 类
namespace SuperStarTrek.Systems
{
    internal class WarpEngines : Subsystem
    {
        private readonly Enterprise _enterprise; // 声明私有变量 _enterprise，类型为 Enterprise
        private readonly IReadWrite _io; // 声明私有变量 _io，类型为 IReadWrite

        // WarpEngines 类的构造函数
        internal WarpEngines(Enterprise enterprise, IReadWrite io)
            : base("Warp Engines", Command.NAV, io) // 调用父类 Subsystem 的构造函数
        {
            _enterprise = enterprise; // 初始化 _enterprise 变量
            _io = io; // 初始化 _io 变量
        }

        // 重写父类 Subsystem 的 ExecuteCommandCore 方法
        protected override CommandResult ExecuteCommandCore(Quadrant quadrant)
        {
            // 尝试读取航向和航速，并计算移动距离
            if (_io.TryReadCourse("Course", "   Lt. Sulu", out var course) &&
                TryGetWarpFactor(out var warpFactor) &&
                TryGetDistanceToMove(warpFactor, out var distanceToMove))
            {
                // 克林贡人移动并开火
                var result = quadrant.KlingonsMoveAndFire();
                if (result.IsGameOver) { return result; }

                // 修复系统，改变随机系统状态，移动企业号
                _enterprise.RepairSystems(warpFactor);
                _enterprise.VaryConditionOfRandomSystem();
                var timeElapsed = _enterprise.Move(course, warpFactor, distanceToMove);

                // 如果企业号停靠，则放下护盾，加油，补充光子鱼雷
                if (_enterprise.IsDocked)
                {
                    _enterprise.ShieldControl.DropShields();
                    _enterprise.Refuel();
                    _enterprise.PhotonTubes.ReplenishTorpedoes();
                }

                // 显示信息
                _enterprise.Quadrant.Display(Strings.NowEntering);

                return CommandResult.Elapsed(timeElapsed); // 返回命令结果
            }

            return CommandResult.Ok; // 返回命令结果
        }

        // 尝试获取航速
        private bool TryGetWarpFactor(out float warpFactor)
        {
            var maximumWarp = IsDamaged ? 0.2f : 8; // 计算最大航速
            if (_io.TryReadNumberInRange("Warp Factor", 0, maximumWarp, out warpFactor)) // 尝试读取航速
            {
                return warpFactor > 0; // 如果航速大于0，则返回 true
            }

            // 如果航速小于等于0或者航速大于最大航速，则输出相应信息
            _io.WriteLine(
                IsDamaged && warpFactor > maximumWarp
                    ? "Warp engines are damaged.  Maximum speed = warp 0.2"
                    : $"  Chief Engineer Scott reports, 'The engines won't take warp {warpFactor} !'");

            return false; // 返回 false
        }

        // 尝试获取移动距离
        private bool TryGetDistanceToMove(float warpFactor, out int distanceToTravel)
        {
            distanceToTravel = (int)Math.Round(warpFactor * 8, MidpointRounding.AwayFromZero); // 计算移动距离
            if (distanceToTravel <= _enterprise.Energy) { return true; } // 如果移动距离小于等于能量，则返回 true

            // 如果能量不足，则输出相应信息
            _io.WriteLine("Engineering reports, 'Insufficient energy available");
            _io.WriteLine($"                      for maneuvering at warp {warpFactor} !'");

            // 如果能量不足且护盾未受损，则输出相应信息
            if (distanceToTravel <= _enterprise.TotalEnergy && !_enterprise.ShieldControl.IsDamaged)
            {
                _io.Write($"Deflector control room acknowledges {_enterprise.ShieldControl.ShieldEnergy} ");
                _io.WriteLine("units of energy");
                _io.WriteLine("                         presently deployed to shields.");
            }

            return false; // 返回 false
        }
    }
}

```
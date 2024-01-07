# `basic-computer-games\84_Super_Star_Trek\csharp\Systems\ShortRangeSensors.cs`

```

// 引入所需的命名空间
using System;
using System.Collections.Generic;
using System.Linq;
using Games.Common.IO;
using SuperStarTrek.Commands;
using SuperStarTrek.Objects;
using SuperStarTrek.Resources;
using SuperStarTrek.Space;

// 创建 ShortRangeSensors 类，继承自 Subsystem 类
namespace SuperStarTrek.Systems
{
    internal class ShortRangeSensors : Subsystem
    {
        // 声明私有变量
        private readonly Enterprise _enterprise;
        private readonly Galaxy _galaxy;
        private readonly Game _game;
        private readonly IReadWrite _io;

        // ShortRangeSensors 类的构造函数
        internal ShortRangeSensors(Enterprise enterprise, Galaxy galaxy, Game game, IReadWrite io)
            : base("Short Range Sensors", Command.SRS, io)
        {
            // 初始化私有变量
            _enterprise = enterprise;
            _galaxy = galaxy;
            _game = game;
            _io = io;
        }

        // 重写 ExecuteCommandCore 方法
        protected override CommandResult ExecuteCommandCore(Quadrant quadrant)
        {
            // 如果 Enterprise 被停靠，则输出相应信息
            if (_enterprise.IsDocked)
            {
                _io.WriteLine(Strings.ShieldsDropped);
            }

            // 如果 Condition 小于 0，则输出相应信息
            if (Condition < 0)
            {
                _io.WriteLine(Strings.ShortRangeSensorsOut);
            }

            // 输出分隔线
            _io.WriteLine("---------------------------------");
            // 获取显示行和状态行，合并成一行输出
            quadrant.GetDisplayLines()
                .Zip(GetStatusLines(), (sectors, status) => $" {sectors}         {status}")
                .ToList()
                .ForEach(l => _io.WriteLine(l));
            // 输出分隔线
            _io.WriteLine("---------------------------------");

            // 返回命令执行结果为 Ok
            return CommandResult.Ok;
        }

        // 获取状态行的方法
        internal IEnumerable<string> GetStatusLines()
        {
            yield return $"Stardate           {_game.Stardate}";
            yield return $"Condition          {_enterprise.Condition}";
            yield return $"Quadrant           {_enterprise.QuadrantCoordinates}";
            yield return $"Sector             {_enterprise.SectorCoordinates}";
            yield return $"Photon torpedoes   {_enterprise.PhotonTubes.TorpedoCount}";
            yield return $"Total energy       {Math.Ceiling(_enterprise.TotalEnergy)}";
            yield return $"Shields            {(int)_enterprise.ShieldControl.ShieldEnergy}";
            yield return $"Klingons remaining {_galaxy.KlingonCount}";
        }
    }
}

```
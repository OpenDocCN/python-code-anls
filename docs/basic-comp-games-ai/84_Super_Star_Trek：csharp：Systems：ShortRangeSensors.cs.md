# `84_Super_Star_Trek\csharp\Systems\ShortRangeSensors.cs`

```
{
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
    namespace SuperStarTrek.Systems;

    internal class ShortRangeSensors : Subsystem
    {
        // 声明私有变量 _enterprise, _galaxy, _game, _io，并初始化
        private readonly Enterprise _enterprise;
        private readonly Galaxy _galaxy;
        private readonly Game _game;
        private readonly IReadWrite _io;

        // 创建 ShortRangeSensors 类的构造函数，接受 Enterprise, Galaxy, Game, IReadWrite 四个参数
        internal ShortRangeSensors(Enterprise enterprise, Galaxy galaxy, Game game, IReadWrite io)
        : base("Short Range Sensors", Command.SRS, io)
    {
        _enterprise = enterprise;  # 将传入的 enterprise 参数赋值给 _enterprise 变量
        _galaxy = galaxy;  # 将传入的 galaxy 参数赋值给 _galaxy 变量
        _game = game;  # 将传入的 game 参数赋值给 _game 变量
        _io = io;  # 将传入的 io 参数赋值给 _io 变量
    }

    protected override CommandResult ExecuteCommandCore(Quadrant quadrant)
    {
        if (_enterprise.IsDocked)  # 如果 _enterprise 对象的 IsDocked 属性为真
        {
            _io.WriteLine(Strings.ShieldsDropped);  # 在 _io 对象上调用 WriteLine 方法，输出 "ShieldsDropped" 字符串
        }

        if (Condition < 0)  # 如果 Condition 变量的值小于 0
        {
            _io.WriteLine(Strings.ShortRangeSensorsOut);  # 在 _io 对象上调用 WriteLine 方法，输出 "ShortRangeSensorsOut" 字符串
        }
        _io.WriteLine("---------------------------------");  // 打印分隔线
        quadrant.GetDisplayLines()  // 获取象限的显示行
            .Zip(GetStatusLines(), (sectors, status) => $" {sectors}         {status}")  // 将象限的显示行和状态行进行合并
            .ToList()  // 转换为列表
            .ForEach(l => _io.WriteLine(l));  // 遍历列表并打印每一行
        _io.WriteLine("---------------------------------");  // 打印分隔线

        return CommandResult.Ok;  // 返回命令执行结果为成功
    }

    internal IEnumerable<string> GetStatusLines()  // 获取状态行的方法
    {
        yield return $"Stardate           {_game.Stardate}";  // 返回星际日期
        yield return $"Condition          {_enterprise.Condition}";  // 返回企业的状态
        yield return $"Quadrant           {_enterprise.QuadrantCoordinates}";  // 返回象限坐标
        yield return $"Sector             {_enterprise.SectorCoordinates}";  // 返回扇区坐标
        yield return $"Photon torpedoes   {_enterprise.PhotonTubes.TorpedoCount}";  // 返回光子鱼雷数量
        yield return $"Total energy       {Math.Ceiling(_enterprise.TotalEnergy)}";  // 返回总能量
        yield return $"Shields            {(int)_enterprise.ShieldControl.ShieldEnergy}";  // 返回护盾能量
        yield return $"Klingons remaining {_galaxy.KlingonCount}";  // 返回剩余克林贡数量
    }
    }
```

这部分代码是一个缩进错误，应该删除。
```
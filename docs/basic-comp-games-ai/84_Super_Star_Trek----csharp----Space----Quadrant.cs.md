# `basic-computer-games\84_Super_Star_Trek\csharp\Space\Quadrant.cs`

```py
using System;
using System.Collections.Generic;
using System.Linq;
using Games.Common.IO;
using Games.Common.Randomness;
using SuperStarTrek.Commands;
using SuperStarTrek.Objects;
using SuperStarTrek.Resources;

namespace SuperStarTrek.Space;
// 定义名为Quadrant的内部类
internal class Quadrant
{
    private readonly QuadrantInfo _info; // 保存QuadrantInfo对象的只读字段
    private readonly IRandom _random; // 保存IRandom对象的只读字段
    private readonly Dictionary<Coordinates, object> _sectors; // 保存坐标和对象的字典的只读字段
    private readonly Enterprise _enterprise; // 保存Enterprise对象的只读字段
    private readonly IReadWrite _io; // 保存IReadWrite对象的只读字段
    private bool _entered = false; // 初始化_entered字段为false

    internal Quadrant(
        QuadrantInfo info, // 传入QuadrantInfo对象
        Enterprise enterprise, // 传入Enterprise对象
        IRandom random, // 传入IRandom对象
        Galaxy galaxy, // 传入Galaxy对象
        IReadWrite io) // 传入IReadWrite对象
    {
        _info = info; // 将传入的QuadrantInfo对象赋值给_info字段
        _random = random; // 将传入的IRandom对象赋值给_random字段
        _io = io; // 将传入的IReadWrite对象赋值给_io字段
        Galaxy = galaxy; // 将传入的Galaxy对象赋值给Galaxy字段

        info.MarkAsKnown(); // 标记QuadrantInfo对象为已知
        _sectors = new() { [enterprise.SectorCoordinates] = _enterprise = enterprise }; // 初始化_sectors字典，将Enterprise对象放入对应的坐标位置
        PositionObject(sector => new Klingon(sector, _random), _info.KlingonCount); // 根据Klingon数量在对应的坐标位置放置Klingon对象
        if (_info.HasStarbase) // 如果有星舰基地
        {
            Starbase = PositionObject(sector => new Starbase(sector, _random, io)); // 在对应的坐标位置放置Starbase对象
        }
        PositionObject(_ => new Star(), _info.StarCount); // 根据星星数量在对应的坐标位置放置Star对象
    }

    internal Coordinates Coordinates => _info.Coordinates; // 获取Quadrant的坐标

    internal bool HasKlingons => _info.KlingonCount > 0; // 判断是否有Klingon

    internal int KlingonCount => _info.KlingonCount; // 获取Klingon数量

    internal bool HasStarbase => _info.HasStarbase; // 判断是否有星舰基地

    internal Starbase Starbase { get; } // 获取Starbase对象

    internal Galaxy Galaxy { get; } // 获取Galaxy对象

    internal bool EnterpriseIsNextToStarbase => // 判断Enterprise是否在Starbase旁边
        _info.HasStarbase &&
        Math.Abs(_enterprise.SectorCoordinates.X - Starbase.Sector.X) <= 1 &&
        Math.Abs(_enterprise.SectorCoordinates.Y - Starbase.Sector.Y) <= 1;

    internal IEnumerable<Klingon> Klingons => _sectors.Values.OfType<Klingon>(); // 获取所有Klingon对象的集合

    public override string ToString() => _info.Name; // 返回Quadrant的名称

    private T PositionObject<T>(Func<Coordinates, T> objectFactory) // 定义一个私有方法用于在指定坐标位置放置对象
    # 获取一个随机的空扇区
    var sector = GetRandomEmptySector();
    # 将对象工厂创建的对象放入扇区字典中
    _sectors[sector] = objectFactory.Invoke(sector);
    # 返回放入扇区字典的对象
    return (T)_sectors[sector];
    }

    # 根据对象工厂和数量定位对象
    private void PositionObject(Func<Coordinates, object> objectFactory, int count)
    {
        # 循环指定次数，定位对象
        for (int i = 0; i < count; i++)
        {
            PositionObject(objectFactory);
        }
    }

    # 显示信息
    internal void Display(string textFormat)
    {
        # 如果未进入，则写入文本格式
        if (!_entered)
        {
            _io.Write(textFormat, this);
            _entered = true;
        }

        # 如果克林贡数量大于0，则写入战斗区域信息
        if (_info.KlingonCount > 0)
        {
            _io.Write(Strings.CombatArea);
            # 如果企业的护盾能量小于等于200，则写入低护盾信息
            if (_enterprise.ShieldControl.ShieldEnergy <= 200) { _io.Write(Strings.LowShields); }
        }

        # 执行SRS指令
        _enterprise.Execute(Command.SRS);
    }

    # 检查指定坐标处是否有对象
    internal bool HasObjectAt(Coordinates coordinates) => _sectors.ContainsKey(coordinates);

    # 检查指定坐标处是否有鱼雷碰撞，并返回信息和游戏是否结束
    internal bool TorpedoCollisionAt(Coordinates coordinates, out string message, out bool gameOver)
    {
        gameOver = false;
        message = default;

        # 根据坐标处的对象类型进行不同的处理
        switch (_sectors.GetValueOrDefault(coordinates))
        {
            # 如果是克林贡，则移除并返回相应信息
            case Klingon klingon:
                message = Remove(klingon);
                gameOver = Galaxy.KlingonCount == 0;
                return true;

            # 如果是星星，则返回相应信息
            case Star _:
                message = $"Star at {coordinates} absorbed torpedo energy.";
                return true;

            # 如果是星舰基地，则移除并返回相应信息
            case Starbase _:
                _sectors.Remove(coordinates);
                _info.RemoveStarbase();
                message = "*** Starbase destroyed ***" +
                    (Galaxy.StarbaseCount > 0 ? Strings.CourtMartial : Strings.RelievedOfCommand);
                gameOver = Galaxy.StarbaseCount == 0;
                return true;

            # 其他情况返回false
            default:
                return false;
        }
    }

    # 移除克林贡对象并返回相应信息
    internal string Remove(Klingon klingon)
    {
        _sectors.Remove(klingon.Sector);
        _info.RemoveKlingon();
        return "*** Klingon destroyed ***";
    }
    # KlingonsMoveAndFire 方法，让克林贡移动并开火
    internal CommandResult KlingonsMoveAndFire()
    {
        # 遍历克林贡列表
        foreach (var klingon in Klingons.ToList())
        {
            # 获取一个随机的空扇区
            var newSector = GetRandomEmptySector();
            # 从扇区列表中移除当前克林贡所在的扇区
            _sectors.Remove(klingon.Sector);
            # 将当前克林贡移动到新的扇区
            _sectors[newSector] = klingon;
            klingon.MoveTo(newSector);
        }

        # 返回克林贡对企业号的开火结果
        return KlingonsFireOnEnterprise();
    }

    # KlingonsFireOnEnterprise 方法，克林贡对企业号开火
    internal CommandResult KlingonsFireOnEnterprise()
    {
        # 如果企业号旁边有星舰基地并且克林贡列表不为空
        if (EnterpriseIsNextToStarbase && Klingons.Any())
        {
            # 星舰基地保护企业号
            Starbase.ProtectEnterprise();
            return CommandResult.Ok;
        }

        # 遍历克林贡列表
        foreach (var klingon in Klingons)
        {
            # 克林贡对企业号开火，并获取结果
            var result = klingon.FireOn(_enterprise);
            # 如果游戏结束，返回结果
            if (result.IsGameOver) { return result; }
        }

        return CommandResult.Ok;
    }

    # 获取一个随机的空扇区
    private Coordinates GetRandomEmptySector()
    {
        while (true)
        {
            # 获取一个随机的扇区坐标
            var sector = _random.NextCoordinate();
            # 如果扇区列表中不包含该扇区，返回该扇区坐标
            if (!_sectors.ContainsKey(sector))
            {
                return sector;
            }
        }
    }

    # 获取显示行的集合
    internal IEnumerable<string> GetDisplayLines() => Enumerable.Range(0, 8).Select(x => GetDisplayLine(x));

    # 获取显示行
    private string GetDisplayLine(int x) =>
        # 将每个扇区的内容连接成一行
        string.Join(
            " ",
            Enumerable
                .Range(0, 8)
                .Select(y => new Coordinates(x, y))
                .Select(c => _sectors.GetValueOrDefault(c))
                .Select(o => o?.ToString() ?? "   "));

    # 设置企业号所在的扇区
    internal void SetEnterpriseSector(Coordinates sector)
    {
        # 从扇区列表中移除企业号当前所在的扇区
        _sectors.Remove(_enterprise.SectorCoordinates);
        # 将企业号放置到新的扇区
        _sectors[sector] = _enterprise;
    }
# 闭合前面的函数定义
```
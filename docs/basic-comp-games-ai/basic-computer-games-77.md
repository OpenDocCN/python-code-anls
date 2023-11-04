# BasicComputerGames源码解析 77

# `84_Super_Star_Trek/csharp/Objects/Klingon.cs`

这段代码定义了一个名为Klingon的类，它是一个Klingon人形的类，用于在游戏中进行AI控制。

首先，它引入了Games.Common.Randomness命名空间中的IRandom类，用于产生随机数。接着，它实现了ToString方法，用于将Klingon对象转换为字符串。

然后，它定义了一个Klingon的构造函数，该函数接收一个坐标和一个IRandom实例作为参数，用于初始化Klingon对象。

Klingon对象还实现了FireOn和TakeHit方法，用于控制Klingon对象的攻击和防御行为。

最后，它继承了Objects命名空间中的所有类，以便使用其接口的方法来操作游戏世界中的对象。


```
using Games.Common.Randomness;
using SuperStarTrek.Commands;
using SuperStarTrek.Space;

namespace SuperStarTrek.Objects;

internal class Klingon
{
    private readonly IRandom _random;

    internal Klingon(Coordinates sector, IRandom random)
    {
        Sector = sector;
        _random = random;
        Energy = _random.NextFloat(100, 300);
    }

    internal float Energy { get; private set; }

    internal Coordinates Sector { get; private set; }

    public override string ToString() => "+K+";

    internal CommandResult FireOn(Enterprise enterprise)
    {
        var attackStrength = _random.NextFloat();
        var distanceToEnterprise = Sector.GetDistanceTo(enterprise.SectorCoordinates);
        var hitStrength = (int)(Energy * (2 + attackStrength) / distanceToEnterprise);
        Energy /= 3 + attackStrength;

        return enterprise.TakeHit(Sector, hitStrength);
    }

    internal bool TakeHit(int hitStrength)
    {
        if (hitStrength < 0.15 * Energy) { return false; }

        Energy -= hitStrength;
        return true;
    }

    internal void MoveTo(Coordinates newSector) => Sector = newSector;
}

```

# `84_Super_Star_Trek/csharp/Objects/Star.cs`



这段代码定义了一个名为Star的内部类，继承自名为Objects的命名空间。

内部类Star包含一个名为ToString的静态方法，该方法返回一个字符串表示自己的内部对象。

在内部类Objects中，没有定义任何成员变量或方法，它只是一个用于继承其他类或模型的命名空间。

由于Star是内部类，因此它无法直接访问外部Objects中的任何成员变量或方法。


```
namespace SuperStarTrek.Objects;

internal class Star
{
    public override string ToString() => " * ";
}

```

# `84_Super_Star_Trek/csharp/Objects/Starbase.cs`



这段代码定义了一个名为 `Starbase` 的类，用于创建一个星际飞船。这个类包含以下几个方法：

1. `Starbase` 类接受三个参数：一个坐标 `sector`，一个随机数生成器 `random`，和一个 IReadWrite 接口的实例 `_io`。
2. 构造函数，将坐标 `sector`、随机数生成器 `random` 和 IReadWrite 接口的实例 `_io` 传入。
3. `Sector` 属性，返回创建的星际飞船所属的船渠。
4. `ToString` 方法，返回一个字符串表示所有的 `Starbase` 对象连接在一起。
5. `TryRepair` 方法，用于尝试修复受损的飞船。该方法接收一个 Enterprise 对象和一个输出浮点数来表示修复所需的时间。如果修复成功，该输出将设置为 1，否则将设置为 0.9 并返回修复是否成功。修复时间会根据每台机器的类型和状态进行调整。
6. `ProtectEnterprise` 方法，用于保护 Enterprise 对象。该方法将输出 "Protected: The ship is protected."。


```
using Games.Common.IO;
using Games.Common.Randomness;
using SuperStarTrek.Resources;
using SuperStarTrek.Space;

namespace SuperStarTrek.Objects;

internal class Starbase
{
    private readonly IReadWrite _io;
    private readonly float _repairDelay;

    internal Starbase(Coordinates sector, IRandom random, IReadWrite io)
    {
        Sector = sector;
        _repairDelay = random.NextFloat(0.5f);
        _io = io;
    }

    internal Coordinates Sector { get; }

    public override string ToString() => ">!<";

    internal bool TryRepair(Enterprise enterprise, out float repairTime)
    {
        repairTime = enterprise.DamagedSystemCount * 0.1f + _repairDelay;
        if (repairTime >= 1) { repairTime = 0.9f; }

        _io.Write(Strings.RepairEstimate, repairTime);
        if (_io.GetYesNo(Strings.RepairPrompt, IReadWriteExtensions.YesNoMode.TrueOnY))
        {
            foreach (var system in enterprise.Systems)
            {
                system.Repair();
            }
            return true;
        }

        repairTime = 0;
        return false;
    }

    internal void ProtectEnterprise() => _io.WriteLine(Strings.Protected);
}

```

# `84_Super_Star_Trek/csharp/Resources/Strings.cs`



这段代码是一个程序集，包含了多个字符串资源类，这些类定义了游戏中的各种对话和消息。这些资源类提供了玩家与游戏世界之间交互所需的文本信息。

具体来说，这些资源类包括：CombatArea、Congratulations、CourtMartial、Destroyed、EndOfMission、Enterprise、Instructions、LowShields、NoEnemyShips、NoStarbase、NowEntering、Orders、PermissionDenied、Protected、RegionNames、RepairEstimate、RepairPrompt、ReplayPrompt、ShieldsDropped、ShieldsSet、ShortRangeSensorsOut、StartText、Stranded、Title和Title。

每个资源类都有一个名为“GetResource”的静态方法，这个方法根据传入的“name”参数来查找或读取资源文件中的文本内容。在游戏程序中，玩家可以通过与对话框中的NPC(非玩家角色)交互来获取这些对话和消息，以便更好地了解游戏世界。


```
﻿using System.IO;
using System.Reflection;
using System.Runtime.CompilerServices;

namespace SuperStarTrek.Resources
{
    internal static class Strings
    {
        internal static string CombatArea => GetResource();

        internal static string Congratulations => GetResource();

        internal static string CourtMartial => GetResource();

        internal static string Destroyed => GetResource();

        internal static string EndOfMission => GetResource();

        internal static string Enterprise => GetResource();

        internal static string Instructions => GetResource();

        internal static string LowShields => GetResource();

        internal static string NoEnemyShips => GetResource();

        internal static string NoStarbase => GetResource();

        internal static string NowEntering => GetResource();

        internal static string Orders => GetResource();

        internal static string PermissionDenied => GetResource();

        internal static string Protected => GetResource();

        internal static string RegionNames => GetResource();

        internal static string RelievedOfCommand => GetResource();

        internal static string RepairEstimate => GetResource();

        internal static string RepairPrompt => GetResource();

        internal static string ReplayPrompt => GetResource();

        internal static string ShieldsDropped => GetResource();

        internal static string ShieldsSet => GetResource();

        internal static string ShortRangeSensorsOut => GetResource();

        internal static string StartText => GetResource();

        internal static string Stranded => GetResource();

        internal static string Title => GetResource();

        private static string GetResource([CallerMemberName] string name = "")
        {
            var streamName = $"SuperStarTrek.Resources.{name}.txt";
            using var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(streamName);
            using var reader = new StreamReader(stream);

            return reader.ReadToEnd();
        }
    }
}

```

# `84_Super_Star_Trek/csharp/Space/Coordinates.cs`

This is a class called `Coordinates` that represents a point in a 2D grid. The `Coordinates` class has two public fields: `X` and `Y`, which are the x and y coordinates of the point, respectively. It also has two internal fields: `RegionIndex` and `SubRegionIndex`, which are the indexes for the lower left sub-region of the grid.

The `X` and `Y` fields are validated to ensure that they are within the valid range of 0 to 7. If they are not valid, an `ArgumentOutOfRangeException` is thrown.

The `Validated` method is a helper method that takes a value and an argument name, and returns the valid value.

The `IsValid` method is another helper method that checks if a given value is within the valid range of 0 to 7.

The `ToString` method returns a string representation of the coordinates in the format `X, Y`.

The `Deconstruct` method is a helper method that deconstructs the coordinates, giving the x and y values.

The `TryCreate` method is a helper method that tries to create a new coordinate object from the given x and y values. It does this by first rounding the values to the nearest integer, and then using the `TryCreate` method from the `Coordinates` class to create the coordinate object if the rounding is successful.

The `DirectionAndDistance` class is a helper class that returns the direction and distance to a given coordinate direction.

The `GetDirectionAndDistanceTo` method is a helper method that returns the direction and distance to a given coordinate direction.

The `GetDistanceTo` method is a helper method that returns the distance to a given coordinate direction.


```
using System;
using SuperStarTrek.Utils;

namespace SuperStarTrek.Space;

// Represents the corrdintate of a quadrant in the galaxy, or a sector in a quadrant.
// Note that the origin is top-left, x increase downwards, and y increases to the right.
internal record Coordinates
{
    internal Coordinates(int x, int y)
    {
        X = Validated(x, nameof(x));
        Y = Validated(y, nameof(y));

        RegionIndex = (X << 1) + (Y >> 2);
        SubRegionIndex = Y % 4;
    }

    internal int X { get; }

    internal int Y { get; }

    internal int RegionIndex { get; }

    internal int SubRegionIndex { get; }

    private static int Validated(int value, string argumentName)
    {
        if (value >= 0 && value <= 7) { return value; }

        throw new ArgumentOutOfRangeException(argumentName, value, "Must be 0 to 7 inclusive");
    }

    private static bool IsValid(int value) => value >= 0 && value <= 7;

    public override string ToString() => $"{X+1} , {Y+1}";

    internal void Deconstruct(out int x, out int y)
    {
        x = X;
        y = Y;
    }

    internal static bool TryCreate(float x, float y, out Coordinates coordinates)
    {
        var roundedX = Round(x);
        var roundedY = Round(y);

        if (IsValid(roundedX) && IsValid(roundedY))
        {
            coordinates = new Coordinates(roundedX, roundedY);
            return true;
        }

        coordinates = default;
        return false;

        static int Round(float value) => (int)Math.Round(value, MidpointRounding.AwayFromZero);
    }

    internal (float Direction, float Distance) GetDirectionAndDistanceTo(Coordinates destination) =>
        DirectionAndDistance.From(this).To(destination);

    internal float GetDistanceTo(Coordinates destination)
    {
        var (_, distance) = GetDirectionAndDistanceTo(destination);
        return distance;
    }
}

```

# `84_Super_Star_Trek/csharp/Space/Course.cs`

This is a class that defines a process where a player moves their ship in a game of board game. The class includes methods for movement, which is based on the direction the ship is facing and the number of steps taken. The class also includes methods for handling the movement of the ship to specific sections of the board, as well as a method for getting the ship to a specific destination based on the number of steps taken and the ship's current position.

The class also defines a coordinate system that represents the position of the ship on the board. This allows for easy conversion between the position of the ship in the game to a position on the real board. The class also includes methods for getting the number of steps taken to reach a specific position, as well as a method for getting the ship's current position.

Note that this is just one possible implementation of a ship movement class and may not be the only way to implement the class.


```
using System;
using System.Collections.Generic;

namespace SuperStarTrek.Space;

// Implements the course calculations from the original code:
//     530 FORI=1TO9:C(I,1)=0:C(I,2)=0:NEXTI
//     540 C(3,1)=-1:C(2,1)=-1:C(4,1)=-1:C(4,2)=-1:C(5,2)=-1:C(6,2)=-1
//     600 C(1,2)=1:C(2,2)=1:C(6,1)=1:C(7,1)=1:C(8,1)=1:C(8,2)=1:C(9,2)=1
//
//     3110 X1=C(C1,1)+(C(C1+1,1)-C(C1,1))*(C1-INT(C1))
//     3140 X2=C(C1,2)+(C(C1+1,2)-C(C1,2))*(C1-INT(C1))
internal class Course
{
    private static readonly (int DeltaX, int DeltaY)[] cardinals = new[]
    {
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
        (1, 0),
        (1, 1),
        (0, 1)
    };

    internal Course(float direction)
    {
        if (direction < 1 || direction > 9)
        {
            throw new ArgumentOutOfRangeException(
                nameof(direction),
                direction,
                "Must be between 1 and 9, inclusive.");
        }

        var cardinalDirection = (int)(direction - 1) % 8;
        var fractionalDirection = direction - (int)direction;

        var baseCardinal = cardinals[cardinalDirection];
        var nextCardinal = cardinals[cardinalDirection + 1];

        DeltaX = baseCardinal.DeltaX + (nextCardinal.DeltaX - baseCardinal.DeltaX) * fractionalDirection;
        DeltaY = baseCardinal.DeltaY + (nextCardinal.DeltaY - baseCardinal.DeltaY) * fractionalDirection;
    }

    internal float DeltaX { get; }

    internal float DeltaY { get; }

    internal IEnumerable<Coordinates> GetSectorsFrom(Coordinates start)
    {
        (float x, float y) = start;

        while(true)
        {
            x += DeltaX;
            y += DeltaY;

            if (!Coordinates.TryCreate(x, y, out var coordinates))
            {
                yield break;
            }

            yield return coordinates;
        }
    }

    internal (bool, Coordinates, Coordinates) GetDestination(Coordinates quadrant, Coordinates sector, int distance)
    {
        var (xComplete, quadrantX, sectorX) = GetNewCoordinate(quadrant.X, sector.X, DeltaX * distance);
        var (yComplete, quadrantY, sectorY) = GetNewCoordinate(quadrant.Y, sector.Y, DeltaY * distance);

        return (xComplete && yComplete, new Coordinates(quadrantX, quadrantY), new Coordinates(sectorX, sectorY));
    }

    private static (bool, int, int) GetNewCoordinate(int quadrant, int sector, float sectorsTravelled)
    {
        var galacticCoordinate = quadrant * 8 + sector + sectorsTravelled;
        var newQuadrant = (int)(galacticCoordinate / 8);
        var newSector = (int)(galacticCoordinate - newQuadrant * 8);

        if (newSector < 0)
        {
            newQuadrant -= 1;
            newSector += 8;
        }

        return newQuadrant switch
        {
            < 0 => (false, 0, 0),
            > 7 => (false, 7, 7),
            _ => (true, newQuadrant, newSector)
        };
    }
}

```

# `84_Super_Star_Trek/csharp/Space/Galaxy.cs`

This appears to be a implementation of the Starbase simulation game. It uses C# and the Enumerable and Tuples Visual公共服务 to perform general linear algebra operations on arrays and to create星际段落（Starbase）游戏中的各个部分。游戏的主要逻辑似乎围绕着地区（Region）和子地区（SubRegion）上建立的星际段落展开。

注意：在使用 Count() 函数时，您可能需要添加一个新的分类 StarbaseCount 以便在 .NET 5 中正常工作。


```
using System.Collections.Generic;
using System.Linq;
using Games.Common.Randomness;
using SuperStarTrek.Resources;

using static System.StringSplitOptions;

namespace SuperStarTrek.Space;

internal class Galaxy
{
    private static readonly string[] _regionNames;
    private static readonly string[] _subRegionIdentifiers;
    private readonly QuadrantInfo[][] _quadrants;

    static Galaxy()
    {
        _regionNames = Strings.RegionNames.Split(new[] { ' ', '\n' }, RemoveEmptyEntries | TrimEntries);
        _subRegionIdentifiers = new[] { "I", "II", "III", "IV" };
    }

    internal Galaxy(IRandom random)
    {
        _quadrants = Enumerable
            .Range(0, 8)
            .Select(x => Enumerable
                .Range(0, 8)
                .Select(y => new Coordinates(x, y))
                .Select(c => QuadrantInfo.Create(c, GetQuadrantName(c), random))
                .ToArray())
            .ToArray();

        if (StarbaseCount == 0)
        {
            var randomQuadrant = this[random.NextCoordinate()];
            randomQuadrant.AddStarbase();

            if (randomQuadrant.KlingonCount < 2)
            {
                randomQuadrant.AddKlingon();
            }
        }
    }

    internal QuadrantInfo this[Coordinates coordinate] => _quadrants[coordinate.X][coordinate.Y];

    internal int KlingonCount => _quadrants.SelectMany(q => q).Sum(q => q.KlingonCount);

    internal int StarbaseCount => _quadrants.SelectMany(q => q).Count(q => q.HasStarbase);

    internal IEnumerable<IEnumerable<QuadrantInfo>> Quadrants => _quadrants;

    private static string GetQuadrantName(Coordinates coordinates) =>
        $"{_regionNames[coordinates.RegionIndex]} {_subRegionIdentifiers[coordinates.SubRegionIndex]}";

    internal IEnumerable<IEnumerable<QuadrantInfo>> GetNeighborhood(Quadrant quadrant) =>
        Enumerable.Range(-1, 3)
            .Select(dx => dx + quadrant.Coordinates.X)
            .Select(x => GetNeighborhoodRow(quadrant, x));
    private IEnumerable<QuadrantInfo> GetNeighborhoodRow(Quadrant quadrant, int x) =>
        Enumerable.Range(-1, 3)
            .Select(dy => dy + quadrant.Coordinates.Y)
            .Select(y => y < 0 || y > 7 || x < 0 || x > 7 ? null : _quadrants[x][y]);
}

```

# `84_Super_Star_Trek/csharp/Space/Quadrant.cs`

This is a sample implementation of the Klingon中获得全部星图控制权的功能。在游戏中，玩家需要不断地摧毁 Klingon 并获得全部星图控制权，才能在游戏中获得胜利。

注意：这个 Klingon 获得全部星图控制权的实现非常简单，仅仅实现了最基本的功能，并没有涉及到 AI、游戏逻辑等内容，所以这个程序可能比较原始。


```
using System;
using System.Collections.Generic;
using System.Linq;
using Games.Common.IO;
using Games.Common.Randomness;
using SuperStarTrek.Commands;
using SuperStarTrek.Objects;
using SuperStarTrek.Resources;

namespace SuperStarTrek.Space;

internal class Quadrant
{
    private readonly QuadrantInfo _info;
    private readonly IRandom _random;
    private readonly Dictionary<Coordinates, object> _sectors;
    private readonly Enterprise _enterprise;
    private readonly IReadWrite _io;
    private bool _entered = false;

    internal Quadrant(
        QuadrantInfo info,
        Enterprise enterprise,
        IRandom random,
        Galaxy galaxy,
        IReadWrite io)
    {
        _info = info;
        _random = random;
        _io = io;
        Galaxy = galaxy;

        info.MarkAsKnown();
        _sectors = new() { [enterprise.SectorCoordinates] = _enterprise = enterprise };
        PositionObject(sector => new Klingon(sector, _random), _info.KlingonCount);
        if (_info.HasStarbase)
        {
            Starbase = PositionObject(sector => new Starbase(sector, _random, io));
        }
        PositionObject(_ => new Star(), _info.StarCount);
    }

    internal Coordinates Coordinates => _info.Coordinates;

    internal bool HasKlingons => _info.KlingonCount > 0;

    internal int KlingonCount => _info.KlingonCount;

    internal bool HasStarbase => _info.HasStarbase;

    internal Starbase Starbase { get; }

    internal Galaxy Galaxy { get; }

    internal bool EnterpriseIsNextToStarbase =>
        _info.HasStarbase &&
        Math.Abs(_enterprise.SectorCoordinates.X - Starbase.Sector.X) <= 1 &&
        Math.Abs(_enterprise.SectorCoordinates.Y - Starbase.Sector.Y) <= 1;

    internal IEnumerable<Klingon> Klingons => _sectors.Values.OfType<Klingon>();

    public override string ToString() => _info.Name;

    private T PositionObject<T>(Func<Coordinates, T> objectFactory)
    {
        var sector = GetRandomEmptySector();
        _sectors[sector] = objectFactory.Invoke(sector);
        return (T)_sectors[sector];
    }

    private void PositionObject(Func<Coordinates, object> objectFactory, int count)
    {
        for (int i = 0; i < count; i++)
        {
            PositionObject(objectFactory);
        }
    }

    internal void Display(string textFormat)
    {
        if (!_entered)
        {
            _io.Write(textFormat, this);
            _entered = true;
        }

        if (_info.KlingonCount > 0)
        {
            _io.Write(Strings.CombatArea);
            if (_enterprise.ShieldControl.ShieldEnergy <= 200) { _io.Write(Strings.LowShields); }
        }

        _enterprise.Execute(Command.SRS);
    }

    internal bool HasObjectAt(Coordinates coordinates) => _sectors.ContainsKey(coordinates);

    internal bool TorpedoCollisionAt(Coordinates coordinates, out string message, out bool gameOver)
    {
        gameOver = false;
        message = default;

        switch (_sectors.GetValueOrDefault(coordinates))
        {
            case Klingon klingon:
                message = Remove(klingon);
                gameOver = Galaxy.KlingonCount == 0;
                return true;

            case Star _:
                message = $"Star at {coordinates} absorbed torpedo energy.";
                return true;

            case Starbase _:
                _sectors.Remove(coordinates);
                _info.RemoveStarbase();
                message = "*** Starbase destroyed ***" +
                    (Galaxy.StarbaseCount > 0 ? Strings.CourtMartial : Strings.RelievedOfCommand);
                gameOver = Galaxy.StarbaseCount == 0;
                return true;

            default:
                return false;
        }
    }

    internal string Remove(Klingon klingon)
    {
        _sectors.Remove(klingon.Sector);
        _info.RemoveKlingon();
        return "*** Klingon destroyed ***";
    }

    internal CommandResult KlingonsMoveAndFire()
    {
        foreach (var klingon in Klingons.ToList())
        {
            var newSector = GetRandomEmptySector();
            _sectors.Remove(klingon.Sector);
            _sectors[newSector] = klingon;
            klingon.MoveTo(newSector);
        }

        return KlingonsFireOnEnterprise();
    }

    internal CommandResult KlingonsFireOnEnterprise()
    {
        if (EnterpriseIsNextToStarbase && Klingons.Any())
        {
            Starbase.ProtectEnterprise();
            return CommandResult.Ok;
        }

        foreach (var klingon in Klingons)
        {
            var result = klingon.FireOn(_enterprise);
            if (result.IsGameOver) { return result; }
        }

        return CommandResult.Ok;
    }

    private Coordinates GetRandomEmptySector()
    {
        while (true)
        {
            var sector = _random.NextCoordinate();
            if (!_sectors.ContainsKey(sector))
            {
                return sector;
            }
        }
    }

    internal IEnumerable<string> GetDisplayLines() => Enumerable.Range(0, 8).Select(x => GetDisplayLine(x));

    private string GetDisplayLine(int x) =>
        string.Join(
            " ",
            Enumerable
                .Range(0, 8)
                .Select(y => new Coordinates(x, y))
                .Select(c => _sectors.GetValueOrDefault(c))
                .Select(o => o?.ToString() ?? "   "));

    internal void SetEnterpriseSector(Coordinates sector)
    {
        _sectors.Remove(_enterprise.SectorCoordinates);
        _sectors[sector] = _enterprise;
    }
}

```

# `84_Super_Star_Trek/csharp/Space/QuadrantInfo.cs`

This looks like a class in a program that simulates a Klingon太空站. It has a四元组信息类`QuadrantInfo`来记录各个位置的所有人数量以及星际争霸中拥有的星星数量等信息，并具有添加，移除和标记为已知等功能。


```
using Games.Common.Randomness;

namespace SuperStarTrek.Space;

internal class QuadrantInfo
{
    private bool _isKnown;

    private QuadrantInfo(Coordinates coordinates, string name, int klingonCount, int starCount, bool hasStarbase)
    {
        Coordinates = coordinates;
        Name = name;
        KlingonCount = klingonCount;
        StarCount = starCount;
        HasStarbase = hasStarbase;
    }

    internal Coordinates Coordinates { get; }

    internal string Name { get; }

    internal int KlingonCount { get; private set; }

    internal bool HasStarbase { get; private set; }

    internal int StarCount { get; }

    internal static QuadrantInfo Create(Coordinates coordinates, string name, IRandom random)
    {
        var klingonCount = random.NextFloat() switch
        {
            > 0.98f => 3,
            > 0.95f => 2,
            > 0.80f => 1,
            _ => 0
        };
        var hasStarbase = random.NextFloat() > 0.96f;
        var starCount = random.Next1To8Inclusive();

        return new QuadrantInfo(coordinates, name, klingonCount, starCount, hasStarbase);
    }

    internal void AddKlingon() => KlingonCount += 1;

    internal void AddStarbase() => HasStarbase = true;

    internal void MarkAsKnown() => _isKnown = true;

    internal string Scan()
    {
        _isKnown = true;
        return ToString();
    }

    public override string ToString() => _isKnown ? $"{KlingonCount}{(HasStarbase ? 1 : 0)}{StarCount}" : "***";

    internal void RemoveKlingon()
    {
        if (KlingonCount > 0)
        {
            KlingonCount -= 1;
        }
    }

    internal void RemoveStarbase() => HasStarbase = false;
}

```

# `84_Super_Star_Trek/csharp/Systems/DamageControl.cs`



这段代码是一个名为DamageControl的类，属于SuperStarTrek.Systems命名空间。

它是一个子系统，继承自Games.Common.IO.SuperStarTrek.Objects.Subsystem。

在这个类中，使用了一个Enterprise变量和一个IReadWrite变量，分别代表整个游戏的玩家引用和载波的数据库。

DamageControl类的作用是处理关于机体受损伤情况的数据，包括损伤报告、修复时间和修复后的效果。

当IsDamaged变量为true时，说明机体的损伤状态不可用，同时通过IReadWrite的WriteLine方法将"Damage Control report not available"输出到屏幕上。

当IsDamaged变量为false时，说明机体的损伤状态可用，同时通过IReadWrite的WriteLine方法输出机体状态的修复报告，每份报告包含机体名称和修复率两个部分，例如：

```
Device             State of Repair
Enterprise      Not Damaged
```

如果机体的损伤状态为true，但是该机体被绑定在船体上，则可以通过IsDocked属性判断是否可以进行修复操作。

如果可以进行修复操作，则通过IReadWrite的WriteLine方法输出机体状态的修复报告，每份报告包含机体名称和修复后的效果两个部分，例如：

```
Device             State of Repair
Enterprise      Not Damaged
```

   
   
   


```
using Games.Common.IO;
using SuperStarTrek.Commands;
using SuperStarTrek.Objects;
using SuperStarTrek.Space;

namespace SuperStarTrek.Systems;

internal class DamageControl : Subsystem
{
    private readonly Enterprise _enterprise;
    private readonly IReadWrite _io;

    internal DamageControl(Enterprise enterprise, IReadWrite io)
        : base("Damage Control", Command.DAM, io)
    {
        _enterprise = enterprise;
        _io = io;
    }

    protected override CommandResult ExecuteCommandCore(Quadrant quadrant)
    {
        if (IsDamaged)
        {
            _io.WriteLine("Damage Control report not available");
        }
        else
        {
            _io.WriteLine();
            WriteDamageReport();
        }

        if (_enterprise.DamagedSystemCount > 0 && _enterprise.IsDocked)
        {
            if (quadrant.Starbase.TryRepair(_enterprise, out var repairTime))
            {
                WriteDamageReport();
                return CommandResult.Elapsed(repairTime);
            }
        }

        return CommandResult.Ok;
    }

    internal void WriteDamageReport()
    {
        _io.WriteLine();
        _io.WriteLine("Device             State of Repair");
        foreach (var system in _enterprise.Systems)
        {
            _io.Write(system.Name.PadRight(25));
            _io.WriteLine((int)(system.Condition * 100) * 0.01F);
        }
        _io.WriteLine();
    }
}

```

# `84_Super_Star_Trek/csharp/Systems/LibraryComputer.cs`

这段代码是一个名为LibraryComputer的内部类，属于SuperStarTrek.Systems命名空间。它实现了Subsystem接口，用于管理一个游戏中的计算机。

具体来说，这段代码的作用是：

1. 创建一个图书馆计算机实例，需要一个IReadWrite的输入和一个或多个计算机函数作为参数。
2. 向计算机发送一个命令，要求计算机在命令之后处于可用状态。
3. 遍历所有计算机函数，并调用它们的Execute方法来执行指定的指令。
4. 等待用户输入，以便在计算机启动后接收更多的命令。

这段代码的实现依赖于IReadWrite组件和SuperStarTrek.Space、SuperStarTrek.Systems命名空间中定义的其他组件和类。


```
using Games.Common.IO;
using SuperStarTrek.Commands;
using SuperStarTrek.Space;
using SuperStarTrek.Systems.ComputerFunctions;

namespace SuperStarTrek.Systems;

internal class LibraryComputer : Subsystem
{
    private readonly IReadWrite _io;
    private readonly ComputerFunction[] _functions;

    internal LibraryComputer(IReadWrite io, params ComputerFunction[] functions)
        : base("Library-Computer", Command.COM, io)
    {
        _io = io;
        _functions = functions;
    }

    protected override bool CanExecuteCommand() => IsOperational("Computer disabled");

    protected override CommandResult ExecuteCommandCore(Quadrant quadrant)
    {
        var index = GetFunctionIndex();
        _io.WriteLine();

        _functions[index].Execute(quadrant);

        return CommandResult.Ok;
    }

    private int GetFunctionIndex()
    {
        while (true)
        {
            var index = (int)_io.ReadNumber("Computer active and waiting command");
            if (index >= 0 && index <= 5) { return index; }

            for (int i = 0; i < _functions.Length; i++)
            {
                _io.WriteLine($"   {i} = {_functions[i].Description}");
            }
        }
    }
}

```

# `84_Super_Star_Trek/csharp/Systems/LongRangeSensors.cs`

这段代码定义了一个名为 `LongRangeSensors` 的类，继承自 `SuperStarTrek.Commands` 和 `SuperStarTrek.Space` 命名空间中的 `Subsystem` 类。这个类负责执行长距离扫描操作，将结果存储在 `SuperStarTrek.Systems.Galaxy` 命名空间中的 `_galaxy` 变量中，同时将扫描结果存储在 `SuperStarTrek.Systems.IO` 命名空间中的 `_io` 变量中。

这个类的构造函数接收两个参数，一个 `Galaxy` 类型的参数，一个 `IReadWrite` 类型的参数，分别用于保存游戏主机的上下文和扫描结果。在构造函数中，还调用了父类的 `base` 构造函数，用于生成该类继承自哪个 `Subsystem` 类，以及该类的命令名称（即 "LRS"）和命令参数（即 "stream"）。

该类的 `CanExecuteCommand` 和 `ExecuteCommandCore` 方法分别用于检查该类是否可以执行动作以及执行动作的命令结果。在 `ExecuteCommandCore` 方法中，使用 `_io.WriteLine` 方法将扫描操作的名称和结果输出到游戏主机和扫描结果之间。然后，使用 LINQ 方法遍历游戏主机周围的 `Neighborhood` 对象，并检查每个 `Scan` 方法的结果。最后，将结果字符串连接成一个字符串并输出。

总之，这个类负责执行长距离扫描操作，并输出扫描结果。它是在游戏主机的指挥下执行操作，并在扫描完成后输出结果。


```
using System.Linq;
using Games.Common.IO;
using SuperStarTrek.Commands;
using SuperStarTrek.Space;

namespace SuperStarTrek.Systems;

internal class LongRangeSensors : Subsystem
{
    private readonly Galaxy _galaxy;
    private readonly IReadWrite _io;

    internal LongRangeSensors(Galaxy galaxy, IReadWrite io)
        : base("Long Range Sensors", Command.LRS, io)
    {
        _galaxy = galaxy;
        _io = io;
    }

    protected override bool CanExecuteCommand() => IsOperational("{name} are inoperable");

    protected override CommandResult ExecuteCommandCore(Quadrant quadrant)
    {
        _io.WriteLine($"Long range scan for quadrant {quadrant.Coordinates}");
        _io.WriteLine("-------------------");
        foreach (var quadrants in _galaxy.GetNeighborhood(quadrant))
        {
            _io.WriteLine(": " + string.Join(" : ", quadrants.Select(q => q?.Scan() ?? "***")) + " :");
            _io.WriteLine("-------------------");
        }

        return CommandResult.Ok;
    }
}


```

# `84_Super_Star_Trek/csharp/Systems/PhaserControl.cs`

This is a script written in C# for a game in which the player controls an Enterprise, and its weapons, can fire projectsiles at enemies. The script defines a `Klingon` class, which represents an enemy unit, and has methods for taking damage, firing onEnterprise, and handling damage.

The script also defines a `Quadrant` class, which represents a coordinate quarter-plane, and is used to store information about which enemies have been fired at, and assigns a strength to each fired weapon.

The script has several functions, including `ResolveHitOn`, which is called when a weapon fires at an enemy unit, and assigns a random hit strength to the enemy based on the distance to the hit point, and the strength of the hit.

The script also defines a `GetPhaserStrength`, which reads the energy available for firing weapons from the player, and a `GetPerTargetPhaserStrength`, which calculates the phaser strength of each fired weapon based on the strength of the hit and the number of targets.

The last defined function is `StartFireWeapon`, which starts firing all weapons at the enterprise, and is used by the player to start a new attack or to cancel an attack.


```
using System.Linq;
using Games.Common.IO;
using Games.Common.Randomness;
using SuperStarTrek.Commands;
using SuperStarTrek.Objects;
using SuperStarTrek.Resources;
using SuperStarTrek.Space;

namespace SuperStarTrek.Systems;

internal class PhaserControl : Subsystem
{
    private readonly Enterprise _enterprise;
    private readonly IReadWrite _io;
    private readonly IRandom _random;

    internal PhaserControl(Enterprise enterprise, IReadWrite io, IRandom random)
        : base("Phaser Control", Command.PHA, io)
    {
        _enterprise = enterprise;
        _io = io;
        _random = random;
    }

    protected override bool CanExecuteCommand() => IsOperational("Phasers inoperative");

    protected override CommandResult ExecuteCommandCore(Quadrant quadrant)
    {
        if (!quadrant.HasKlingons)
        {
            _io.WriteLine(Strings.NoEnemyShips);
            return CommandResult.Ok;
        }

        if (_enterprise.Computer.IsDamaged)
        {
            _io.WriteLine("Computer failure hampers accuracy");
        }

        _io.Write($"Phasers locked on target;  ");

        var phaserStrength = GetPhaserStrength();
        if (phaserStrength < 0) { return CommandResult.Ok; }

        _enterprise.UseEnergy(phaserStrength);

        var perEnemyStrength = GetPerTargetPhaserStrength(phaserStrength, quadrant.KlingonCount);

        foreach (var klingon in quadrant.Klingons.ToList())
        {
            ResolveHitOn(klingon, perEnemyStrength, quadrant);
        }

        return quadrant.KlingonsFireOnEnterprise();
    }

    private float GetPhaserStrength()
    {
        while (true)
        {
            _io.WriteLine($"Energy available = {_enterprise.Energy} units");
            var phaserStrength = _io.ReadNumber("Number of units to fire");

            if (phaserStrength <= _enterprise.Energy) { return phaserStrength; }
        }
    }

    private float GetPerTargetPhaserStrength(float phaserStrength, int targetCount)
    {
        if (_enterprise.Computer.IsDamaged)
        {
            phaserStrength *= _random.NextFloat();
        }

        return phaserStrength / targetCount;
    }

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

# `84_Super_Star_Trek/csharp/Systems/PhotonTubes.cs`

This is a script for a Photon Tubes subsystem in the Star Trek: Bridge Crew game. The Photon Tubes are a道具 that allow the crew to transport up to 40 photons of energy for use in battle or other紧急 situations.

The Photon Tubes class is defined as a subclass of the `Subsystem` class, which means it inherits from the `BaseSubsystem` class. This allows it to inherit any methods or functions from the `BaseSubsystem` class, as well as overriding any methods defined in its own class.

The Photon Tubes class has a single constructor that takes three parameters: the number of photons to transport, the enterprise the tubes are belonging to, and an `IReadWrite` object that will be used to write logs and information to the console.

The `CanExecuteCommand` method is a decorated version of the `CanExecuteCommand` method provided by the `BaseSubsystem` class. It checks if the tubes have any Photons left and if the tubes are currently operational. If the tubes are not operational, the method returns a `CommandResult.Ok` value.

The `ExecuteCommandCore` method is the primary method used to actually execute the command. It is overridden by the `BaseSubsystem` class and should be called by the Photon Tubes class. This method takes a `Quadrant` parameter, which represents the sector of the enterprise where the tubes will be executed.

The `ExecuteCommand` method first checks if the tubes have any Photons left. If they do, it decreases the number of photons by 1. It then sends the command to the photons to track the tubes' positions. The method then checks if any of the tubes have been hit by the photons and sends a message to the console if it has. Finally, it returns the command to the Bridge.

The `ReplenishTorpedoes` method can be used to replenish the tubes' Photons by calling the `Tubes.ReplenishTorpedoes` method. This method will replace all of the tubes' Photons with the current amount of Photons the tubes have.


```
using Games.Common.IO;
using SuperStarTrek.Commands;
using SuperStarTrek.Objects;
using SuperStarTrek.Space;

namespace SuperStarTrek.Systems;

internal class PhotonTubes : Subsystem
{
    private readonly int _tubeCount;
    private readonly Enterprise _enterprise;
    private readonly IReadWrite _io;

    internal PhotonTubes(int tubeCount, Enterprise enterprise, IReadWrite io)
        : base("Photon Tubes", Command.TOR, io)
    {
        TorpedoCount = _tubeCount = tubeCount;
        _enterprise = enterprise;
        _io = io;
    }

    internal int TorpedoCount { get; private set; }

    protected override bool CanExecuteCommand() => HasTorpedoes() && IsOperational("{name} are not operational");

    private bool HasTorpedoes()
    {
        if (TorpedoCount > 0) { return true; }

        _io.WriteLine("All photon torpedoes expended");
        return false;
    }

    protected override CommandResult ExecuteCommandCore(Quadrant quadrant)
    {
        if (!_io.TryReadCourse("Photon torpedo course", "Ensign Chekov", out var course))
        {
            return CommandResult.Ok;
        }

        TorpedoCount -= 1;

        var isHit = false;
        _io.WriteLine("Torpedo track:");
        foreach (var sector in course.GetSectorsFrom(_enterprise.SectorCoordinates))
        {
            _io.WriteLine($"                {sector}");

            if (quadrant.TorpedoCollisionAt(sector, out var message, out var gameOver))
            {
                _io.WriteLine(message);
                isHit = true;
                if (gameOver) { return CommandResult.GameOver; }
                break;
            }
        }

        if (!isHit) { _io.WriteLine("Torpedo missed!"); }

        return quadrant.KlingonsFireOnEnterprise();
    }

    internal void ReplenishTorpedoes() => TorpedoCount = _tubeCount;
}

```

# `84_Super_Star_Trek/csharp/Systems/ShieldControl.cs`

这段代码是一个名为`ShieldControl`的类，属于`SuperStarTrek.Systems`命名空间。它是一个子系统，用于控制货舰上的护盾。

具体来说，这段代码执行以下操作：

1. 初始化一个名为`_enterprise`的变量，它来自一个名为`Enterprise`的类，它是一个更大的游戏世界类，具有自己的`TotalEnergy`属性。

2. 初始化一个名为`_io`的变量，它来自一个名为`IReadWrite`的类，用于读写数据。

3. 调用父类的`base`方法，并传递一个字符串参数`Command.SHE`，以及一个包含两个参数的匿名函数`ExecuteCommandCore`，这个函数在子类中实现。

4. 在`ExecuteCommandCore`函数中执行以下操作：

 - 通过`_io.WriteLine`方法向 Enterprise 发送一条消息，指出 shield 不可用。

 - 从 `_io.ReadNumber`方法中读取一个整数参数，指定要 Shield 保护的单位数。

 - 如果 `Validate`方法返回 `true`，那么将 `ShieldEnergy` 设置为 `requested` 设置的总能量，并使用 `_io.Write` 方法向 Enterprise 发送一个消息，指定要 Shield 保护的单位数。

 - 如果 `Validate`方法返回 `false`，那么使用 `_io.WriteLine` 方法报告结果。

5. `ShieldControl` 类还实现了 `Subsystem` 和 `Command` 接口，这些接口提供更高级别的抽象和依赖关系。


```
using Games.Common.IO;
using SuperStarTrek.Commands;
using SuperStarTrek.Objects;
using SuperStarTrek.Resources;
using SuperStarTrek.Space;

namespace SuperStarTrek.Systems;

internal class ShieldControl : Subsystem
{
    private readonly Enterprise _enterprise;
    private readonly IReadWrite _io;

    internal ShieldControl(Enterprise enterprise, IReadWrite io)
        : base("Shield Control", Command.SHE, io)
    {
        _enterprise = enterprise;
        _io = io;
    }

    internal float ShieldEnergy { get; set; }

    protected override bool CanExecuteCommand() => IsOperational("{name} inoperable");

    protected override CommandResult ExecuteCommandCore(Quadrant quadrant)
    {
        _io.WriteLine($"Energy available = {_enterprise.TotalEnergy}");
        var requested = _io.ReadNumber($"Number of units to shields");

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

    private bool Validate(float requested)
    {
        if (requested > _enterprise.TotalEnergy)
        {
            _io.WriteLine("Shield Control reports, 'This is not the Federation Treasury.'");
            return false;
        }

        return requested >= 0 && requested != ShieldEnergy;
    }

    internal void AbsorbHit(int hitStrength) => ShieldEnergy -= hitStrength;

    internal void DropShields() => ShieldEnergy = 0;
}

```

# `84_Super_Star_Trek/csharp/Systems/ShortRangeSensors.cs`

This is a C# class that defines a `ShortRangeSensors` class that is part of the Star Trek: Enterprise command and control system.

This class is used to represent a device that allows for short-range sensor inputs. It is internal to the `Galaxy` and `Game` classes and has a dependency on the `IComponent` class.

The `ShortRangeSensors` class has a constructor that takes an `Enterprise` object, a `Galaxy`, and a `Game`, and an `IChangeInheritance<IComponent>` interface.

The class overrides the `ExecuteCommandCore` method to handle commands related to the short-range sensors. If the enterprise is docked, the method prints out a message indicating that the shields have been dropped. If the short-range sensors are out of commission, the method prints out a message indicating that the sensors are not functioning properly.

The class also has a method called `GetStatusLines`, which returns a list of status lines for the short-range sensors, including the stardate, condition, quadrant, sector, photon torpedoes, total energy, and shields.


```

using System;
using System.Collections.Generic;
using System.Linq;
using Games.Common.IO;
using SuperStarTrek.Commands;
using SuperStarTrek.Objects;
using SuperStarTrek.Resources;
using SuperStarTrek.Space;

namespace SuperStarTrek.Systems;

internal class ShortRangeSensors : Subsystem
{
    private readonly Enterprise _enterprise;
    private readonly Galaxy _galaxy;
    private readonly Game _game;
    private readonly IReadWrite _io;

    internal ShortRangeSensors(Enterprise enterprise, Galaxy galaxy, Game game, IReadWrite io)
        : base("Short Range Sensors", Command.SRS, io)
    {
        _enterprise = enterprise;
        _galaxy = galaxy;
        _game = game;
        _io = io;
    }

    protected override CommandResult ExecuteCommandCore(Quadrant quadrant)
    {
        if (_enterprise.IsDocked)
        {
            _io.WriteLine(Strings.ShieldsDropped);
        }

        if (Condition < 0)
        {
            _io.WriteLine(Strings.ShortRangeSensorsOut);
        }

        _io.WriteLine("---------------------------------");
        quadrant.GetDisplayLines()
            .Zip(GetStatusLines(), (sectors, status) => $" {sectors}         {status}")
            .ToList()
            .ForEach(l => _io.WriteLine(l));
        _io.WriteLine("---------------------------------");

        return CommandResult.Ok;
    }

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

```

# `84_Super_Star_Trek/csharp/Systems/Subsystem.cs`



这段代码定义了一个名为 "Subsystem" 的内部抽象类，用于表示游戏中的一个子系统。这个子系统类可以接收一个名称、一个命令和一个 IReadWrite 类型的数据流作为构造函数的参数。

在这个子系统类中，我们可以看到一个 protected 类型的变量 "Condition"，它用于判断子系统是否处于受损状态。如果条件小于零，那么这个子系统处于受损状态，这时不会输出任何信息并返回一个 CommandResult 类型的结果。如果条件大于零，子系统处于未受损状态。

另外，这个子系统类还有一个 protected 类型的变量 "Command"，它用于接收命令并对其进行执行。还有一个 protected类型的构造函数，用于将一个 Subsystem 实例初始化并注册到游戏的主进程中。

在内部抽象类 "Subsystem" 的方法中，我们可以看到 "CanExecuteCommand" 方法用于检查子系统是否处于正常状态，如果处于正常状态，那么方法返回 true，否则返回 false。另外，还有一个 "ExecuteCommand" 方法，用于执行给定的命令，并返回一个 CommandResult 类型的结果。如果子系统处于受损状态，该方法将执行修复操作并返回一个 CommandResult 类型的结果。

最后，这个子系统类有两个内部方法 "Repair" 和 "Repair"，它们分别用于修复子和修复子系统。这两个方法都有条件的判断，如果条件小于零，那么它们将使子系统处于未受损状态。如果条件大于零，那么它们将使子系统处于已修复状态。


```
using Games.Common.IO;
using SuperStarTrek.Commands;
using SuperStarTrek.Space;

namespace SuperStarTrek.Systems;

internal abstract class Subsystem
{
    private readonly IReadWrite _io;

    protected Subsystem(string name, Command command, IReadWrite io)
    {
        Name = name;
        Command = command;
        Condition = 0;
        _io = io;
    }

    internal string Name { get; }

    internal float Condition { get; private set; }

    internal bool IsDamaged => Condition < 0;

    internal Command Command { get; }

    protected virtual bool CanExecuteCommand() => true;

    protected bool IsOperational(string notOperationalMessage)
    {
        if (IsDamaged)
        {
            _io.WriteLine(notOperationalMessage.Replace("{name}", Name));
            return false;
        }

        return true;
    }

    internal CommandResult ExecuteCommand(Quadrant quadrant)
        => CanExecuteCommand() ? ExecuteCommandCore(quadrant) : CommandResult.Ok;

    protected abstract CommandResult ExecuteCommandCore(Quadrant quadrant);

    internal virtual void Repair()
    {
        if (IsDamaged)
        {
            Condition = 0;
        }
    }

    internal virtual bool Repair(float repairWorkDone)
    {
        if (IsDamaged)
        {
            Condition += repairWorkDone;
            if (Condition > -0.1f && Condition < 0)
            {
                Condition = -0.1f;
            }
        }

        return !IsDamaged;
    }

    internal void TakeDamage(float damage) => Condition -= damage;
}

```

# `84_Super_Star_Trek/csharp/Systems/WarpEngines.cs`

This is a command-line interface (CLI) script for a game in the "Homeworld" universe. It appears to be a text-based interface with some support for controller inputs, and it calls a class called "Enterprise" and a "DropShields" method.

The Enterprise class seems to have a number of methods for managing the ship's weapons and shields, and it takes inputs from the CLI to determine what actions it should take. For example, the "TryGetWarpFactor" method attempts to get the ship's warp factor by reading from the ship's "神经系统" (which appears to be a fictional device), and the "TryGetDistanceToMove" method attempts to calculate the distance to move the ship based on its warp factor and energy.

The script also seems to use a number of placeholders for things that are not yet implemented, such as "CommandResult.Elapsed" and "CommandResult.Ok".


```
using System;
using Games.Common.IO;
using SuperStarTrek.Commands;
using SuperStarTrek.Objects;
using SuperStarTrek.Resources;
using SuperStarTrek.Space;

namespace SuperStarTrek.Systems
{
    internal class WarpEngines : Subsystem
    {
        private readonly Enterprise _enterprise;
        private readonly IReadWrite _io;

        internal WarpEngines(Enterprise enterprise, IReadWrite io)
            : base("Warp Engines", Command.NAV, io)
        {
            _enterprise = enterprise;
            _io = io;
        }

        protected override CommandResult ExecuteCommandCore(Quadrant quadrant)
        {
            if (_io.TryReadCourse("Course", "   Lt. Sulu", out var course) &&
                TryGetWarpFactor(out var warpFactor) &&
                TryGetDistanceToMove(warpFactor, out var distanceToMove))
            {
                var result = quadrant.KlingonsMoveAndFire();
                if (result.IsGameOver) { return result; }

                _enterprise.RepairSystems(warpFactor);
                _enterprise.VaryConditionOfRandomSystem();
                var timeElapsed = _enterprise.Move(course, warpFactor, distanceToMove);

                if (_enterprise.IsDocked)
                {
                    _enterprise.ShieldControl.DropShields();
                    _enterprise.Refuel();
                    _enterprise.PhotonTubes.ReplenishTorpedoes();
                }

                _enterprise.Quadrant.Display(Strings.NowEntering);

                return CommandResult.Elapsed(timeElapsed);
            }

            return CommandResult.Ok;
        }

        private bool TryGetWarpFactor(out float warpFactor)
        {
            var maximumWarp = IsDamaged ? 0.2f : 8;
            if (_io.TryReadNumberInRange("Warp Factor", 0, maximumWarp, out warpFactor))
            {
                return warpFactor > 0;
            }

            _io.WriteLine(
                IsDamaged && warpFactor > maximumWarp
                    ? "Warp engines are damaged.  Maximum speed = warp 0.2"
                    : $"  Chief Engineer Scott reports, 'The engines won't take warp {warpFactor} !'");

            return false;
        }

        private bool TryGetDistanceToMove(float warpFactor, out int distanceToTravel)
        {
            distanceToTravel = (int)Math.Round(warpFactor * 8, MidpointRounding.AwayFromZero);
            if (distanceToTravel <= _enterprise.Energy) { return true; }

            _io.WriteLine("Engineering reports, 'Insufficient energy available");
            _io.WriteLine($"                      for maneuvering at warp {warpFactor} !'");

            if (distanceToTravel <= _enterprise.TotalEnergy && !_enterprise.ShieldControl.IsDamaged)
            {
                _io.Write($"Deflector control room acknowledges {_enterprise.ShieldControl.ShieldEnergy} ");
                _io.WriteLine("units of energy");
                _io.WriteLine("                         presently deployed to shields.");
            }

            return false;
        }
    }
}

```
# BasicComputerGames源码解析 78

# `84_Super_Star_Trek/csharp/Systems/ComputerFunctions/ComputerFunction.cs`



这段代码定义了一个名为 ComputerFunction 的内部类，表示计算机函数。这个内部类包含一个字符串描述符来描述函数的作用，以及一个 IReadWrite 接口的实现，用于读写数据到或从外部存储设备。

从这个内部类继承自 ComputerFunction 类，我们可以推断出 ComputerFunction 类也包含一个 string 描述符和一个 IReadWrite 接口的实现。

进一步观察可以发现，这个 ComputerFunction 类实现了一个 abstract method called Execute，这个方法是内部类的方法，但并不暴露给外部类。

所以，这段代码的作用是定义了一个 ComputerFunction 内部类，包含一个字符串描述符和一个 IReadWrite 接口的实现，但并没有公开 Execute 方法，需要通过继承来使用这个内部类的函数实现。


```
using Games.Common.IO;
using SuperStarTrek.Space;

namespace SuperStarTrek.Systems.ComputerFunctions;

internal abstract class ComputerFunction
{
    protected ComputerFunction(string description, IReadWrite io)
    {
        Description = description;
        IO = io;
    }

    internal string Description { get; }

    protected IReadWrite IO { get; }

    internal abstract void Execute(Quadrant quadrant);
}

```

# `84_Super_Star_Trek/csharp/Systems/ComputerFunctions/CumulativeGalacticRecord.cs`



这段代码是一个名为 `CumulativeGalacticRecord` 的类，属于 `SuperStarTrek.Systems.ComputerFunctions` 命名空间。它实现了 `GalacticReport` 接口，用于在游戏叙事中记录平方区域的数据。

具体来说，这个类接收两个参数：`IReadWrite io` 和 `Galaxy galaxy`。`io` 是一个用于输入和输出数据的流，` galaxy` 是用于在游戏叙事中描述平方区域的类。

类的构造函数使用 `base` 关键字来调用父类 `GalacticReport` 的构造函数，并传入所需的参数。

类的 `WriteHeader` 方法是一个重写的方法，用于写入数据头。它使用 `IO.WriteLine` 方法来输出消息，并使用字符串格式化来将日期和时间添加到行首。

类的 `GetRowData` 方法是一个重写的方法，用于返回每个单元格的数据行。它使用 `Select` 方法来遍历每个平方区域，并使用字符串格式化来将每个单元格的数据添加到字符串中。

最终，这个类实现了 `GalacticReport` 接口，用于记录平方区域的数据，并可供游戏叙事使用。


```
using System.Collections.Generic;
using System.Linq;
using Games.Common.IO;
using SuperStarTrek.Space;

namespace SuperStarTrek.Systems.ComputerFunctions;

internal class CumulativeGalacticRecord : GalacticReport
{
    internal CumulativeGalacticRecord(IReadWrite io, Galaxy galaxy)
        : base("Cumulative galactic record", io, galaxy)
    {
    }

    protected override void WriteHeader(Quadrant quadrant)
    {
        IO.WriteLine();
        IO.WriteLine($"Computer record of galaxy for quadrant {quadrant.Coordinates}");
        IO.WriteLine();
    }

    protected override IEnumerable<string> GetRowData() =>
        Galaxy.Quadrants.Select(row => " " + string.Join("   ", row));
}

```

# `84_Super_Star_Trek/csharp/Systems/ComputerFunctions/DirectionDistanceCalculator.cs`

这段代码是一个名为`DirectionDistanceCalculator`的类，属于`SuperStarTrek.Systems.ComputerFunctions`命名空间。这个类继承自另一个名为`NavigationCalculator`的类，其构造函数接收一个`Enterprise`实例和一个`IReadWrite`接口实例作为参数。

`DirectionDistanceCalculator`的作用是计算指定坐标平面内两点之间的最短距离，并返回该距离。为了实现这一目标，该类实现了一个名为`Execute`的内部方法。

该方法的实现主要分为两步：

1. 在方法调用时，先输出一段文字说明该计算器的用途，以及告知用户需要输入什么信息。

2. 然后，通过`_io`读取用户输入的两组坐标，并计算出两点之间的最短距离。为了确保输入坐标是正确的，方法还使用了`WriteDirectionAndDistance`方法将坐标分别输出并等待用户输入。

最终，这段代码的主要作用是提供一个简单的计算两点之间最短距离的函数，可以帮助游戏中处理导航等问题。


```
using Games.Common.IO;
using SuperStarTrek.Objects;
using SuperStarTrek.Space;

namespace SuperStarTrek.Systems.ComputerFunctions;

internal class DirectionDistanceCalculator : NavigationCalculator
{
    private readonly Enterprise _enterprise;
    private readonly IReadWrite _io;

    internal DirectionDistanceCalculator(Enterprise enterprise, IReadWrite io)
        : base("Direction/distance calculator", io)
    {
        _enterprise = enterprise;
        _io = io;
    }

    internal override void Execute(Quadrant quadrant)
    {
        IO.WriteLine("Direction/distance calculator:");
        IO.Write($"You are at quadrant {_enterprise.QuadrantCoordinates}");
        IO.WriteLine($" sector {_enterprise.SectorCoordinates}");
        IO.WriteLine("Please enter");

        WriteDirectionAndDistance(
            _io.GetCoordinates("  Initial coordinates"),
            _io.GetCoordinates("  Final coordinates"));
    }
}

```

# `84_Super_Star_Trek/csharp/Systems/ComputerFunctions/GalacticReport.cs`



这段代码是一个名为GalacticReport的内部类，属于SuperStarTrek.Systems.ComputerFunctions命名空间。它用于输出一个名为“SuperStarTrek.Systems.ComputerFunctions.GalacticReport”的类。

具体来说，这段代码实现了一个GalacticReport类，该类包含以下方法：

- WriteHeader：向结果写入头信息。
- GetRowData：获取结果行数据。
- Execute：执行计算。

GalacticReport类继承自ComputerFunction类，因此GalacticReport类也具有ComputerFunction的接口。ComputerFunction接口定义了以下方法：

- WriteHeader：向结果写入头信息。
- GetRowData：获取结果行数据。
- Execute：执行计算。

从GalacticReport类中可以看出，它实现了ComputerFunction接口，因此可以调用ComputerFunction的方法。


```
using System.Collections.Generic;
using System.Linq;
using Games.Common.IO;
using SuperStarTrek.Space;

namespace SuperStarTrek.Systems.ComputerFunctions;

internal abstract class GalacticReport : ComputerFunction
{
    internal GalacticReport(string description, IReadWrite io, Galaxy galaxy)
        : base(description, io)
    {
        Galaxy = galaxy;
    }

    protected Galaxy Galaxy { get; }

    protected abstract void WriteHeader(Quadrant quadrant);

    protected abstract IEnumerable<string> GetRowData();

    internal sealed override void Execute(Quadrant quadrant)
    {
        WriteHeader(quadrant);
        IO.WriteLine("       1     2     3     4     5     6     7     8");
        IO.WriteLine("     ----- ----- ----- ----- ----- ----- ----- -----");

        foreach (var (row, index) in GetRowData().Select((r, i) => (r, i)))
        {
            IO.WriteLine($" {index+1}   {row}");
            IO.WriteLine("     ----- ----- ----- ----- ----- ----- ----- -----");
        }
    }
}

```

# `84_Super_Star_Trek/csharp/Systems/ComputerFunctions/GalaxyRegionMap.cs`

这段代码定义了一个名为 `GalaxyRegionMap` 的类，继承自 `GalacticReport` 类，用于在游戏 `SuperStarTrek` 中的 `Systems` 命名空间中存储和读取 `Region` 数据。

该类实现了两个重写了自 `GalacticReport` 类的方法：

* `WriteHeader`：写入标题行，包含一个字符串，该字符串描述了整个银河系的名称。
* `GetRowData`：获取行数据，返回一个字符串数组，其中包含每个区域子名称（即每个 `Region` 数据中的区域子名称）。

`GalaxyRegionMap` 类还包括以下构造函数：

* `(IReadWrite io, Galaxy galaxy)`，用于初始化 `Region` 对象，其中 `io` 是 `SuperStarTrek.Space.IReadWrite` 类的实例，用于写入数据到文件，` galaxy` 是 `SuperStarTrek.Space` 命名空间中的一个 `Galaxy` 类实例，用于获取或设置游戏对象的引用。

另外，该类还包括以下几个内部方法：

* `Region names, Header`：将所有的区域子名称存储在 `Region names` 字符串中，并设置为该 `Region` 对象的标题行。
* `Split region names by line, TrimEnd`：将 `Region names` 字符串拆分为每个子名称，并将字符串的结尾换行符（`\r`）去除。
* `Select`：使用 LINQ 异步编程，从 `Region names` 字符串中选择所有区域子名称，并将它们存储在新的 `Region names` 字符串中。

总之，这个 `GalaxyRegionMap` 类实现了将游戏中的 `Region` 数据存储为文本文件并对其进行读取和写入的功能，是游戏开发中的一个重要组件。


```
using System.Collections.Generic;
using System.Linq;
using Games.Common.IO;
using SuperStarTrek.Resources;
using SuperStarTrek.Space;

namespace SuperStarTrek.Systems.ComputerFunctions;

internal class GalaxyRegionMap : GalacticReport
{
    internal GalaxyRegionMap(IReadWrite io, Galaxy galaxy)
        : base("Galaxy 'region name' map", io, galaxy)
    {
    }

    protected override void WriteHeader(Quadrant quadrant) =>
        IO.WriteLine("                        The Galaxy");

    protected override IEnumerable<string> GetRowData() =>
        Strings.RegionNames.Split('\n').Select(n => n.TrimEnd('\r'));
}

```

# `84_Super_Star_Trek/csharp/Systems/ComputerFunctions/NavigationCalculator.cs`

这段代码是一个名为`NavigationCalculator`的内部类，属于名为`ComputerFunction`的抽象类。这个类描述了一个计算两个坐标之间方向和距离的函数。

这个类的实现主要依赖于两个私有方法：`WriteDirectionAndDistance`和`WriteDirectionAndDistance`。这两个方法分别用于将计算得到的坐标和距离写入控制台输出流。

在`WriteDirectionAndDistance`方法中，首先通过调用私有方法`DirectionAndDistance.From`从给定的坐标`from`计算得到方向和距离。然后，使用这两个坐标计算从`from`到`to`的 direction。最后，将计算得到的`direction`和`distance`分别写入控制台输出流。

整段代码的主要作用是创建一个可以计算两个坐标之间方向和距离的函数，该函数可以被其他类或游戏中的计算机所调用。


```
using Games.Common.IO;
using SuperStarTrek.Space;
using SuperStarTrek.Utils;

namespace SuperStarTrek.Systems.ComputerFunctions;

internal abstract class NavigationCalculator : ComputerFunction
{
    protected NavigationCalculator(string description, IReadWrite io)
        : base(description, io)
    {
    }

    protected void WriteDirectionAndDistance(Coordinates from, Coordinates to)
    {
        var (direction, distance) = from.GetDirectionAndDistanceTo(to);
        Write(direction, distance);
    }

    protected void WriteDirectionAndDistance((float X, float Y) from, (float X, float Y) to)
    {
        var (direction, distance) = DirectionAndDistance.From(from.X, from.Y).To(to.X, to.Y);
        Write(direction, distance);
    }

    private void Write(float direction, float distance)
    {
        IO.WriteLine($"Direction = {direction}");
        IO.WriteLine($"Distance = {distance}");
    }
}

```

# `84_Super_Star_Trek/csharp/Systems/ComputerFunctions/StarbaseDataCalculator.cs`

这段代码定义了一个名为 `StarbaseDataCalculator` 的类，继承自 `NavigationCalculator` 类，用于在玩家控制的游戏引擎中处理星际舰队的计算。

在这个内部类中，我们创建了一个 `StarbaseDataCalculator` 类的实例，并使用它继承自父类的 `Execute` 方法。

在 `Execute` 方法中，我们首先检查当前是否正在一个星际舰队的视图下，如果不是，我们输出一条消息并返回。

否则，我们输出一个消息，指出我们正在从星际舰队的当前位置到指定位置，然后写入方向和距离，从星际舰队的当前位置到指定位置。


```
using Games.Common.IO;
using SuperStarTrek.Objects;
using SuperStarTrek.Resources;
using SuperStarTrek.Space;

namespace SuperStarTrek.Systems.ComputerFunctions;

internal class StarbaseDataCalculator : NavigationCalculator
{
    private readonly Enterprise _enterprise;

    internal StarbaseDataCalculator(Enterprise enterprise, IReadWrite io)
        : base("Starbase nav data", io)
    {
        _enterprise = enterprise;
    }

    internal override void Execute(Quadrant quadrant)
    {
        if (!quadrant.HasStarbase)
        {
            IO.WriteLine(Strings.NoStarbase);
            return;
        }

        IO.WriteLine("From Enterprise to Starbase:");

        WriteDirectionAndDistance(_enterprise.SectorCoordinates, quadrant.Starbase.Sector);
    }
}

```

# `84_Super_Star_Trek/csharp/Systems/ComputerFunctions/StatusReport.cs`



这段代码是一个名为 "StatusReport" 的类，属于 "SuperStarTrek.Systems.ComputerFunctions" 命名空间。这个类的作用是在游戏或太空游戏中生成状态报告，并显示一些信息，如星际飞船的星基数量、联邦在星球上的数量以及当前游戏或太空游戏中剩余的星际飞船数量。

这个类有一个内部类 "Execute"，这个内部类实现了 "ComputerFunction" 基类，因此可以访问到 "Execute" 方法。

在 "Execute" 方法中，首先输出当前游戏或太空游戏的剩余星际飞船数量，然后输出 "The Federation" 如何在游戏中或太空游戏中维护的星基数量。如果当前有星基，这个类将输出 "starbase" 数量。如果当前没有星基，这个类将输出 "Your stupidity has left you on your own in the galaxy -- you have no starbases left!" 消息。

最后，这个 "StatusReport" 类还通过 "Execute" 方法调用 "Enterprise.Execute" 方法，这个方法没有定义，因此可以推断出它可能是指挥 "Enterprise" 星舰的命令。


```
using Games.Common.IO;
using SuperStarTrek.Commands;
using SuperStarTrek.Objects;
using SuperStarTrek.Space;

namespace SuperStarTrek.Systems.ComputerFunctions;

internal class StatusReport : ComputerFunction
{
    private readonly Game _game;
    private readonly Galaxy _galaxy;
    private readonly Enterprise _enterprise;

    internal StatusReport(Game game, Galaxy galaxy, Enterprise enterprise, IReadWrite io)
        : base("Status report", io)
    {
        _game = game;
        _galaxy = galaxy;
        _enterprise = enterprise;
    }

    internal override void Execute(Quadrant quadrant)
    {
        IO.WriteLine("   Status report:");
        IO.Write("Klingon".Pluralize(_galaxy.KlingonCount));
        IO.WriteLine($" left:  {_galaxy.KlingonCount}");
        IO.WriteLine($"Mission must be completed in {_game.StardatesRemaining:0.#} stardates.");

        if (_galaxy.StarbaseCount > 0)
        {
            IO.Write($"The Federation is maintaining {_galaxy.StarbaseCount} ");
            IO.Write("starbase".Pluralize(_galaxy.StarbaseCount));
            IO.WriteLine(" in the galaxy.");
        }
        else
        {
            IO.WriteLine("Your stupidity has left you on your own in");
            IO.WriteLine("  the galaxy -- you have no starbases left!");
        }

        _enterprise.Execute(Command.DAM);
    }
}

```

# `84_Super_Star_Trek/csharp/Systems/ComputerFunctions/TorpedoDataCalculator.cs`

这段代码是一个名为TorpedoDataCalculator的内部类，属于SuperStarTrek.Systems.ComputerFunctions命名空间。它继承自另一个内部类NavigationCalculator，但TorpedoDataCalculator负责计算Torpedo的数据，包括从Enterprise船前往Klingon战巡的路径。

具体来说，这段代码的作用是计算从Enterprise船到Klingon战巡的路径，并输出一条相应的消息。在计算路径时，如果目标没有Klingon，则输出“No enemy ships”，否则会递归计算每个Klingon船的位置。计算路径的具体实现包括IO的编写。


```
using Games.Common.IO;
using SuperStarTrek.Objects;
using SuperStarTrek.Resources;
using SuperStarTrek.Space;

namespace SuperStarTrek.Systems.ComputerFunctions;

internal class TorpedoDataCalculator : NavigationCalculator
{
    private readonly Enterprise _enterprise;

    internal TorpedoDataCalculator(Enterprise enterprise, IReadWrite io)
        : base("Photon torpedo data", io)
    {
        _enterprise = enterprise;
    }

    internal override void Execute(Quadrant quadrant)
    {
        if (!quadrant.HasKlingons)
        {
            IO.WriteLine(Strings.NoEnemyShips);
            return;
        }

        IO.WriteLine("From Enterprise to Klingon battle cruiser".Pluralize(quadrant.KlingonCount));

        foreach (var klingon in quadrant.Klingons)
        {
            WriteDirectionAndDistance(_enterprise.SectorCoordinates, klingon.Sector);
        }
    }
}

```

# `84_Super_Star_Trek/csharp/Utils/DirectionAndDistance.cs`

This appears to be a code snippet written in Assembly language for the Microchip PIC Microcontroller. It defines a `GetDirection()` function and a `GetDistance()` function, both of which take in two parameters: `deltaX` and `deltaY`, and return values of type `float`.

The `GetDirection()` function appears to calculate the direction of the line connecting two points `(deltaX, deltaY)` and `(deltaX, deltaY)` in the direction of the line connecting them. The function uses the ratio of the distances from the two points to the distance between them to calculate the direction. The direction is returned as an incremental value, with a bias towards the left if `deltaX` is greater than `deltaY`, and a bias towards


```
using System;
using SuperStarTrek.Space;

namespace SuperStarTrek.Utils
{
    internal class DirectionAndDistance
    {
        private readonly float _fromX;
        private readonly float _fromY;

        private DirectionAndDistance(float fromX, float fromY)
        {
            _fromX = fromX;
            _fromY = fromY;
        }

        internal static DirectionAndDistance From(Coordinates coordinates) => From(coordinates.X, coordinates.Y);

        internal static DirectionAndDistance From(float x, float y) => new DirectionAndDistance(x, y);

        internal (float Direction, float Distance) To(Coordinates coordinates) => To(coordinates.X, coordinates.Y);

        internal (float Direction, float Distance) To(float x, float y)
        {
            var deltaX = x - _fromX;
            var deltaY = y - _fromY;

            return (GetDirection(deltaX, deltaY), GetDistance(deltaX, deltaY));
        }

        // The algorithm here is mathematically equivalent to the following code in the original,
        // where X is deltaY and A is deltaX
        //     8220 X=X-A:A=C1-W1:IFX<0THEN8350
        //     8250 IFA<0THEN8410
        //     8260 IFX>0THEN8280
        //     8270 IFA=0THENC1=5:GOTO8290
        //     8280 C1=1
        //     8290 IFABS(A)<=ABS(X)THEN8330
        //     8310 PRINT"DIRECTION =";C1+(((ABS(A)-ABS(X))+ABS(A))/ABS(A)):GOTO8460
        //     8330 PRINT"DIRECTION =";C1+(ABS(A)/ABS(X)):GOTO8460
        //     8350 IFA>0THENC1=3:GOTO8420
        //     8360 IFX<>0THENC1=5:GOTO8290
        //     8410 C1=7
        //     8420 IFABS(A)>=ABS(X)THEN8450
        //     8430 PRINT"DIRECTION =";C1+(((ABS(X)-ABS(A))+ABS(X))/ABS(X)):GOTO8460
        //     8450 PRINT"DIRECTION =";C1+(ABS(X)/ABS(A))
        //     8460 PRINT"DISTANCE =";SQR(X^2+A^2):IFH8=1THEN1990
        private static float GetDirection(float deltaX, float deltaY)
        {
            var deltaXDominant = Math.Abs(deltaX) > Math.Abs(deltaY);
            var fractionalPart = deltaXDominant ? deltaY / deltaX : -deltaX / deltaY;
            var nearestCardinal = deltaXDominant switch
            {
                true => deltaX > 0 ? 7 : 3,
                false => deltaY > 0 ? 1 : 5
            };

            var direction = nearestCardinal + fractionalPart;
            return direction < 1 ? direction + 8 : direction;
        }

        private static float GetDistance(float deltaX, float deltaY) =>
            (float)Math.Sqrt(Math.Pow(deltaX, 2) + Math.Pow(deltaY, 2));
    }
}

```

# `84_Super_Star_Trek/java/Enterprise.java`

This is a program that appears to simulate a repair system for a ship. It is written in Java and uses a class called `RepairComputer` that contains methods for repairing the ship's computer.

The `RepairComputer` class has a method called `repairDevice()` that takes an integer representing the device number and an integer representing the repair cost. The method updates the device status and returns a message indicating whether the repair was successful.

The `repairDevice()` method first checks if the device number is equal to `DEVICE_WARP_ENGINES`, `DEVICE_SHORT_RANGE_SENSORS`, `DEVICE_LONG_RANGE_SENSORS`, `DEVICE_PHASER_CONTROL`, or `DEVICE_PHOTON_TUBES`. If the device number is not one of these, the method does nothing and returns a message. If the device number is equal to `DEVICE_WARP_ENGINES`, the method updates the device status and returns a message indicating that the repair was successful. If the device number is equal to `DEVICE_SHORT_RANGE_SENSORS`, the method updates the device status and returns a message indicating that the repair was successful. If the device number is equal to `DEVICE_LONG_RANGE_SENSORS`, the method updates the device status and returns a message indicating that the repair was successful. If the device number is equal to `DEVICE_PHASER_CONTROL`, the method updates the device status and returns a message indicating that the repair was successful. If the device number is equal to `DEVICE_PHOTON_TUBES`, the method updates the device status and returns a message indicating that the repair was successful. If the device number is equal to `DEVICE_DAMAGE_CONTROL`, the method updates the device status and returns a message indicating that the repair was successful. If the device number is equal to `DEVICE_LIBRARY_COMPUTER`, the method updates the device status and returns a message indicating that the repair was successful.

The `printDeviceName()` method takes an integer representing the device number and returns a string indicating the name of the device. This is used to display a message indicating the name of the device when the repair is complete.


```
import java.util.stream.IntStream;

/**
 * The starship Enterprise.
 */
public class Enterprise {

    public static final int COORD_X = 0;
    public static final int COORD_Y = 1;

    // devices
    static final int DEVICE_WARP_ENGINES = 1;
    static final int DEVICE_SHORT_RANGE_SENSORS = 2;
    static final int DEVICE_LONG_RANGE_SENSORS = 3;
    static final int DEVICE_PHASER_CONTROL = 4;
    static final int DEVICE_PHOTON_TUBES = 5;
    static final int DEVICE_DAMAGE_CONTROL = 6;
    static final int DEVICE_SHIELD_CONTROL = 7;
    static final int DEVICE_LIBRARY_COMPUTER = 8;
    final double[] deviceStatus = new double[9];   // 8  device damage stats

    // position
    final int[][] cardinalDirections = new int[10][3];   // 9x2 vectors in cardinal directions
    int quadrantX;
    int quadrantY;
    int sectorX;
    int sectorY;

    // ship status
    boolean docked = false;
    int energy = 3000;
    int torpedoes = 10;
    int shields = 0;
    double repairCost;

    final int initialEnergy = energy;
    final int initialTorpedoes = torpedoes;

    public Enterprise() {
        // random initial position
        this.setQuadrant(new int[]{ Util.fnr(), Util.fnr() });
        this.setSector(new int[]{ Util.fnr(), Util.fnr() });
        // init cardinal directions
        IntStream.range(1, 9).forEach(i -> {
            cardinalDirections[i][1] = 0;
            cardinalDirections[i][2] = 0;
        });
        cardinalDirections[3][1] = -1;
        cardinalDirections[2][1] = -1;
        cardinalDirections[4][1] = -1;
        cardinalDirections[4][2] = -1;
        cardinalDirections[5][2] = -1;
        cardinalDirections[6][2] = -1;
        cardinalDirections[1][2] = 1;
        cardinalDirections[2][2] = 1;
        cardinalDirections[6][1] = 1;
        cardinalDirections[7][1] = 1;
        cardinalDirections[8][1] = 1;
        cardinalDirections[8][2] = 1;
        cardinalDirections[9][2] = 1;
        // init devices
        IntStream.range(1, 8).forEach(i -> deviceStatus[i] = 0);
    }

    public int getShields() {
        return shields;
    }

    /**
     * Enterprise is hit by enemies.
     * @param hits the number of hit points
     */
    public void sufferHitPoints(int hits) {
        this.shields = shields - hits;
    }

    public int getEnergy() {
        return energy;
    }

    public void replenishSupplies() {
        this.energy = this.initialEnergy;
        this.torpedoes = this.initialTorpedoes;
    }

    public void decreaseEnergy(final double amount) {
        this.energy -= amount;
    }

    public void decreaseTorpedoes(final int amount) {
        torpedoes -= amount;
    }

    public void dropShields() {
        this.shields = 0;
    }

    public int getTotalEnergy() {
        return (shields + energy);
    }

    public int getInitialEnergy() {
        return initialEnergy;
    }

    public int getTorpedoes() {
        return torpedoes;
    }

    public double[] getDeviceStatus() {
        return deviceStatus;
    }

    public int[][] getCardinalDirections() {
        return cardinalDirections;
    }

    public void setDeviceStatus(final int device, final double status) {
        this.deviceStatus[device] = status;
    }

    public boolean isDocked() {
        return docked;
    }

    public void setDocked(boolean docked) {
        this.docked = docked;
    }

    public int[] getQuadrant() {
        return new int[] {quadrantX, quadrantY};
    }

    public void setQuadrant(final int[] quadrant) {
        this.quadrantX = quadrant[COORD_X];
        this.quadrantY = quadrant[COORD_Y];
    }

    public int[] getSector() {
        return new int[] {sectorX, sectorY};
    }

    public void setSector(final int[] sector) {
        this.sectorX = sector[COORD_X];
        this.sectorY = sector[COORD_Y];
    }

    public int[] moveShip(final float course, final int n, final String quadrantMap, final double stardate, final double initialStardate, final int missionDuration, final GameCallback callback) {
        int ic1 = Util.toInt(course);
        float x1 = cardinalDirections[ic1][1] + (cardinalDirections[ic1 + 1][1] - cardinalDirections[ic1][1]) * (course - ic1);
        float x = sectorX;
        float y = sectorY;
        float x2 = cardinalDirections[ic1][2] + (cardinalDirections[ic1 + 1][2] - cardinalDirections[ic1][2]) * (course - ic1);
        final int initialQuadrantX = quadrantX;
        final int initialQuadrantY = quadrantY;
        for (int i = 1; i <= n; i++) {
            sectorX += x1;
            sectorY += x2;
            if (sectorX < 1 || sectorX >= 9 || sectorY < 1 || sectorY >= 9) {
                // exceeded quadrant limits
                x = 8 * quadrantX + x + n * x1;
                y = 8 * quadrantY + y + n * x2;
                quadrantX = Util.toInt(x / 8);
                quadrantY = Util.toInt(y / 8);
                sectorX = Util.toInt(x - quadrantX * 8);
                sectorY = Util.toInt(y - quadrantY * 8);
                if (sectorX == 0) {
                    quadrantX = quadrantX - 1;
                    sectorX = 8;
                }
                if (sectorY == 0) {
                    quadrantY = quadrantY - 1;
                    sectorY = 8;
                }
                boolean hitEdge = false;
                if (quadrantX < 1) {
                    hitEdge = true;
                    quadrantX = 1;
                    sectorX = 1;
                }
                if (quadrantX > 8) {
                    hitEdge = true;
                    quadrantX = 8;
                    sectorX = 8;
                }
                if (quadrantY < 1) {
                    hitEdge = true;
                    quadrantY = 8;
                    sectorY = 8;
                }
                if (quadrantY > 8) {
                    hitEdge = true;
                    quadrantY = 8;
                    sectorY = 8;
                }
                if (hitEdge) {
                    Util.println("LT. UHURA REPORTS MESSAGE FROM STARFLEET COMMAND:");
                    Util.println("  'PERMISSION TO ATTEMPT CROSSING OF GALACTIC PERIMETER");
                    Util.println("  IS HEREBY *DENIED*.  SHUT DOWN YOUR ENGINES.'");
                    Util.println("CHIEF ENGINEER SCOTT REPORTS  'WARP ENGINES SHUT DOWN");
                    Util.println("  AT SECTOR " + sectorX + "," + sectorY + " OF QUADRANT " + quadrantX + "," + quadrantY + ".'");
                    if (stardate > initialStardate + missionDuration) callback.endGameFail(false);
                }
                if (8 * quadrantX + quadrantY == 8 * initialQuadrantX + initialQuadrantY) {
                    break;
                }
                callback.incrementStardate(1);
                maneuverEnergySR(n);
                callback.enterNewQuadrant();
                return this.getSector();
            } else {
                int S8 = Util.toInt(sectorX) * 24 + Util.toInt(sectorY) * 3 - 26; // S8 = pos
                if (!("  ".equals(Util.midStr(quadrantMap, S8, 2)))) {
                    sectorX = Util.toInt(sectorX - x1);
                    sectorY = Util.toInt(sectorY - x2);
                    Util.println("WARP ENGINES SHUT DOWN AT ");
                    Util.println("SECTOR " + sectorX + "," + sectorY + " DUE TO BAD NAVIGATION");
                    break;
                }
            }
        }
        sectorX = Util.toInt(sectorX);
        sectorY = Util.toInt(sectorY);
        return this.getSector();
    }

    void randomRepairCost() {
        repairCost = .5 * Util.random();
    }

    public void repairDamagedDevices(final float warp) {
        // repair damaged devices and print damage report
        for (int i = 1; i <= 8; i++) {
            if (deviceStatus[i] < 0) {
                deviceStatus[i] += Math.min(warp, 1);
                if ((deviceStatus[i] > -.1) && (deviceStatus[i] < 0)) {
                    deviceStatus[i] = -.1;
                    break;
                } else if (deviceStatus[i] >= 0) {
                    Util.println("DAMAGE CONTROL REPORT:  ");
                    Util.println(Util.tab(8) + printDeviceName(i) + " REPAIR COMPLETED.");
                }
            }
        }
    }

    public void maneuverEnergySR(final int N) {
        energy = energy - N - 10;
        if (energy >= 0) return;
        Util.println("SHIELD CONTROL SUPPLIES ENERGY TO COMPLETE THE MANEUVER.");
        shields = shields + energy;
        energy = 0;
        if (shields <= 0) shields = 0;
    }

    void shieldControl() {
        if (deviceStatus[DEVICE_SHIELD_CONTROL] < 0) {
            Util.println("SHIELD CONTROL INOPERABLE");
            return;
        }
        Util.println("ENERGY AVAILABLE = " + (energy + shields));
        int energyToShields = Util.toInt(Util.inputFloat("NUMBER OF UNITS TO SHIELDS"));
        if (energyToShields < 0 || shields == energyToShields) {
            Util.println("<SHIELDS UNCHANGED>");
            return;
        }
        if (energyToShields > energy + energyToShields) {
            Util.println("SHIELD CONTROL REPORTS  'THIS IS NOT THE FEDERATION TREASURY.'");
            Util.println("<SHIELDS UNCHANGED>");
            return;
        }
        energy = energy + shields - energyToShields;
        shields = energyToShields;
        Util.println("DEFLECTOR CONTROL ROOM REPORT:");
        Util.println("  'SHIELDS NOW AT " + Util.toInt(shields) + " UNITS PER YOUR COMMAND.'");
    }

    void damageControl(GameCallback callback) {
        if (deviceStatus[DEVICE_DAMAGE_CONTROL] < 0) {
            Util.println("DAMAGE CONTROL REPORT NOT AVAILABLE");
        } else {
            Util.println("\nDEVICE             STATE OF REPAIR");
            for (int deviceNr = 1; deviceNr <= 8; deviceNr++) {
                Util.print(printDeviceName(deviceNr) + Util.leftStr(GalaxyMap.QUADRANT_ROW, 25 - Util.strlen(printDeviceName(deviceNr))) + " " + Util.toInt(deviceStatus[deviceNr] * 100) * .01 + "\n");
            }
        }
        if (!docked) return;

        double deltaToRepair = 0;
        for (int i = 1; i <= 8; i++) {
            if (deviceStatus[i] < 0) deltaToRepair += .1;
        }
        if (deltaToRepair > 0) {
            deltaToRepair += repairCost;
            if (deltaToRepair >= 1) deltaToRepair = .9;
            Util.println("TECHNICIANS STANDING BY TO EFFECT REPAIRS TO YOUR SHIP;");
            Util.println("ESTIMATED TIME TO REPAIR:'" + .01 * Util.toInt(100 * deltaToRepair) + " STARDATES");
            final String reply = Util.inputStr("WILL YOU AUTHORIZE THE REPAIR ORDER (Y/N)");
            if ("Y".equals(reply)) {
                for (int deviceNr = 1; deviceNr <= 8; deviceNr++) {
                    if (deviceStatus[deviceNr] < 0) deviceStatus[deviceNr] = 0;
                }
                callback.incrementStardate(deltaToRepair + .1);
            }
        }
    }

    public static String printDeviceName(final int deviceNumber) {  // 8790
        // PRINTS DEVICE NAME
        switch (deviceNumber) {
            case DEVICE_WARP_ENGINES:
                return "WARP ENGINES";
            case DEVICE_SHORT_RANGE_SENSORS:
                return "SHORT RANGE SENSORS";
            case DEVICE_LONG_RANGE_SENSORS:
                return "LONG RANGE SENSORS";
            case DEVICE_PHASER_CONTROL:
                return "PHASER CONTROL";
            case DEVICE_PHOTON_TUBES:
                return "PHOTON TUBES";
            case DEVICE_DAMAGE_CONTROL:
                return "DAMAGE CONTROL";
            case DEVICE_SHIELD_CONTROL:
                return "SHIELD CONTROL";
            case DEVICE_LIBRARY_COMPUTER:
                return "LIBRARY-COMPUTER";
        }
        return "";
    }

}

```

# `84_Super_Star_Trek/java/GalaxyMap.java`

This appears to be a Java class that manages a game board. It contains a function called `compareMarker` that compares a marker string to the string `MARKER_EMPTY`, which indicates that the marker should be considered an empty space. If the marker string does not match `MARKER_EMPTY`, the function returns `true`, otherwise it returns `false`.

The class also contains a function called `findEmptyPlaceInQuadrant`, which finds random empty coordinates in a quadrant of the game board. It takes a `quadrantString` argument, which is a string that describes the quadrant, and returns an array with two coordinates `x` and `y` representing the empty space in the quadrant. If the function cannot find an empty space within the quadrant, it returns an array with two coordinates representing the empty space in the original game board.

The class also contains a `shuffleObjects` function that shuffles the contents of a list of objects and returns the first object after shuffling. It is not clear what this function is intended to be used for.

I'm sorry, but I am unable to provide any additional information or context about this class, as it appears to be complete and self-contained.


```
import java.util.stream.IntStream;

/**
 * Map of the galaxy divided in Quadrants and Sectors,
 * populated with stars, starbases, klingons, and the Enterprise.
 */
public class GalaxyMap {

    // markers
    static final String MARKER_EMPTY = "   ";
    static final String MARKER_ENTERPRISE = "<*>";
    static final String MARKER_KLINGON = "+K+";
    static final String MARKER_STARBASE = ">!<";
    static final String MARKER_STAR = " * ";

    static final int AVG_KLINGON_SHIELD_ENERGY = 200;

    // galaxy map
    public static final String QUADRANT_ROW = "                         ";
    String quadrantMap = QUADRANT_ROW + QUADRANT_ROW + QUADRANT_ROW + QUADRANT_ROW + QUADRANT_ROW + QUADRANT_ROW + QUADRANT_ROW + Util.leftStr(QUADRANT_ROW, 17);       // current quadrant map
    final int[][] galaxy = new int[9][9];    // 8x8 galaxy map G
    final int[][] klingonQuadrants = new int[4][4];    // 3x3 position of klingons K
    final int[][] chartedGalaxy = new int[9][9];    // 8x8 charted galaxy map Z

    // galaxy state
    int basesInGalaxy = 0;
    int remainingKlingons;
    int klingonsInGalaxy = 0;
    final Enterprise enterprise = new Enterprise();

    // quadrant state
    int klingons = 0;
    int starbases = 0;
    int stars = 0;
    int starbaseX = 0; // X coordinate of starbase
    int starbaseY = 0; // Y coord of starbase

    public Enterprise getEnterprise() {
        return enterprise;
    }

    public int getBasesInGalaxy() {
        return basesInGalaxy;
    }

    public int getRemainingKlingons() {
        return remainingKlingons;
    }

    public int getKlingonsInGalaxy() {
        return klingonsInGalaxy;
    }

    double fnd(int i) {
        return Math.sqrt((klingonQuadrants[i][1] - enterprise.getSector()[Enterprise.COORD_X]) ^ 2 + (klingonQuadrants[i][2] - enterprise.getSector()[Enterprise.COORD_Y]) ^ 2);
    }

    public GalaxyMap() {
        int quadrantX = enterprise.getQuadrant()[Enterprise.COORD_X];
        int quadrantY = enterprise.getQuadrant()[Enterprise.COORD_Y];
        // populate Klingons, Starbases, Stars
        IntStream.range(1, 8).forEach(x -> {
            IntStream.range(1, 8).forEach(y -> {
                klingons = 0;
                chartedGalaxy[x][y] = 0;
                float random = Util.random();
                if (random > .98) {
                    klingons = 3;
                    klingonsInGalaxy += 3;
                } else if (random > .95) {
                    klingons = 2;
                    klingonsInGalaxy += 2;
                } else if (random > .80) {
                    klingons = 1;
                    klingonsInGalaxy += 1;
                }
                starbases = 0;
                if (Util.random() > .96) {
                    starbases = 1;
                    basesInGalaxy = +1;
                }
                galaxy[x][y] = klingons * 100 + starbases * 10 + Util.fnr();
            });
        });
        if (basesInGalaxy == 0) {
            if (galaxy[quadrantX][quadrantY] < 200) {
                galaxy[quadrantX][quadrantY] = galaxy[quadrantX][quadrantY] + 120;
                klingonsInGalaxy = +1;
            }
            basesInGalaxy = 1;
            galaxy[quadrantX][quadrantY] = +10;
            enterprise.setQuadrant(new int[]{ Util.fnr(), Util.fnr() });
        }
        remainingKlingons = klingonsInGalaxy;
    }

    void newQuadrant(final double stardate, final double initialStardate) {   // 1320
        final int quadrantX = enterprise.getQuadrant()[Enterprise.COORD_X];
        final int quadrantY = enterprise.getQuadrant()[Enterprise.COORD_Y];
        klingons = 0;
        starbases = 0;
        stars = 0;
        enterprise.randomRepairCost();
        chartedGalaxy[quadrantX][quadrantY] = galaxy[quadrantX][quadrantY];
        if (!(quadrantX < 1 || quadrantX > 8 || quadrantY < 1 || quadrantY > 8)) {
            final String quadrantName = getQuadrantName(false, quadrantX, quadrantY);
            if (initialStardate == stardate) {
                Util.println("YOUR MISSION BEGINS WITH YOUR STARSHIP LOCATED\n" +
                        "IN THE GALACTIC QUADRANT, '" + quadrantName + "'.");
            } else {
                Util.println("NOW ENTERING " + quadrantName + " QUADRANT . . .");
            }
            Util.println("");
            klingons = (int) Math.round(galaxy[quadrantX][quadrantY] * .01);
            starbases = (int) Math.round(galaxy[quadrantX][quadrantY] * .1) - 10 * klingons;
            stars = galaxy[quadrantX][quadrantY] - 100 * klingons - 10 * starbases;
            if (klingons != 0) {
                Util.println("COMBAT AREA      CONDITION RED");
                if (enterprise.getShields() <= 200) {
                    Util.println("   SHIELDS DANGEROUSLY LOW");
                }
            }
            IntStream.range(1, 3).forEach(i -> {
                klingonQuadrants[i][1] = 0;
                klingonQuadrants[i][2] = 0;
            });
        }
        IntStream.range(1, 3).forEach(i -> {
            klingonQuadrants[i][3] = 0;
        });
        // position enterprise in quadrant
        insertMarker(MARKER_ENTERPRISE, enterprise.getSector()[Enterprise.COORD_X], enterprise.getSector()[Enterprise.COORD_Y]);
        // position klingons
        if (klingons >= 1) {
            for (int i = 1; i <= klingons; i++) {
                final int[] emptyCoordinate = findEmptyPlaceInQuadrant(quadrantMap);
                insertMarker(MARKER_KLINGON, emptyCoordinate[0], emptyCoordinate[1]);
                klingonQuadrants[i][1] = emptyCoordinate[0];
                klingonQuadrants[i][2] = emptyCoordinate[1];
                klingonQuadrants[i][3] = (int) Math.round(AVG_KLINGON_SHIELD_ENERGY * (0.5 + Util.random()));
            }
        }
        // position bases
        if (starbases >= 1) {
            final int[] emptyCoordinate = findEmptyPlaceInQuadrant(quadrantMap);
            starbaseX = emptyCoordinate[0];
            starbaseY = emptyCoordinate[1];
            insertMarker(MARKER_STARBASE, emptyCoordinate[0], emptyCoordinate[1]);
        }
        // position stars
        for (int i = 1; i <= stars; i++) {
            final int[] emptyCoordinate = findEmptyPlaceInQuadrant(quadrantMap);
            insertMarker(MARKER_STAR, emptyCoordinate[0], emptyCoordinate[1]);
        }
    }

    public void klingonsMoveAndFire(GameCallback callback) {
        for (int i = 1; i <= klingons; i++) {
            if (klingonQuadrants[i][3] == 0) continue;
            insertMarker(MARKER_EMPTY, klingonQuadrants[i][1], klingonQuadrants[i][2]);
            final int[] newCoords = findEmptyPlaceInQuadrant(quadrantMap);
            klingonQuadrants[i][1] = newCoords[0];
            klingonQuadrants[i][2] = newCoords[1];
            insertMarker(MARKER_KLINGON, klingonQuadrants[i][1], klingonQuadrants[i][2]);
        }
        klingonsShoot(callback);
    }

    void klingonsShoot(GameCallback callback) {
        if (klingons <= 0) return; // no klingons
        if (enterprise.isDocked()) {
            Util.println("STARBASE SHIELDS PROTECT THE ENTERPRISE");
            return;
        }
        for (int i = 1; i <= 3; i++) {
            if (klingonQuadrants[i][3] <= 0) continue;
            int hits = Util.toInt((klingonQuadrants[i][3] / fnd(1)) * (2 + Util.random()));
            enterprise.sufferHitPoints(hits);
            klingonQuadrants[i][3] = Util.toInt(klingonQuadrants[i][3] / (3 + Util.random()));
            Util.println(hits + " UNIT HIT ON ENTERPRISE FROM SECTOR " + klingonQuadrants[i][1] + "," + klingonQuadrants[i][2]);
            if (enterprise.getShields() <= 0) callback.endGameFail(true);
            Util.println("      <SHIELDS DOWN TO " + enterprise.getShields() + " UNITS>");
            if (hits < 20) continue;
            if ((Util.random() > .6) || (hits / enterprise.getShields() <= .02)) continue;
            int randomDevice = Util.fnr();
            enterprise.setDeviceStatus(randomDevice, enterprise.getDeviceStatus()[randomDevice]- hits / enterprise.getShields() - .5 * Util.random());
            Util.println("DAMAGE CONTROL REPORTS " + Enterprise.printDeviceName(randomDevice) + " DAMAGED BY THE HIT'");
        }
    }

    public void moveEnterprise(final float course, final float warp, final int n, final double stardate, final double initialStardate, final int missionDuration, final GameCallback callback) {
        insertMarker(MARKER_EMPTY, Util.toInt(enterprise.getSector()[Enterprise.COORD_X]), Util.toInt(enterprise.getSector()[Enterprise.COORD_Y]));
        final int[] sector = enterprise.moveShip(course, n, quadrantMap, stardate, initialStardate, missionDuration, callback);
        int sectorX = sector[Enterprise.COORD_X];
        int sectorY = sector[Enterprise.COORD_Y];
        insertMarker(MARKER_ENTERPRISE, Util.toInt(sectorX), Util.toInt(sectorY));
        enterprise.maneuverEnergySR(n);
        double stardateDelta = 1;
        if (warp < 1) stardateDelta = .1 * Util.toInt(10 * warp);
        callback.incrementStardate(stardateDelta);
        if (stardate > initialStardate + missionDuration) callback.endGameFail(false);
    }

    void shortRangeSensorScan(final double stardate) {
        final int sectorX = enterprise.getSector()[Enterprise.COORD_X];
        final int sectorY = enterprise.getSector()[Enterprise.COORD_Y];
        boolean docked = false;
        String shipCondition; // ship condition (docked, red, yellow, green)
        for (int i = sectorX - 1; i <= sectorX + 1; i++) {
            for (int j = sectorY - 1; j <= sectorY + 1; j++) {
                if ((Util.toInt(i) >= 1) && (Util.toInt(i) <= 8) && (Util.toInt(j) >= 1) && (Util.toInt(j) <= 8)) {
                    if (compareMarker(quadrantMap, MARKER_STARBASE, i, j)) {
                        docked = true;
                    }
                }
            }
        }
        if (!docked) {
            enterprise.setDocked(false);
            if (klingons > 0) {
                shipCondition = "*RED*";
            } else {
                shipCondition = "GREEN";
                if (enterprise.getEnergy() < enterprise.getInitialEnergy() * .1) {
                    shipCondition = "YELLOW";
                }
            }
        } else {
            enterprise.setDocked(true);
            shipCondition = "DOCKED";
            enterprise.replenishSupplies();
            Util.println("SHIELDS DROPPED FOR DOCKING PURPOSES");
            enterprise.dropShields();
        }
        if (enterprise.getDeviceStatus()[Enterprise.DEVICE_SHORT_RANGE_SENSORS] < 0) { // are short range sensors out?
            Util.println("\n*** SHORT RANGE SENSORS ARE OUT ***\n");
            return;
        }
        final String row = "---------------------------------";
        Util.println(row);
        for (int i = 1; i <= 8; i++) {
            String sectorMapRow = "";
            for (int j = (i - 1) * 24 + 1; j <= (i - 1) * 24 + 22; j += 3) {
                sectorMapRow += " " + Util.midStr(quadrantMap, j, 3);
            }
            switch (i) {
                case 1:
                    Util.println(sectorMapRow + "        STARDATE           " + Util.toInt(stardate * 10) * .1);
                    break;
                case 2:
                    Util.println(sectorMapRow + "        CONDITION          " + shipCondition);
                    break;
                case 3:
                    Util.println(sectorMapRow + "        QUADRANT           " + enterprise.getQuadrant()[Enterprise.COORD_X] + "," + enterprise.getQuadrant()[Enterprise.COORD_Y]);
                    break;
                case 4:
                    Util.println(sectorMapRow + "        SECTOR             " + sectorX + "," + sectorY);
                    break;
                case 5:
                    Util.println(sectorMapRow + "        PHOTON TORPEDOES   " + Util.toInt(enterprise.getTorpedoes()));
                    break;
                case 6:
                    Util.println(sectorMapRow + "        TOTAL ENERGY       " + Util.toInt(enterprise.getTotalEnergy()));
                    break;
                case 7:
                    Util.println(sectorMapRow + "        SHIELDS            " + Util.toInt(enterprise.getShields()));
                    break;
                case 8:
                    Util.println(sectorMapRow + "        KLINGONS REMAINING " + Util.toInt(klingonsInGalaxy));
            }
        }
        Util.println(row);
    }

    void longRangeSensorScan() {
        final int quadrantX = enterprise.getQuadrant()[Enterprise.COORD_X];
        final int quadrantY = enterprise.getQuadrant()[Enterprise.COORD_Y];
        if (enterprise.getDeviceStatus()[Enterprise.DEVICE_LONG_RANGE_SENSORS] < 0) {
            Util.println("LONG RANGE SENSORS ARE INOPERABLE");
            return;
        }
        Util.println("LONG RANGE SCAN FOR QUADRANT " + quadrantX + "," + quadrantY);
        final String rowStr = "-------------------";
        Util.println(rowStr);
        final int[] n = new int[4];
        for (int i = quadrantX - 1; i <= quadrantX + 1; i++) {
            n[1] = -1;
            n[2] = -2;
            n[3] = -3;
            for (int j = quadrantY - 1; j <= quadrantY + 1; j++) {
                if (i > 0 && i < 9 && j > 0 && j < 9) {
                    n[j - quadrantY + 2] = galaxy[i][j];
                    chartedGalaxy[i][j] = galaxy[i][j];
                }
            }
            for (int l = 1; l <= 3; l++) {
                Util.print(": ");
                if (n[l] < 0) {
                    Util.print("*** ");
                    continue;
                }
                Util.print(Util.rightStr(Integer.toString(n[l] + 1000), 3) + " ");
            }
            Util.println(": \n" + rowStr);
        }
    }

    void firePhasers(GameCallback callback) {
        final double[] deviceStatus = enterprise.getDeviceStatus();
        final int quadrantX = enterprise.getQuadrant()[Enterprise.COORD_X];
        final int quadrantY = enterprise.getQuadrant()[Enterprise.COORD_Y];
        if (deviceStatus[Enterprise.DEVICE_PHASER_CONTROL] < 0) {
            Util.println("PHASERS INOPERATIVE");
            return;
        }
        if (klingons <= 0) {
            printNoEnemyShipsMessage();
            return;
        }
        if (deviceStatus[Enterprise.DEVICE_LIBRARY_COMPUTER] < 0) Util.println("COMPUTER FAILURE HAMPERS ACCURACY");
        Util.println("PHASERS LOCKED ON TARGET;  ");
        int nrUnitsToFire;
        while (true) {
            Util.println("ENERGY AVAILABLE = " + enterprise.getEnergy() + " UNITS");
            nrUnitsToFire = Util.toInt(Util.inputFloat("NUMBER OF UNITS TO FIRE"));
            if (nrUnitsToFire <= 0) return;
            if (enterprise.getEnergy() - nrUnitsToFire >= 0) break;
        }
        enterprise.decreaseEnergy(nrUnitsToFire);
        if (deviceStatus[Enterprise.DEVICE_SHIELD_CONTROL] < 0) nrUnitsToFire = Util.toInt(nrUnitsToFire * Util.random());
        int h1 = Util.toInt(nrUnitsToFire / klingons);
        for (int i = 1; i <= 3; i++) {
            if (klingonQuadrants[i][3] <= 0) break;
            int hitPoints = Util.toInt((h1 / fnd(0)) * (Util.random() + 2));
            if (hitPoints <= .15 * klingonQuadrants[i][3]) {
                Util.println("SENSORS SHOW NO DAMAGE TO ENEMY AT " + klingonQuadrants[i][1] + "," + klingonQuadrants[i][2]);
                continue;
            }
            klingonQuadrants[i][3] = klingonQuadrants[i][3] - hitPoints;
            Util.println(hitPoints + " UNIT HIT ON KLINGON AT SECTOR " + klingonQuadrants[i][1] + "," + klingonQuadrants[i][2]);
            if (klingonQuadrants[i][3] <= 0) {
                Util.println("*** KLINGON DESTROYED ***");
                klingons -= 1;
                klingonsInGalaxy -= 1;
                insertMarker(MARKER_EMPTY, klingonQuadrants[i][1], klingonQuadrants[i][2]);
                klingonQuadrants[i][3] = 0;
                galaxy[quadrantX][quadrantY] -= 100;
                chartedGalaxy[quadrantX][quadrantY] = galaxy[quadrantX][quadrantY];
                if (klingonsInGalaxy <= 0) callback.endGameSuccess();
            } else {
                Util.println("   (SENSORS SHOW " + klingonQuadrants[i][3] + " UNITS REMAINING)");
            }
        }
        klingonsShoot(callback);
    }

    void firePhotonTorpedo(final double stardate, final double initialStardate, final double missionDuration, GameCallback callback) {
        if (enterprise.getTorpedoes() <= 0) {
            Util.println("ALL PHOTON TORPEDOES EXPENDED");
            return;
        }
        if (enterprise.getDeviceStatus()[Enterprise.DEVICE_PHOTON_TUBES] < 0) {
            Util.println("PHOTON TUBES ARE NOT OPERATIONAL");
        }
        float c1 = Util.inputFloat("PHOTON TORPEDO COURSE (1-9)");
        if (c1 == 9) c1 = 1;
        if (c1 < 1 && c1 >= 9) {
            Util.println("ENSIGN CHEKOV REPORTS,  'INCORRECT COURSE DATA, SIR!'");
            return;
        }
        int ic1 = Util.toInt(c1);
        final int[][] cardinalDirections = enterprise.getCardinalDirections();
        float x1 = cardinalDirections[ic1][1] + (cardinalDirections[ic1 + 1][1] - cardinalDirections[ic1][1]) * (c1 - ic1);
        enterprise.decreaseEnergy(2);
        enterprise.decreaseTorpedoes(1);
        float x2 = cardinalDirections[ic1][2] + (cardinalDirections[ic1 + 1][2] - cardinalDirections[ic1][2]) * (c1 - ic1);
        float x = enterprise.getSector()[Enterprise.COORD_X];
        float y = enterprise.getSector()[Enterprise.COORD_Y];
        Util.println("TORPEDO TRACK:");
        while (true) {
            x = x + x1;
            y = y + x2;
            int x3 = Math.round(x);
            int y3 = Math.round(y);
            if (x3 < 1 || x3 > 8 || y3 < 1 || y3 > 8) {
                Util.println("TORPEDO MISSED"); // 5490
                klingonsShoot(callback);
                return;
            }
            Util.println("               " + x3 + "," + y3);
            if (compareMarker(quadrantMap, MARKER_EMPTY, Util.toInt(x), Util.toInt(y)))  {
                continue;
            } else if (compareMarker(quadrantMap, MARKER_KLINGON, Util.toInt(x), Util.toInt(y))) {
                Util.println("*** KLINGON DESTROYED ***");
                klingons = klingons - 1;
                klingonsInGalaxy = klingonsInGalaxy - 1;
                if (klingonsInGalaxy <= 0) callback.endGameSuccess();
                for (int i = 1; i <= 3; i++) {
                    if (x3 == klingonQuadrants[i][1] && y3 == klingonQuadrants[i][2]) break;
                }
                int i = 3;
                klingonQuadrants[i][3] = 0;
            } else if (compareMarker(quadrantMap, MARKER_STAR, Util.toInt(x), Util.toInt(y))) {
                Util.println("STAR AT " + x3 + "," + y3 + " ABSORBED TORPEDO ENERGY.");
                klingonsShoot(callback);
                return;
            } else if (compareMarker(quadrantMap, MARKER_STARBASE, Util.toInt(x), Util.toInt(y))) {
                Util.println("*** STARBASE DESTROYED ***");
                starbases = starbases - 1;
                basesInGalaxy = basesInGalaxy - 1;
                if (basesInGalaxy == 0 && klingonsInGalaxy <= stardate - initialStardate - missionDuration) {
                    Util.println("THAT DOES IT, CAPTAIN!!  YOU ARE HEREBY RELIEVED OF COMMAND");
                    Util.println("AND SENTENCED TO 99 STARDATES AT HARD LABOR ON CYGNUS 12!!");
                    callback.endGameFail(false);
                } else {
                    Util.println("STARFLEET COMMAND REVIEWING YOUR RECORD TO CONSIDER");
                    Util.println("COURT MARTIAL!");
                    enterprise.setDocked(false);
                }
            }
            insertMarker(MARKER_EMPTY, Util.toInt(x), Util.toInt(y));
            final int quadrantX = enterprise.getQuadrant()[Enterprise.COORD_X];
            final int quadrantY = enterprise.getQuadrant()[Enterprise.COORD_Y];
            galaxy[quadrantX][quadrantY] = klingons * 100 + starbases * 10 + stars;
            chartedGalaxy[quadrantX][quadrantY] = galaxy[quadrantX][quadrantY];
            klingonsShoot(callback);
        }
    }

    public void cumulativeGalacticRecord(final boolean cumulativeReport) {
        final int quadrantX = enterprise.getQuadrant()[Enterprise.COORD_X];
        final int quadrantY = enterprise.getQuadrant()[Enterprise.COORD_Y];
        if (cumulativeReport) {
            Util.println("");
            Util.println("        ");
            Util.println("COMPUTER RECORD OF GALAXY FOR QUADRANT " + quadrantX + "," + quadrantY);
            Util.println("");
        } else {
            Util.println("                        THE GALAXY");
        }
        Util.println("       1     2     3     4     5     6     7     8");
        final String rowDivider = "     ----- ----- ----- ----- ----- ----- ----- -----";
        Util.println(rowDivider);
        for (int i = 1; i <= 8; i++) {
            Util.print(i + "  ");
            if (cumulativeReport) {
                int y = 1;
                String quadrantName = getQuadrantName(false, i, y);
                int tabLen = Util.toInt(15 - .5 * Util.strlen(quadrantName));
                Util.println(Util.tab(tabLen) + quadrantName);
                y = 5;
                quadrantName = getQuadrantName(false, i, y);
                tabLen = Util.toInt(39 - .5 * Util.strlen(quadrantName));
                Util.println(Util.tab(tabLen) + quadrantName);
            } else {
                for (int j = 1; j <= 8; j++) {
                    Util.print("   ");
                    if (chartedGalaxy[i][j] == 0) {
                        Util.print("***");
                    } else {
                        Util.print(Util.rightStr(Integer.toString(chartedGalaxy[i][j] + 1000), 3));
                    }
                }
            }
            Util.println("");
            Util.println(rowDivider);
        }
        Util.println("");
    }

    public void photonTorpedoData() {
        int sectorX = enterprise.getSector()[Enterprise.COORD_X];
        int sectorY = enterprise.getSector()[Enterprise.COORD_Y];
        if (klingons <= 0) {
            printNoEnemyShipsMessage();
            return;
        }
        Util.println("FROM ENTERPRISE TO KLINGON BATTLE CRUISER" + ((klingons > 1)? "S" : ""));
        for (int i = 1; i <= 3; i++) {
            if (klingonQuadrants[i][3] > 0) {
                printDirection(sectorX, sectorY, klingonQuadrants[i][1], klingonQuadrants[i][2]);
            }
        }
    }

    void directionDistanceCalculator() {
        int quadrantX = enterprise.getQuadrant()[Enterprise.COORD_X];
        int quadrantY = enterprise.getQuadrant()[Enterprise.COORD_Y];
        int sectorX = enterprise.getSector()[Enterprise.COORD_X];
        int sectorY = enterprise.getSector()[Enterprise.COORD_Y];
        Util.println("DIRECTION/DISTANCE CALCULATOR:");
        Util.println("YOU ARE AT QUADRANT " + quadrantX + "," + quadrantY + " SECTOR " + sectorX + "," + sectorY);
        Util.print("PLEASE ENTER ");
        int[] initialCoords = Util.inputCoords("  INITIAL COORDINATES (X,Y)");
        int[] finalCoords = Util.inputCoords("  FINAL COORDINATES (X,Y)");
        printDirection(initialCoords[0], initialCoords[1], finalCoords[0], finalCoords[1]);
    }

    void printDirection(int from_x, int from_y, int to_x, int to_y) {
        to_y = to_y - from_y;  // delta 2
        from_y = from_x - to_x;    // delta 1
        if (to_y > 0) {
            if (from_y < 0) {
                from_x = 7;
            } else {
                from_x = 1;
                int tempA = from_y;
                from_y = to_y;
                to_y = tempA;
            }
        } else {
            if (from_y > 0) {
                from_x = 3;
            } else {
                from_x = 5;
                int tempA = from_y;
                from_y = to_y;
                to_y = tempA;
            }
        }

        from_y = Math.abs(from_y);
        to_y = Math.abs(to_y);

        if (from_y > 0 || to_y > 0) {
            if (from_y >= to_y) {
                Util.println("DIRECTION = " + (from_x + to_y / from_y));
            } else {
                Util.println("DIRECTION = " + (from_x + 2 - to_y / from_y));
            }
        }
        Util.println("DISTANCE = " + Util.round(Math.sqrt(to_y ^ 2 + from_y ^ 2), 6));
    }

    void starbaseNavData() {
        int sectorX = enterprise.getSector()[Enterprise.COORD_X];
        int sectorY = enterprise.getSector()[Enterprise.COORD_Y];
        if (starbases != 0) {
            Util.println("FROM ENTERPRISE TO STARBASE:");
            printDirection(sectorX, sectorY, starbaseX, starbaseY);
        } else {
            Util.println("MR. SPOCK REPORTS,  'SENSORS SHOW NO STARBASES IN THIS");
            Util.println(" QUADRANT.'");
        }
    }

    void printNoEnemyShipsMessage() {
        Util.println("SCIENCE OFFICER SPOCK REPORTS  'SENSORS SHOW NO ENEMY SHIPS");
        Util.println("                                IN THIS QUADRANT'");
    }

    String getRegionName(final boolean regionNameOnly, final int y) {
        if (!regionNameOnly) {
            switch (y % 4) {
                case 0:
                    return " I";
                case 1:
                    return " II";
                case 2:
                    return " III";
                case 3:
                    return " IV";
            }
        }
        return "";
    }

    String getQuadrantName(final boolean regionNameOnly, final int x, final int y) {
        if (y <= 4) {
            switch (x) {
                case 1:
                    return "ANTARES" + getRegionName(regionNameOnly, y);
                case 2:
                    return "RIGEL" + getRegionName(regionNameOnly, y);
                case 3:
                    return "PROCYON" + getRegionName(regionNameOnly, y);
                case 4:
                    return "VEGA" + getRegionName(regionNameOnly, y);
                case 5:
                    return "CANOPUS" + getRegionName(regionNameOnly, y);
                case 6:
                    return "ALTAIR" + getRegionName(regionNameOnly, y);
                case 7:
                    return "SAGITTARIUS" + getRegionName(regionNameOnly, y);
                case 8:
                    return "POLLUX" + getRegionName(regionNameOnly, y);
            }
        } else {
            switch (x) {
                case 1:
                    return "SIRIUS" + getRegionName(regionNameOnly, y);
                case 2:
                    return "DENEB" + getRegionName(regionNameOnly, y);
                case 3:
                    return "CAPELLA" + getRegionName(regionNameOnly, y);
                case 4:
                    return "BETELGEUSE" + getRegionName(regionNameOnly, y);
                case 5:
                    return "ALDEBARAN" + getRegionName(regionNameOnly, y);
                case 6:
                    return "REGULUS" + getRegionName(regionNameOnly, y);
                case 7:
                    return "ARCTURUS" + getRegionName(regionNameOnly, y);
                case 8:
                    return "SPICA" + getRegionName(regionNameOnly, y);
            }
        }
        return "UNKNOWN - ERROR";
    }

    void insertMarker(final String marker, final int x, final int y) {
        final int pos = Util.toInt(y) * 3 + Util.toInt(x) * 24 + 1;
        if (marker.length() != 3) {
            System.err.println("ERROR");
            System.exit(-1);
        }
        if (pos == 1) {
            quadrantMap = marker + Util.rightStr(quadrantMap, 189);
        }
        if (pos == 190) {
            quadrantMap = Util.leftStr(quadrantMap, 189) + marker;
        }
        quadrantMap = Util.leftStr(quadrantMap, (pos - 1)) + marker + Util.rightStr(quadrantMap, (190 - pos));
    }

    /**
     * Finds random empty coordinates in a quadrant.
     *
     * @param quadrantString
     * @return an array with a pair of coordinates x, y
     */
    int[] findEmptyPlaceInQuadrant(final String quadrantString) {
        final int x = Util.fnr();
        final int y = Util.fnr();
        if (!compareMarker(quadrantString, MARKER_EMPTY, x, y)) {
            return findEmptyPlaceInQuadrant(quadrantString);
        }
        return new int[]{x, y};
    }

    boolean compareMarker(final String quadrantString, final String marker, final int x, final int y) {
        final int markerRegion = (y - 1) * 3 + (x - 1) * 24 + 1;
        if (Util.midStr(quadrantString, markerRegion, 3).equals(marker)) {
            return true;
        }
        return false;
    }

}

```
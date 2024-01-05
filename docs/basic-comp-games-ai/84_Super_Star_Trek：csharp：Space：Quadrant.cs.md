# `84_Super_Star_Trek\csharp\Space\Quadrant.cs`

```
# 导入所需的模块
import System
import Collections.Generic
import Linq
import Games.Common.IO
import Games.Common.Randomness
import SuperStarTrek.Commands
import SuperStarTrek.Objects
import SuperStarTrek.Resources

# 定义名为Quadrant的类
class Quadrant:
    # 初始化Quadrant类的属性
    def __init__(self, info, random, sectors, enterprise, io):
        self._info = info  # 保存QuadrantInfo对象的引用
        self._random = random  # 保存IRandom对象的引用
        self._sectors = sectors  # 保存包含坐标和对象的字典的引用
        self._enterprise = enterprise  # 保存Enterprise对象的引用
        self._io = io  # 保存IReadWrite对象的引用
        self._entered = False  # 初始化_entered属性为False
    internal Quadrant(
        QuadrantInfo info,  // 传入QuadrantInfo对象作为参数
        Enterprise enterprise,  // 传入Enterprise对象作为参数
        IRandom random,  // 传入IRandom对象作为参数
        Galaxy galaxy,  // 传入Galaxy对象作为参数
        IReadWrite io)  // 传入IReadWrite对象作为参数
    {
        _info = info;  // 将传入的QuadrantInfo对象赋值给私有变量_info
        _random = random;  // 将传入的IRandom对象赋值给私有变量_random
        _io = io;  // 将传入的IReadWrite对象赋值给私有变量_io
        Galaxy = galaxy;  // 将传入的Galaxy对象赋值给公共变量Galaxy

        info.MarkAsKnown();  // 调用QuadrantInfo对象的MarkAsKnown方法
        _sectors = new() { [enterprise.SectorCoordinates] = _enterprise = enterprise };  // 创建一个新的字典，将enterprise.SectorCoordinates作为键，enterprise作为值，并赋值给私有变量_sectors
        PositionObject(sector => new Klingon(sector, _random), _info.KlingonCount);  // 调用PositionObject方法，根据_info.KlingonCount的值在每个扇区中放置相应数量的Klingon对象
        if (_info.HasStarbase)  // 如果info对象中包含Starbase
        {
            Starbase = PositionObject(sector => new Starbase(sector, _random, io));  // 调用PositionObject方法，在每个扇区中放置Starbase对象，并将返回的Starbase对象赋值给公共变量Starbase
        }
        PositionObject(_ => new Star(), _info.StarCount);  // 调用PositionObject方法，在每个扇区中放置相应数量的Star对象
    }

    internal Coordinates Coordinates => _info.Coordinates;  // 返回 _info.Coordinates 的坐标信息

    internal bool HasKlingons => _info.KlingonCount > 0;  // 返回 _info.KlingonCount 是否大于 0

    internal int KlingonCount => _info.KlingonCount;  // 返回 _info.KlingonCount 的值

    internal bool HasStarbase => _info.HasStarbase;  // 返回 _info.HasStarbase 的布尔值

    internal Starbase Starbase { get; }  // 返回 Starbase 对象

    internal Galaxy Galaxy { get; }  // 返回 Galaxy 对象

    internal bool EnterpriseIsNextToStarbase =>  // 返回企业飞船是否邻近星舰基地的布尔值
        _info.HasStarbase &&
        Math.Abs(_enterprise.SectorCoordinates.X - Starbase.Sector.X) <= 1 &&
        Math.Abs(_enterprise.SectorCoordinates.Y - Starbase.Sector.Y) <= 1;

    internal IEnumerable<Klingon> Klingons => _sectors.Values.OfType<Klingon>();  // 返回 _sectors.Values 中的 Klingon 对象的集合
    public override string ToString() => _info.Name;  // 重写 ToString 方法，返回 _info 的 Name 属性值

    private T PositionObject<T>(Func<Coordinates, T> objectFactory)  // 定义一个泛型方法 PositionObject，接受一个返回 T 类型的委托 objectFactory
    {
        var sector = GetRandomEmptySector();  // 获取一个随机的空的区块
        _sectors[sector] = objectFactory.Invoke(sector);  // 使用 objectFactory 创建一个对象，并将其放置在获取的区块上
        return (T)_sectors[sector];  // 返回放置的对象
    }

    private void PositionObject(Func<Coordinates, object> objectFactory, int count)  // 定义一个方法 PositionObject，接受一个返回 object 类型的委托 objectFactory 和一个整数 count
    {
        for (int i = 0; i < count; i++)  // 循环 count 次
        {
            PositionObject(objectFactory);  // 调用 PositionObject 方法，使用 objectFactory 放置对象
        }
    }

    internal void Display(string textFormat)  // 定义一个内部方法 Display，接受一个字符串 textFormat
        if (!_entered)
        {
            # 如果 _entered 变量为 false，则执行以下代码
            _io.Write(textFormat, this);
            # 将当前对象的信息以指定格式写入输出流
            _entered = true;
            # 将 _entered 变量设置为 true
        }

        if (_info.KlingonCount > 0)
        {
            # 如果星舰周围有克林贡战舰
            _io.Write(Strings.CombatArea);
            # 在输出流中写入战斗区域信息
            if (_enterprise.ShieldControl.ShieldEnergy <= 200) { _io.Write(Strings.LowShields); }
            # 如果星舰的护盾能量低于等于200，则在输出流中写入护盾能量低的提示信息
        }

        _enterprise.Execute(Command.SRS);
        # 执行星际雷达扫描命令
    }

    internal bool HasObjectAt(Coordinates coordinates) => _sectors.ContainsKey(coordinates);
    # 判断指定坐标是否有物体存在，返回结果为是否存在的布尔值

    internal bool TorpedoCollisionAt(Coordinates coordinates, out string message, out bool gameOver)
    {
        # 判断指定坐标是否有鱼雷碰撞，同时返回消息和游戏是否结束的布尔值
        gameOver = false;
        # 将游戏是否结束的布尔值设置为 false
        message = default;  # 初始化消息变量为默认值

        switch (_sectors.GetValueOrDefault(coordinates))  # 使用坐标获取对应的值，进行多个条件判断
        {
            case Klingon klingon:  # 如果获取的值是 Klingon 类型的对象
                message = Remove(klingon);  # 调用 Remove 方法处理 Klingon 对象，并将返回的消息赋值给消息变量
                gameOver = Galaxy.KlingonCount == 0;  # 判断 Klingon 的数量是否为 0，如果是则游戏结束
                return true;  # 返回 true

            case Star _:  # 如果获取的值是 Star 类型的对象
                message = $"Star at {coordinates} absorbed torpedo energy.";  # 根据坐标生成消息
                return true;  # 返回 true

            case Starbase _:  # 如果获取的值是 Starbase 类型的对象
                _sectors.Remove(coordinates);  # 移除对应坐标的值
                _info.RemoveStarbase();  # 调用 RemoveStarbase 方法
                message = "*** Starbase destroyed ***" +  # 生成消息
                    (Galaxy.StarbaseCount > 0 ? Strings.CourtMartial : Strings.RelievedOfCommand);  # 根据 Starbase 的数量生成不同的消息
                gameOver = Galaxy.StarbaseCount == 0;  # 判断 Starbase 的数量是否为 0，如果是则游戏结束
                return true;  # 返回 true
            default:
                return false;
        }
    }
```
这段代码是一个 switch 语句的一部分，根据不同的条件执行不同的操作。

```
    internal string Remove(Klingon klingon)
    {
        _sectors.Remove(klingon.Sector);
        _info.RemoveKlingon();
        return "*** Klingon destroyed ***";
    }
```
这段代码定义了一个名为 Remove 的方法，用于移除指定的 Klingon 对象，并返回一个字符串 "*** Klingon destroyed ***"。

```
    internal CommandResult KlingonsMoveAndFire()
    {
        foreach (var klingon in Klingons.ToList())
        {
            var newSector = GetRandomEmptySector();
            _sectors.Remove(klingon.Sector);
            _sectors[newSector] = klingon;
```
这段代码定义了一个名为 KlingonsMoveAndFire 的方法，用于让 Klingon 移动并开火。它遍历了 Klingons 列表中的每个 Klingon 对象，并对其进行移动和开火的操作。
            klingon.MoveTo(newSector);  # 将克林贡飞船移动到新的扇区

        }

        return KlingonsFireOnEnterprise();  # 调用 KlingonsFireOnEnterprise 方法并返回结果
    }

    internal CommandResult KlingonsFireOnEnterprise()  # 定义内部方法 KlingonsFireOnEnterprise，返回类型为 CommandResult
    {
        if (EnterpriseIsNextToStarbase && Klingons.Any())  # 如果企业号在星舰基地旁边并且有克林贡飞船
        {
            Starbase.ProtectEnterprise();  # 星舰基地保护企业号
            return CommandResult.Ok;  # 返回命令结果为 Ok
        }

        foreach (var klingon in Klingons)  # 遍历克林贡飞船列表
        {
            var result = klingon.FireOn(_enterprise);  # 克林贡飞船对企业号开火，并将结果保存在 result 变量中
            if (result.IsGameOver) { return result; }  # 如果游戏结束，返回结果
        }
        return CommandResult.Ok;
    }
    # 返回一个表示成功的命令结果

    private Coordinates GetRandomEmptySector()
    {
        while (true)
        {
            var sector = _random.NextCoordinate();
            # 生成一个随机坐标
            if (!_sectors.ContainsKey(sector))
            {
                return sector;
            }
            # 如果该坐标不在已占用的坐标集合中，则返回该坐标
        }
    }

    internal IEnumerable<string> GetDisplayLines() => Enumerable.Range(0, 8).Select(x => GetDisplayLine(x));
    # 返回一个包含8个元素的字符串集合，每个元素表示一个显示行

    private string GetDisplayLine(int x) =>
        string.Join(
            " ",
```

            Enumerable
                .Range(0, 8)  # 创建一个从 0 到 7 的整数序列
                .Select(y => new Coordinates(x, y))  # 对每个整数应用函数，创建一个 Coordinates 对象
                .Select(c => _sectors.GetValueOrDefault(c))  # 对每个 Coordinates 对象应用函数，获取对应的值
                .Select(o => o?.ToString() ?? "   "));  # 对每个值应用函数，如果值不为空则调用 ToString() 方法，否则返回空格

    internal void SetEnterpriseSector(Coordinates sector)
    {
        _sectors.Remove(_enterprise.SectorCoordinates);  # 从 _sectors 字典中移除指定的键值对
        _sectors[sector] = _enterprise;  # 将指定的键值对添加到 _sectors 字典中
    }
}
```
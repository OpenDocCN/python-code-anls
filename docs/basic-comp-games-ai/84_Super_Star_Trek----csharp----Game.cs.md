# `84_Super_Star_Trek\csharp\Game.cs`

```
# 导入所需的模块
import System
import Games.Common.IO
import Games.Common.Randomness
import SuperStarTrek.Objects
import SuperStarTrek.Resources
import SuperStarTrek.Space
import SuperStarTrek.Systems
import SuperStarTrek.Systems.ComputerFunctions

# 定义命名空间
namespace SuperStarTrek;

# 定义 Game 类
internal class Game:
    # 初始化 Game 类的属性
    def __init__(self, io, random):
        self._io = io  # 初始化 TextIO 类的实例
        self._random = random  # 初始化 IRandom 类的实例
        self._initialStardate = 0  # 初始化初始星期
        self._finalStarDate = 0  # 初始化最终星期
        self._currentStardate = 0.0  # 初始化当前星期
        self._currentQuadrant = Coordinates()  # 初始化当前象限坐标
    // 声明私有变量 _galaxy，用于存储星系信息
    private Galaxy _galaxy;
    // 声明私有变量 _initialKlingonCount，用于存储初始克林贡人数
    private int _initialKlingonCount;
    // 声明私有变量 _enterprise，用于存储企业号信息

    // Game 类的构造函数，接受 TextIO 和 IRandom 对象作为参数
    internal Game(TextIO io, IRandom random)
    {
        _io = io; // 将参数 io 赋值给私有变量 _io
        _random = random; // 将参数 random 赋值给私有变量 _random
    }

    // 返回当前星际日期
    internal float Stardate => _currentStardate;

    // 返回剩余的星际日期
    internal float StardatesRemaining => _finalStarDate - _currentStardate;

    // 进行游戏介绍
    internal void DoIntroduction()
    {
        _io.Write(Strings.Title); // 在控制台输出游戏标题

        // 如果用户需要游戏说明
        if (_io.GetYesNo("Do you need instructions", IReadWriteExtensions.YesNoMode.FalseOnN))
        {
            _io.Write(Strings.Instructions);  # 在控制台输出游戏指令

            _io.WaitForAnyKeyButEnter("to continue");  # 等待用户按下除了回车键以外的任意键继续游戏
        }
    }

    internal void Play()
    {
        Initialise();  # 初始化游戏
        var gameOver = false;  # 初始化游戏结束标志为假

        while (!gameOver)  # 当游戏未结束时循环执行以下代码
        {
            var command = _io.ReadCommand();  # 从控制台读取用户输入的命令

            var result = _enterprise.Execute(command);  # 执行用户输入的命令并得到执行结果

            gameOver = result.IsGameOver || CheckIfStranded();  # 如果执行结果为游戏结束或者检查到船被困住，则游戏结束标志为真
            _currentStardate += result.TimeElapsed;  # 更新当前星日期
            gameOver |= _currentStardate > _finalStarDate;  # 如果当前星日期大于最终星日期，则游戏结束标志为真
        }

        if (_galaxy.KlingonCount > 0)
        {
            # 如果星系中克林贡人的数量大于0，则向输出流写入任务结束的消息和当前星际日期以及克林贡人的数量
            _io.Write(Strings.EndOfMission, _currentStardate, _galaxy.KlingonCount);
        }
        else
        {
            # 如果星系中克林贡人的数量等于0，则向输出流写入祝贺消息和计算效率后的结果
            _io.Write(Strings.Congratulations, CalculateEfficiency());
        }
    }

    private void Initialise()
    {
        # 初始化当前星际日期为初始星际日期，并在20到40之间随机生成一个数乘以100
        _currentStardate = _initialStardate = _random.Next(20, 40) * 100;
        # 计算最终星际日期为初始星际日期加上25到35之间的随机数
        _finalStarDate = _initialStardate + _random.Next(25, 35);

        # 初始化当前象限为随机生成的坐标
        _currentQuadrant = _random.NextCoordinate();

        # 创建一个新的星系对象，传入随机数生成器作为参数
        _galaxy = new Galaxy(_random);
        _initialKlingonCount = _galaxy.KlingonCount;  # 将_galaxy对象的KlingonCount属性的值赋给_initialKlingonCount变量

        _enterprise = new Enterprise(3000, _random.NextCoordinate(), _io, _random);  # 创建一个新的Enterprise对象，传入参数3000、_random.NextCoordinate()、_io和_random
        _enterprise
            .Add(new WarpEngines(_enterprise, _io))  # 在_enterprise对象上添加WarpEngines对象
            .Add(new ShortRangeSensors(_enterprise, _galaxy, this, _io))  # 在_enterprise对象上添加ShortRangeSensors对象，传入参数_enterprise、_galaxy、this和_io
            .Add(new LongRangeSensors(_galaxy, _io))  # 在_enterprise对象上添加LongRangeSensors对象，传入参数_galaxy和_io
            .Add(new PhaserControl(_enterprise, _io, _random))  # 在_enterprise对象上添加PhaserControl对象，传入参数_enterprise、_io和_random
            .Add(new PhotonTubes(10, _enterprise, _io))  # 在_enterprise对象上添加PhotonTubes对象，传入参数10、_enterprise和_io
            .Add(new ShieldControl(_enterprise, _io))  # 在_enterprise对象上添加ShieldControl对象，传入参数_enterprise和_io
            .Add(new DamageControl(_enterprise, _io))  # 在_enterprise对象上添加DamageControl对象，传入参数_enterprise和_io
            .Add(new LibraryComputer(
                _io,
                new CumulativeGalacticRecord(_io, _galaxy),  # 在LibraryComputer对象中创建CumulativeGalacticRecord对象，传入参数_io和_galaxy
                new StatusReport(this, _galaxy, _enterprise, _io),  # 在LibraryComputer对象中创建StatusReport对象，传入参数this、_galaxy、_enterprise和_io
                new TorpedoDataCalculator(_enterprise, _io),  # 在LibraryComputer对象中创建TorpedoDataCalculator对象，传入参数_enterprise和_io
                new StarbaseDataCalculator(_enterprise, _io),  # 在LibraryComputer对象中创建StarbaseDataCalculator对象，传入参数_enterprise和_io
                new DirectionDistanceCalculator(_enterprise, _io),  # 在LibraryComputer对象中创建DirectionDistanceCalculator对象，传入参数_enterprise和_io
                new GalaxyRegionMap(_io, _galaxy)));  # 在LibraryComputer对象中创建GalaxyRegionMap对象，传入参数_io和_galaxy
        _io.Write(Strings.Enterprise);  # 在控制台输出字符串 "Enterprise"
        _io.Write(
            Strings.Orders,  # 在控制台输出字符串 "Orders"
            _galaxy.KlingonCount,  # 在控制台输出 _galaxy.KlingonCount 的值
            _finalStarDate,  # 在控制台输出 _finalStarDate 的值
            _finalStarDate - _initialStardate,  # 在控制台输出 _finalStarDate - _initialStardate 的值
            _galaxy.StarbaseCount > 1 ? "are" : "is",  # 如果 _galaxy.StarbaseCount 大于 1，则输出 "are"，否则输出 "is"
            _galaxy.StarbaseCount,  # 在控制台输出 _galaxy.StarbaseCount 的值
            _galaxy.StarbaseCount > 1 ? "s" : "");  # 如果 _galaxy.StarbaseCount 大于 1，则输出 "s"，否则输出空字符串

        _io.WaitForAnyKeyButEnter("when ready to accept command");  # 等待用户按下除了 Enter 键以外的任意键，然后继续执行

        _enterprise.StartIn(BuildCurrentQuadrant());  # 在 _enterprise 对象上调用 StartIn 方法，并传入 BuildCurrentQuadrant() 方法的返回值作为参数
    }

    private Quadrant BuildCurrentQuadrant() => new(_galaxy[_currentQuadrant], _enterprise, _random, _galaxy, _io);  # 创建并返回一个 Quadrant 对象，传入 _galaxy[_currentQuadrant]、_enterprise、_random、_galaxy、_io 作为参数

    internal bool Replay() => _galaxy.StarbaseCount > 0 && _io.ReadExpectedString(Strings.ReplayPrompt, "Aye");  # 返回一个布尔值，表示 _galaxy.StarbaseCount 大于 0 并且用户输入的字符串与 "Aye" 相匹配

    private bool CheckIfStranded()  # 检查是否处于困境
{
    # 如果企业舰被困，向输出流写入"被困"的字符串
    if (_enterprise.IsStranded) { _io.Write(Strings.Stranded); }
    # 返回企业舰是否被困的布尔值
    return _enterprise.IsStranded;
}

# 计算效率的私有方法，使用隐式返回语法
private float CalculateEfficiency() =>
    # 计算效率公式并返回结果
    1000 * (float)Math.Pow(_initialKlingonCount / (_currentStardate - _initialStardate), 2);
}
```
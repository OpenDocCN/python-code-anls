# `basic-computer-games\84_Super_Star_Trek\csharp\Game.cs`

```
using System;  # 导入系统命名空间
using Games.Common.IO;  # 导入游戏通用输入输出命名空间
using Games.Common.Randomness;  # 导入游戏通用随机数生成命名空间
using SuperStarTrek.Objects;  # 导入超级星际迷航游戏对象命名空间
using SuperStarTrek.Resources;  # 导入超级星际迷航游戏资源命名空间
using SuperStarTrek.Space;  # 导入超级星际迷航游戏空间命名空间
using SuperStarTrek.Systems;  # 导入超级星际迷航游戏系统命名空间
using SuperStarTrek.Systems.ComputerFunctions;  # 导入超级星际迷航游戏计算机功能命名空间

namespace SuperStarTrek;  # 声明超级星际迷航游戏命名空间

internal class Game  # 声明内部游戏类
{
    private readonly TextIO _io;  # 声明只读文本输入输出对象
    private readonly IRandom _random;  # 声明只读随机数生成对象

    private int _initialStardate;  # 声明初始星际日期
    private int _finalStarDate;  # 声明最终星际日期
    private float _currentStardate;  # 声明当前星际日期
    private Coordinates _currentQuadrant;  # 声明当前象限坐标
    private Galaxy _galaxy;  # 声明星系对象
    private int _initialKlingonCount;  # 声明初始克林贡数量
    private Enterprise _enterprise;  # 声明企业号对象

    internal Game(TextIO io, IRandom random)  # 声明游戏类构造函数，接受文本输入输出对象和随机数生成对象
    {
        _io = io;  # 初始化文本输入输出对象
        _random = random;  # 初始化随机数生成对象
    }

    internal float Stardate => _currentStardate;  # 声明星际日期属性，返回当前星际日期

    internal float StardatesRemaining => _finalStarDate - _currentStardate;  # 声明剩余星际日期属性，返回最终星际日期减去当前星际日期

    internal void DoIntroduction()  # 声明游戏介绍方法
    {
        _io.Write(Strings.Title);  # 输出游戏标题

        if (_io.GetYesNo("Do you need instructions", IReadWriteExtensions.YesNoMode.FalseOnN))  # 如果需要说明
        {
            _io.Write(Strings.Instructions);  # 输出游戏说明

            _io.WaitForAnyKeyButEnter("to continue");  # 等待按下除了回车键以外的任意键继续
        }
    }

    internal void Play()  # 声明游戏进行方法
    {
        Initialise();  # 调用初始化方法
        var gameOver = false;  # 声明游戏结束标志为假

        while (!gameOver)  # 当游戏未结束时
        {
            var command = _io.ReadCommand();  # 读取命令

            var result = _enterprise.Execute(command);  # 执行命令并获取结果

            gameOver = result.IsGameOver || CheckIfStranded();  # 游戏结束标志为结果为游戏结束或者检查是否被困
            _currentStardate += result.TimeElapsed;  # 当前星际日期增加结果中的时间流逝
            gameOver |= _currentStardate > _finalStarDate;  # 游戏结束标志为游戏结束标志或者当前星际日期大于最终星际日期
        }

        if (_galaxy.KlingonCount > 0)  # 如果星系中克林贡数量大于0
        {
            _io.Write(Strings.EndOfMission, _currentStardate, _galaxy.KlingonCount);  # 输出任务结束信息，包括当前星际日期和克林贡数量
        }
        else  # 否则
        {
            _io.Write(Strings.Congratulations, CalculateEfficiency());  # 输出祝贺信息和计算效率
        }
    }

    private void Initialise()  # 声明初始化方法
    {
        # 设置初始星日期和当前星日期为一个随机数
        _currentStardate = _initialStardate = _random.Next(20, 40) * 100;
        # 计算最终星日期
        _finalStarDate = _initialStardate + _random.Next(25, 35);
    
        # 生成当前象限的坐标
        _currentQuadrant = _random.NextCoordinate();
    
        # 创建一个新的星系对象
        _galaxy = new Galaxy(_random);
        # 记录初始克林贡数量
        _initialKlingonCount = _galaxy.KlingonCount;
    
        # 创建一个新的企业飞船对象
        _enterprise = new Enterprise(3000, _random.NextCoordinate(), _io, _random);
        # 为企业飞船添加各种系统
        _enterprise
            .Add(new WarpEngines(_enterprise, _io))
            .Add(new ShortRangeSensors(_enterprise, _galaxy, this, _io))
            .Add(new LongRangeSensors(_galaxy, _io))
            .Add(new PhaserControl(_enterprise, _io, _random))
            .Add(new PhotonTubes(10, _enterprise, _io))
            .Add(new ShieldControl(_enterprise, _io))
            .Add(new DamageControl(_enterprise, _io))
            .Add(new LibraryComputer(
                _io,
                new CumulativeGalacticRecord(_io, _galaxy),
                new StatusReport(this, _galaxy, _enterprise, _io),
                new TorpedoDataCalculator(_enterprise, _io),
                new StarbaseDataCalculator(_enterprise, _io),
                new DirectionDistanceCalculator(_enterprise, _io),
                new GalaxyRegionMap(_io, _galaxy)));
    
        # 输出企业飞船信息
        _io.Write(Strings.Enterprise);
        # 输出指令信息
        _io.Write(
            Strings.Orders,
            _galaxy.KlingonCount,
            _finalStarDate,
            _finalStarDate - _initialStardate,
            _galaxy.StarbaseCount > 1 ? "are" : "is",
            _galaxy.StarbaseCount,
            _galaxy.StarbaseCount > 1 ? "s" : "");
        
        # 等待用户输入命令
        _io.WaitForAnyKeyButEnter("when ready to accept command");
    
        # 在当前象限启动企业飞船
        _enterprise.StartIn(BuildCurrentQuadrant());
    }
    
    # 创建当前象限对象
    private Quadrant BuildCurrentQuadrant() => new(_galaxy[_currentQuadrant], _enterprise, _random, _galaxy, _io);
    
    # 回放游戏
    internal bool Replay() => _galaxy.StarbaseCount > 0 && _io.ReadExpectedString(Strings.ReplayPrompt, "Aye");
    
    # 检查是否搁浅
    private bool CheckIfStranded()
    {
        # 如果企业号被困，向输出流写入"Stranded"字符串
        if (_enterprise.IsStranded) { _io.Write(Strings.Stranded); }
        # 返回企业号是否被困的布尔值
        return _enterprise.IsStranded;
    }

    # 计算效率的私有方法，使用隐式返回值
    private float CalculateEfficiency() =>
        # 返回计算得到的效率值
        1000 * (float)Math.Pow(_initialKlingonCount / (_currentStardate - _initialStardate), 2);
# 闭合前面的函数定义
```
# `basic-computer-games\84_Super_Star_Trek\csharp\Game.cs`

```

// 引入所需的命名空间
using System;
using Games.Common.IO;
using Games.Common.Randomness;
using SuperStarTrek.Objects;
using SuperStarTrek.Resources;
using SuperStarTrek.Space;
using SuperStarTrek.Systems;
using SuperStarTrek.Systems.ComputerFunctions;

namespace SuperStarTrek
{
    // 定义游戏类
    internal class Game
    {
        // 声明私有变量
        private readonly TextIO _io;
        private readonly IRandom _random;
        private int _initialStardate;
        private int _finalStarDate;
        private float _currentStardate;
        private Coordinates _currentQuadrant;
        private Galaxy _galaxy;
        private int _initialKlingonCount;
        private Enterprise _enterprise;

        // 游戏类的构造函数
        internal Game(TextIO io, IRandom random)
        {
            _io = io;
            _random = random;
        }

        // 返回当前星日期
        internal float Stardate => _currentStardate;

        // 返回剩余星日期
        internal float StardatesRemaining => _finalStarDate - _currentStardate;

        // 游戏介绍
        internal void DoIntroduction()
        {
            _io.Write(Strings.Title);

            if (_io.GetYesNo("Do you need instructions", IReadWriteExtensions.YesNoMode.FalseOnN))
            {
                _io.Write(Strings.Instructions);

                _io.WaitForAnyKeyButEnter("to continue");
            }
        }

        // 游戏进行
        internal void Play()
        {
            Initialise();
            var gameOver = false;

            while (!gameOver)
            {
                var command = _io.ReadCommand();

                var result = _enterprise.Execute(command);

                gameOver = result.IsGameOver || CheckIfStranded();
                _currentStardate += result.TimeElapsed;
                gameOver |= _currentStardate > _finalStarDate;
            }

            if (_galaxy.KlingonCount > 0)
            {
                _io.Write(Strings.EndOfMission, _currentStardate, _galaxy.KlingonCount);
            }
            else
            {
                _io.Write(Strings.Congratulations, CalculateEfficiency());
            }
        }

        // 初始化游戏
        private void Initialise()
        {
            _currentStardate = _initialStardate = _random.Next(20, 40) * 100;
            _finalStarDate = _initialStardate + _random.Next(25, 35);

            _currentQuadrant = _random.NextCoordinate();

            _galaxy = new Galaxy(_random);
            _initialKlingonCount = _galaxy.KlingonCount;

            _enterprise = new Enterprise(3000, _random.NextCoordinate(), _io, _random);
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

            _io.Write(Strings.Enterprise);
            _io.Write(
                Strings.Orders,
                _galaxy.KlingonCount,
                _finalStarDate,
                _finalStarDate - _initialStardate,
                _galaxy.StarbaseCount > 1 ? "are" : "is",
                _galaxy.StarbaseCount,
                _galaxy.StarbaseCount > 1 ? "s" : "");

            _io.WaitForAnyKeyButEnter("when ready to accept command");

            _enterprise.StartIn(BuildCurrentQuadrant());
        }

        // 构建当前象限
        private Quadrant BuildCurrentQuadrant() => new(_galaxy[_currentQuadrant], _enterprise, _random, _galaxy, _io);

        // 重新开始游戏
        internal bool Replay() => _galaxy.StarbaseCount > 0 && _io.ReadExpectedString(Strings.ReplayPrompt, "Aye");

        // 检查是否被困
        private bool CheckIfStranded()
        {
            if (_enterprise.IsStranded) { _io.Write(Strings.Stranded); }
            return _enterprise.IsStranded;
        }

        // 计算效率
        private float CalculateEfficiency() =>
            1000 * (float)Math.Pow(_initialKlingonCount / (_currentStardate - _initialStardate), 2);
    }
}

```
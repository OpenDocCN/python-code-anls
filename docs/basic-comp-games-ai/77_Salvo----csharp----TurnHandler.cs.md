# `basic-computer-games\77_Salvo\csharp\TurnHandler.cs`

```

// 使用 Salvo.Targetting 命名空间
using Salvo.Targetting;

// 声明 Salvo 命名空间
namespace Salvo;

// 定义 TurnHandler 类
internal class TurnHandler
{
    // 声明私有字段
    private readonly IReadWrite _io;
    private readonly Fleet _humanFleet;
    private readonly Fleet _computerFleet;
    private readonly bool _humanStarts;
    private readonly HumanShotSelector _humanShotSelector;
    private readonly ComputerShotSelector _computerShotSelector;
    private readonly Func<Winner?> _turnAction;
    private int _turnNumber;

    // TurnHandler 类的构造函数
    public TurnHandler(IReadWrite io, IRandom random)
    {
        // 初始化私有字段
        _io = io;
        _computerFleet = new Fleet(random);
        _humanFleet = new Fleet(io);
        _turnAction = AskWhoStarts()
            ? () => PlayHumanTurn() ?? PlayComputerTurn()
            : () => PlayComputerTurn() ?? PlayHumanTurn();
        _humanShotSelector = new HumanShotSelector(_humanFleet, io);
        _computerShotSelector = new ComputerShotSelector(_computerFleet, random, io);
    }

    // PlayTurn 方法
    public Winner? PlayTurn()
    {
        // 输出当前回合数
        _io.Write(Strings.Turn(++_turnNumber));
        // 调用 _turnAction 委托
        return _turnAction.Invoke();
    }

    // AskWhoStarts 方法
    private bool AskWhoStarts()
    {
        // 循环询问谁先开始
        while (true)
        {
            var startResponse = _io.ReadString(Prompts.Start);
            if (startResponse.Equals(Strings.WhereAreYourShips, StringComparison.InvariantCultureIgnoreCase))
            {
                // 如果回答是 "Where are your ships?"，则展示计算机舰队的船只
                foreach (var ship in _computerFleet.Ships)
                {
                    _io.WriteLine(ship);
                }
            }
            else
            {
                // 如果回答是 "yes"，则返回 true，否则返回 false
                return startResponse.Equals("yes", StringComparison.InvariantCultureIgnoreCase);
            }
        }
    }

    // PlayComputerTurn 方法
    private Winner? PlayComputerTurn()
    {
        // 获取计算机的射击次数
        var numberOfShots = _computerShotSelector.NumberOfShots;
        _io.Write(Strings.IHaveShots(numberOfShots));
        // 如果射击次数为 0，则返回人类获胜
        if (numberOfShots == 0) { return Winner.Human; }
        // 如果计算机可以瞄准所有剩余的方格，则返回计算机获胜
        if (_computerShotSelector.CanTargetAllRemainingSquares)
        {
            _io.Write(Streams.IHaveMoreShotsThanSquares);
            return Winner.Computer;
        }

        // 计算机舰队接收射击
        _humanFleet.ReceiveShots(
            _computerShotSelector.GetShots(_turnNumber),
            ship =>
            { 
                _io.Write(Strings.IHit(ship.Name));
                _computerShotSelector.RecordHit(ship, _turnNumber);
            });

        return null;
    }

    // PlayHumanTurn 方法
    private Winner? PlayHumanTurn()
    {
        // 获取人类的射击次数
        var numberOfShots = _humanShotSelector.NumberOfShots;
        _io.Write(Strings.YouHaveShots(numberOfShots));
        // 如果射击次数为 0，则返回计算机获胜
        if (numberOfShots == 0) { return Winner.Computer; }
        // 如果人类可以瞄准所有剩余的方格，则返回人类获胜
        if (_humanShotSelector.CanTargetAllRemainingSquares) 
        { 
            _io.WriteLine(Streams.YouHaveMoreShotsThanSquares);
            return Winner.Human;
        }
        
        // 人类舰队接收射击
        _computerFleet.ReceiveShots(
            _humanShotSelector.GetShots(_turnNumber), 
            ship => _io.Write(Strings.YouHit(ship.Name)));
        
        return null;
    }
}

```
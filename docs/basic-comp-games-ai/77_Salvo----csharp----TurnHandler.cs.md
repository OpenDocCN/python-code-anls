# `basic-computer-games\77_Salvo\csharp\TurnHandler.cs`

```py
{
    // 命名空间 Salvo 下的内部类 TurnHandler
    internal class TurnHandler
    {
        // 私有字段，用于读写操作
        private readonly IReadWrite _io;
        // 私有字段，表示玩家的舰队
        private readonly Fleet _humanFleet;
        // 私有字段，表示计算机的舰队
        private readonly Fleet _computerFleet;
        // 私有字段，表示玩家是否先手
        private readonly bool _humanStarts;
        // 私有字段，表示玩家的射击选择器
        private readonly HumanShotSelector _humanShotSelector;
        // 私有字段，表示计算机的射击选择器
        private readonly ComputerShotSelector _computerShotSelector;
        // 私有字段，表示执行回合动作的函数
        private readonly Func<Winner?> _turnAction;
        // 私有字段，表示回合数
        private int _turnNumber;

        // 公共构造函数，接受读写操作和随机数生成器
        public TurnHandler(IReadWrite io, IRandom random)
        {
            // 初始化读写操作字段
            _io = io;
            // 初始化计算机舰队对象
            _computerFleet = new Fleet(random);
            // 初始化玩家舰队对象
            _humanFleet = new Fleet(io);
            // 根据询问结果确定先手玩家，并设置回合动作函数
            _turnAction = AskWhoStarts()
                ? () => PlayHumanTurn() ?? PlayComputerTurn()
                : () => PlayComputerTurn() ?? PlayHumanTurn();
            // 初始化玩家射击选择器
            _humanShotSelector = new HumanShotSelector(_humanFleet, io);
            // 初始化计算机射击选择器
            _computerShotSelector = new ComputerShotSelector(_computerFleet, random, io);
        }

        // 公共方法，执行回合并返回胜利者
        public Winner? PlayTurn()
        {
            // 输出回合数
            _io.Write(Strings.Turn(++_turnNumber));
            // 执行回合动作函数并返回结果
            return _turnAction.Invoke();
        }

        // 私有方法，询问谁先手
        private bool AskWhoStarts()
        {
            // 无限循环，直到得到有效的回答
            while (true)
            {
                // 询问玩家谁先手
                var startResponse = _io.ReadString(Prompts.Start);
                // 如果回答是查看计算机舰队的指令
                if (startResponse.Equals(Strings.WhereAreYourShips, StringComparison.InvariantCultureIgnoreCase))
                {
                    // 输出计算机舰队的船只信息
                    foreach (var ship in _computerFleet.Ships)
                    {
                        _io.WriteLine(ship);
                    }
                }
                else
                {
                    // 返回玩家是否选择先手
                    return startResponse.Equals("yes", StringComparison.InvariantCultureIgnoreCase);
                }
            }
        }

        // 私有方法，执行计算机回合并返回胜利者
        private Winner? PlayComputerTurn()
    {
        // 获取计算机射击次数
        var numberOfShots = _computerShotSelector.NumberOfShots;
        // 输出计算机射击次数
        _io.Write(Strings.IHaveShots(numberOfShots));
        // 如果计算机射击次数为0，则返回人类获胜
        if (numberOfShots == 0) { return Winner.Human; }
        // 如果计算机可以瞄准所有剩余的方块
        if (_computerShotSelector.CanTargetAllRemainingSquares)
        {
            // 输出计算机拥有更多的射击次数
            _io.Write(Streams.IHaveMoreShotsThanSquares);
            // 返回计算机获胜
            return Winner.Computer;
        }
    
        // 人类舰队接收射击
        _humanFleet.ReceiveShots(
            // 获取人类射击选择器的射击
            _computerShotSelector.GetShots(_turnNumber),
            // 如果击中船只，则输出击中信息，并记录击中
            ship =>
            { 
                _io.Write(Strings.IHit(ship.Name));
                _computerShotSelector.RecordHit(ship, _turnNumber);
            });
    
        // 返回空值
        return null;
    }
    
    // 进行人类回合
    private Winner? PlayHumanTurn()
    {
        // 获取人类射击次数
        var numberOfShots = _humanShotSelector.NumberOfShots;
        // 输出人类射击次数
        _io.Write(Strings.YouHaveShots(numberOfShots));
        // 如果人类射击次数为0，则返回计算机获胜
        if (numberOfShots == 0) { return Winner.Computer; }
        // 如果人类可以瞄准所有剩余的方块
        if (_humanShotSelector.CanTargetAllRemainingSquares) 
        { 
            // 输出人类拥有更多的射击次数
            _io.WriteLine(Streams.YouHaveMoreShotsThanSquares);
            // 返回人类获胜
            return Winner.Human;
        }
        
        // 计算机舰队接收射击
        _computerFleet.ReceiveShots(
            // 获取计算机射击选择器的射击
            _humanShotSelector.GetShots(_turnNumber), 
            // 如果击中船只，则输出击中信息
            ship => _io.Write(Strings.YouHit(ship.Name)));
        
        // 返回空值
        return null;
    }
# 闭合前面的函数定义
```
# `15_Boxing\csharp\Round.cs`

```
    // 声明私有变量，用于存储玩家和对手的信息
    private readonly Boxer _player;
    private readonly Boxer _opponent;
    // 声明私有变量，用于存储当前回合数
    private readonly int _round;
    // 声明一个栈，用于存储动作
    private Stack<Action> _work = new();
    // 声明私有变量，用于存储玩家的攻击策略
    private readonly PlayerAttackStrategy _playerAttackStrategy;
    // 声明私有变量，用于存储对手的攻击策略
    private readonly OpponentAttackStrategy _opponentAttackStrategy;
    // 声明一个公共属性，用于表示游戏是否结束
    public bool GameEnded { get; private set; }

    // 构造函数，初始化玩家、对手和回合数，并将 ResetPlayers 方法压入栈中
    public Round(Boxer player, Opponent opponent, int round)
    {
        _player = player;
        _opponent = opponent;
        _round = round;
        _work.Push(ResetPlayers);
        _work.Push(CheckOpponentWin);  # 将 CheckOpponentWin 函数推入工作栈
        _work.Push(CheckPlayerWin);  # 将 CheckPlayerWin 函数推入工作栈

        void NotifyGameEnded() => GameEnded = true;  # 定义 NotifyGameEnded 函数，当游戏结束时将 GameEnded 设为 true
        _playerAttackStrategy = new PlayerAttackStrategy(player, opponent, NotifyGameEnded, _work);  # 创建玩家攻击策略对象，传入玩家、对手、NotifyGameEnded 函数和工作栈
        _opponentAttackStrategy = new OpponentAttackStrategy(opponent, player, NotifyGameEnded, _work);  # 创建对手攻击策略对象，传入对手、玩家、NotifyGameEnded 函数和工作栈
    }

    public void Start()
    {
        while (_work.Count > 0)  # 当工作栈不为空时
        {
            var action = _work.Pop();  # 从工作栈中弹出一个动作
            // This delay does not exist in the VB code but it makes a bit easier to follow the game.
            // I assume the computers at the time were slow enough
            // so that they did not need this delay...
            Thread.Sleep(300);  # 线程休眠 300 毫秒
            action();  # 执行弹出的动作
        }
    }
    # 检查对手是否获胜
    public void CheckOpponentWin()
    {
        # 如果对手获胜，打印出对手的名字和祝贺语，并将游戏结束标志设为真
        if (_opponent.IsWinner)
        {
            Console.WriteLine($"{_opponent} WINS (NICE GOING, {_opponent}).");
            GameEnded = true;
        }
    }

    # 检查玩家是否获胜
    public void CheckPlayerWin()
    {
        # 如果玩家获胜，打印出玩家的名字和祝贺语，并将游戏结束标志设为真
        if (_player.IsWinner)
        {
            Console.WriteLine($"{_player}  AMAZINGLY WINS!!");
            GameEnded = true;
        }
    }

    # 重置玩家状态
    private void ResetPlayers()
        {
            _player.ResetForNewRound();  # 重置玩家的状态和属性，为新的回合做准备
            _opponent.ResetForNewRound();  # 重置对手的状态和属性，为新的回合做准备
            _work.Push(RoundBegins);  # 将 RoundBegins 方法推入工作栈，准备执行下一个方法
        }

        private void RoundBegins()
        {
            Console.WriteLine();  # 输出空行
            Console.WriteLine($"ROUND {_round} BEGINS...");  # 输出当前回合开始的信息
            _work.Push(CheckRoundWinner);  # 将 CheckRoundWinner 方法推入工作栈，准备执行下一个方法
            for (var i = 0; i < 7; i++)
            {
                _work.Push(DecideWhoAttacks);  # 将 DecideWhoAttacks 方法推入工作栈，准备执行下一个方法
            }
        }

        private void CheckRoundWinner()
        {
            if (_opponent.DamageTaken > _player.DamageTaken)  # 检查对手受到的伤害是否大于玩家受到的伤害
        {
            // 如果玩家获胜，打印玩家获胜的消息，并记录玩家的胜利次数
            Console.WriteLine($"{_player} WINS ROUND {_round}");
            _player.RecordWin();
        }
        else
        {
            // 如果对手获胜，打印对手获胜的消息，并记录对手的胜利次数
            Console.WriteLine($"{_opponent} WINS ROUND {_round}");
            _opponent.RecordWin();
        }
    }

    private void DecideWhoAttacks()
    {
        // 决定谁进行攻击，根据骰子滚动的结果来决定
        _work.Push( GameUtils.RollSatisfies(10, x => x > 5) ? _opponentAttackStrategy.Attack : _playerAttackStrategy.Attack );
    }
}
```
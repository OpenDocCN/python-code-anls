# `basic-computer-games\15_Boxing\csharp\Round.cs`

```py
namespace Boxing;

class Round
{
    // 定义私有字段，分别表示玩家、对手、回合数、工作栈、玩家攻击策略、对手攻击策略
    private readonly Boxer _player;
    private readonly Boxer _opponent;
    private readonly int _round;
    private Stack<Action> _work = new();
    private readonly PlayerAttackStrategy _playerAttackStrategy;
    private readonly OpponentAttackStrategy _opponentAttackStrategy;

    // 定义公共属性，表示游戏是否结束
    public bool GameEnded { get; private set; }

    // 构造函数，初始化玩家、对手、回合数，并设置工作栈的初始动作
    public Round(Boxer player, Opponent opponent, int round)
    {
        _player = player;
        _opponent = opponent;
        _round = round;
        _work.Push(ResetPlayers);
        _work.Push(CheckOpponentWin);
        _work.Push(CheckPlayerWin);

        // 定义局部函数，用于通知游戏结束
        void NotifyGameEnded() => GameEnded = true;
        // 初始化玩家攻击策略
        _playerAttackStrategy = new PlayerAttackStrategy(player, opponent, NotifyGameEnded, _work);
        // 初始化对手攻击策略
        _opponentAttackStrategy = new OpponentAttackStrategy(opponent, player, NotifyGameEnded, _work);
    }

    // 开始回合
    public void Start()
    {
        // 循环执行工作栈中的动作
        while (_work.Count > 0)
        {
            var action = _work.Pop();
            // 添加延迟，以便更容易跟踪游戏进程
            // 假设当时的计算机速度较慢，因此不需要这个延迟...
            Thread.Sleep(300);
            action();
        }
    }

    // 检查对手是否获胜
    public void CheckOpponentWin()
    {
        if (_opponent.IsWinner)
        {
            Console.WriteLine($"{_opponent} WINS (NICE GOING, {_opponent}).");
            GameEnded = true;
        }
    }

    // 检查玩家是否获胜
    public void CheckPlayerWin()
    {
        if (_player.IsWinner)
        {
            Console.WriteLine($"{_player}  AMAZINGLY WINS!!");
            GameEnded = true;
        }
    }

    // 重置玩家和对手状态
    private void ResetPlayers()
    {
        _player.ResetForNewRound();
        _opponent.ResetForNewRound();
        _work.Push(RoundBegins);
    }

    // 回合开始
    private void RoundBegins()
    {
        // 输出空行
        Console.WriteLine();
        // 输出当前回合开始的信息
        Console.WriteLine($"ROUND {_round} BEGINS...");
        // 将 CheckRoundWinner 方法添加到工作栈中
        _work.Push(CheckRoundWinner);
        // 循环执行7次
        for (var i = 0; i < 7; i++)
        {
            // 将 DecideWhoAttacks 方法添加到工作栈中
            _work.Push(DecideWhoAttacks);
        }
    }
    
    // 检查回合的胜者
    private void CheckRoundWinner()
    {
        // 如果对手受到的伤害大于玩家受到的伤害
        if (_opponent.DamageTaken > _player.DamageTaken)
        {
            // 输出玩家赢得回合的信息
            Console.WriteLine($"{_player} WINS ROUND {_round}");
            // 记录玩家的胜利
            _player.RecordWin();
        }
        else
        {
            // 输出对手赢得回合的信息
            Console.WriteLine($"{_opponent} WINS ROUND {_round}");
            // 记录对手的胜利
            _opponent.RecordWin();
        }
    }
    
    // 决定谁进行攻击
    private void DecideWhoAttacks()
    {
        // 根据 GameUtils.RollSatisfies 方法的结果，将对手的攻击策略或玩家的攻击策略添加到工作栈中
        _work.Push( GameUtils.RollSatisfies(10, x => x > 5) ? _opponentAttackStrategy.Attack : _playerAttackStrategy.Attack );
    }
# 闭合前面的函数定义
```
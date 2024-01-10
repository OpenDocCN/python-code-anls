# `basic-computer-games\04_Awari\csharp\Game.cs`

```
namespace Awari;

public class Game
{
    // 获取玩家的6个坑的豆子数量
    public int[] PlayerPits => _beans[0..6];
    // 获取电脑的6个坑的豆子数量
    public int[] ComputerPits => _beans[7..13];
    // 获取玩家的家的豆子数量
    public int PlayerHome => _beans[_playerHome];
    // 获取电脑的家的豆子数量
    public int ComputerHome => _beans[_computerHome];

    // 判断游戏是否结束
    private bool IsDone =>
        PlayerPits.All(b => b == 0) // 如果玩家的所有坑都为空
     || ComputerPits.All(b => b == 0); // 或者如果电脑的所有坑都为空

    // 游戏状态
    public GameState State { get; private set; }

    // 重置游戏
    public void Reset()
    {
        State = GameState.PlayerMove;

        Array.Fill(_beans, _initialPitValue);
        _beans[_playerHome] = 0;
        _beans[_computerHome] = 0;

        _moveCount = 0;
        _notWonGameMoves[^1] = 0;
    }

    // 判断玩家移动是否合法
    public bool IsLegalPlayerMove(int move) =>
        move is > 0 and < 7
     && _beans[move - 1] > 0; // 数组是从0开始的，但移动是从1开始的

    // 玩家移动
    public void PlayerMove(int move) => MoveAndRegister(move - 1, _playerHome);

    // 电脑的回合
    public List<int> ComputerTurn()
    {
        // 保存电脑在一轮中的移动列表（1或2个）
        List<int> moves = new();

        moves.Add(ComputerMove()); // ComputerMove() 返回所做的移动

        // 只有在可能的情况下才进行第二次移动
        if (State == GameState.ComputerSecondMove)
            moves.Add(ComputerMove());

        return moves;
    }

    // 获取游戏结果
    public GameOutcome GetOutcome()
    {
        if (State != GameState.Done)
            throw new InvalidOperationException("Game is not yet done.");

        int difference = _beans[_playerHome] - _beans[_computerHome];
        var winner = difference switch
        {
            < 0 => GameWinner.Computer,
            0 => GameWinner.Draw,
            > 0 => GameWinner.Player,
        };

        return new GameOutcome(winner, Math.Abs(difference));
    }

    // 移动并记录
    private void MoveAndRegister(int pit, int homePosition)
    {
        // 记录上一个移动的豆子的位置
        int lastMovedBean = Move(_beans, pit, homePosition);
    
        // 将玩家和计算机的移动编码成一个“基数为6”的数字
        // 例如，如果玩家移动5，计算机移动2，玩家移动4，
        // 那么编码为((5 * 6) * 6) + (2 * 6) + 4 = 196
        if (pit > 6) pit -= 7;
        // 移动次数加一
        _moveCount++;
        if (_moveCount < 9)
            // 将当前移动编码到未赢得游戏的移动列表中
            _notWonGameMoves[^1] = _notWonGameMoves[^1] * 6 + pit;
    
        // 根据当前状态、游戏是否结束以及上一个移动的豆子是否移动到玩家的家位置来确定下一个状态
        State = (State, IsDone, lastMovedBean == homePosition) switch
        {
            (_, true, _) => GameState.Done,
            (GameState.PlayerMove, _, true) => GameState.PlayerSecondMove,
            (GameState.PlayerMove, _, false) => GameState.ComputerMove,
            (GameState.PlayerSecondMove, _, _) => GameState.ComputerMove,
            (GameState.ComputerMove, _, true) => GameState.ComputerSecondMove,
            (GameState.ComputerMove, _, false) => GameState.PlayerMove,
            (GameState.ComputerSecondMove, _, _) => GameState.PlayerMove,
            _ => throw new InvalidOperationException("Unexpected game state"),
        };
    
        // 如果游戏结束但计算机没有赢得游戏，则进行一些记录
        if (State == GameState.Done
         && _beans[_playerHome] >= _beans[_computerHome])
            // 为下一局游戏添加一个条目
            _notWonGameMoves.Add(0);
    }
    
    // 移动豆子的方法
    private static int Move(int[] beans, int pit, int homePosition)
    {
        // 获取需要移动的豆子数量
        int beansToMove = beans[pit];
        // 清空当前坑的豆子数量
        beans[pit] = 0;
    
        // 将当前坑的豆子按顺时针方向分配到其他坑中
        for (; beansToMove >= 1; beansToMove--)
        {
            // 如果坑的编号超过13，则回到1重新开始
            pit = (pit + 1) % 14;
            // 在当前坑中增加一颗豆子
            beans[pit]++;
        }
    
        // 如果最后一颗豆子落在一个空坑中，并且不是玩家或电脑的家，并且对面的坑不为空
        if (beans[pit] == 1 
         && pit is not _playerHome and not _computerHome 
         && beans[12 - pit] != 0) 
        {
            // 将最后一颗豆子和对面坑中的豆子一起移动到玩家的家中
            beans[homePosition] = beans[homePosition] + beans[12 - pit] + 1;
            beans[pit] = 0;
            beans[12 - pit] = 0;
        }
    
        // 返回最终落子的坑的编号
        return pit;
    }
    
    // 电脑移动的方法
    private int ComputerMove()
    {
        // 确定电脑的移动
        int move = DetermineComputerMove();
        // 移动并记录移动
        MoveAndRegister(move, homePosition: _computerHome);
    
        // 将数组索引（在7到12之间）转换为坑的编号（在1到6之间）并返回结果
        return move - 6;
    }
    
    // 确定电脑移动的方法
    private int DetermineComputerMove()
    {
        int bestScore = -99;
        int move = 0;
    
        // 对于电脑可能的每个移动，模拟移动以计算得分并选择最佳移动
        for (int j = 7; j < 13; j++)
        {
            if (_beans[j] <= 0)
                continue;
    
            int score = SimulateMove(j);
    
            if (score >= bestScore)
            {
                move = j;
                bestScore = score;
            }
        }
    
        return move;
    }
    
    // 模拟移动的方法
    private int SimulateMove(int move)
    }
    
    // 计算最佳下一个玩家移动的得分
    private static int ScoreBestNextPlayerMove(int[] hypotheticalBeans)
    {
        // 初始化最佳得分为0
        int bestScore = 0;

        // 遍历6个可能的下一步
        for (int i = 0; i < 6; i++)
        {
            // 如果当前位置的豆子数量小于等于0，则跳过
            if (hypotheticalBeans[i] <= 0)
                continue;

            // 计算下一个玩家的得分
            int score = ScoreNextPlayerMove(hypotheticalBeans, i);

            // 如果得分比当前最佳得分大，则更新最佳得分
            if (score > bestScore)
                bestScore = score;
        }

        // 返回最佳得分
        return bestScore;
    }

    // 计算下一个玩家的得分
    private static int ScoreNextPlayerMove(int[] hypotheticalBeans, int move)
    {
        // 计算最后一个豆子的落点
        int target = hypotheticalBeans[move] + move;
        int score = 0;

        // 如果落点超出范围，则表示玩家将豆子放入自己的坑中，得分加1
        if (target > 13)
        {
            // 防止超出我们拥有的坑的数量
            target %= 14;
            score = 1;
        }

        // 如果玩家的移动导致落入空坑中，则将对面坑的值加到得分中
        if (hypotheticalBeans[target] == 0 && target is not _playerHome and not _computerHome)
            score += hypotheticalBeans[12 - target];

        return score;
    }

    // 玩家的家的位置
    private const int _playerHome = 6;
    // 电脑的家的位置
    private const int _computerHome = 13;
    // 初始坑的值
    private const int _initialPitValue = 3;

    // 14个坑的豆子数量
    private readonly int[] _beans = new int[14];
    // 未赢得游戏的移动列表（平局或失败）
    private readonly List<int> _notWonGameMoves = new() { 0 };
    // 移动次数
    private int _moveCount;
# 结束大括号，表示枚举和记录结构定义的结束
}

# 定义游戏状态的枚举
public enum GameState
{
    # 玩家移动状态
    PlayerMove,
    # 玩家第二次移动状态
    PlayerSecondMove,
    # 电脑移动状态
    ComputerMove,
    # 电脑第二次移动状态
    ComputerSecondMove,
    # 游戏结束状态
    Done,
}

# 定义游戏胜利者的枚举
public enum GameWinner
{
    # 玩家胜利
    Player,
    # 电脑胜利
    Computer,
    # 平局
    Draw,
}

# 定义游戏结果的记录结构
public record struct GameOutcome(GameWinner Winner, int Difference);
```
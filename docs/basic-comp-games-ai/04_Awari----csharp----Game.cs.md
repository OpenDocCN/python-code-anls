# `04_Awari\csharp\Game.cs`

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

    // 判断游戏是否结束的布尔值
    private bool IsDone =>
        PlayerPits.All(b => b == 0) // 如果玩家的所有坑都为空
     || ComputerPits.All(b => b == 0); // 或者如果电脑的所有坑都为空

    // 游戏的状态
    public GameState State { get; private set; }

    // 重置游戏
    public void Reset()
    {
        State = GameState.PlayerMove; // 设置游戏状态为玩家移动

        Array.Fill(_beans, _initialPitValue); // 用初始坑的豆子数量填充豆子数组
        _beans[_playerHome] = 0;  // 将玩家的家中的豆子数量设置为0
        _beans[_computerHome] = 0;  // 将计算机的家中的豆子数量设置为0

        _moveCount = 0;  // 将移动计数器设置为0
        _notWonGameMoves[^1] = 0;  // 将未赢得游戏的移动设置为0
    }

    public bool IsLegalPlayerMove(int move) =>
        move is > 0 and < 7  // 判断移动是否在1到6之间
     && _beans[move - 1] > 0;  // 判断移动位置上是否有豆子，数组是从0开始的，但是移动是从1开始的

    public void PlayerMove(int move) => MoveAndRegister(move - 1, _playerHome);  // 玩家移动，将移动位置和玩家的家作为参数传递给MoveAndRegister函数

    public List<int> ComputerTurn()
    {
        // 保持计算机在单次回合中所做的移动的列表（1或2）
        List<int> moves = new();

        moves.Add(ComputerMove());  // 将计算机所做的移动添加到列表中，ComputerMove()返回所做的移动
        // 只有在第二次移动可行时，才执行
        if (State == GameState.ComputerSecondMove)
            moves.Add(ComputerMove()); // 将计算机的移动添加到移动列表中

        return moves; // 返回移动列表
    }

    public GameOutcome GetOutcome()
    {
        if (State != GameState.Done) // 如果游戏状态不是已完成
            throw new InvalidOperationException("Game is not yet done."); // 抛出无效操作异常，表示游戏尚未完成

        int difference = _beans[_playerHome] - _beans[_computerHome]; // 计算玩家和计算机家中豆子数量的差值
        var winner = difference switch // 根据差值进行判断
        {
            < 0 => GameWinner.Computer, // 如果差值小于0，计算机获胜
            0 => GameWinner.Draw, // 如果差值等于0，平局
            > 0 => GameWinner.Player, // 如果差值大于0，玩家获胜
        };
        return new GameOutcome(winner, Math.Abs(difference));
    }  # 返回一个新的GameOutcome对象，包括获胜者和差值的绝对值

    private void MoveAndRegister(int pit, int homePosition)
    {
        int lastMovedBean = Move(_beans, pit, homePosition);  # 调用Move方法移动豆子，并将结果赋给lastMovedBean变量

        // encode moves by player and computer into a 'base 6' number
        // e.g. if the player moves 5, the computer moves 2, and the player moves 4,
        // that would be encoded as ((5 * 6) * 6) + (2 * 6) + 4 = 196
        if (pit > 6) pit -= 7;  # 如果坑位大于6，则减去7
        _moveCount++;  # 移动次数加1
        if (_moveCount < 9)
            _notWonGameMoves[^1] = _notWonGameMoves[^1] * 6 + pit;  # 如果移动次数小于9，则将当前移动编码为base 6的数字并存储

        // determine next state based on current state, whether the game's done, and whether the last moved bean moved
        // into the player's home position
        State = (State, IsDone, lastMovedBean == homePosition) switch  # 根据当前状态、游戏是否结束以及最后移动的豆子是否移动到玩家的家位置来确定下一个状态
        {
            (_, true, _) => GameState.Done,  # 如果游戏结束，则状态为Done
            (GameState.PlayerMove, _, true) => GameState.PlayerSecondMove,  // 如果游戏状态为玩家移动且是玩家的第一次移动，则返回玩家的第二次移动状态
            (GameState.PlayerMove, _, false) => GameState.ComputerMove,  // 如果游戏状态为玩家移动且不是玩家的第一次移动，则返回计算机移动状态
            (GameState.PlayerSecondMove, _, _) => GameState.ComputerMove,  // 如果游戏状态为玩家的第二次移动，则返回计算机移动状态
            (GameState.ComputerMove, _, true) => GameState.ComputerSecondMove,  // 如果游戏状态为计算机移动且是计算机的第一次移动，则返回计算机的第二次移动状态
            (GameState.ComputerMove, _, false) => GameState.PlayerMove,  // 如果游戏状态为计算机移动且不是计算机的第一次移动，则返回玩家移动状态
            (GameState.ComputerSecondMove, _, _) => GameState.PlayerMove,  // 如果游戏状态为计算机的第二次移动，则返回玩家移动状态
            _ => throw new InvalidOperationException("Unexpected game state"),  // 如果游戏状态不符合以上任何情况，则抛出异常
        };

        // 如果游戏结束但计算机没有获胜，则进行一些记录
        if (State == GameState.Done
         && _beans[_playerHome] >= _beans[_computerHome])
            // 添加下一局游戏的记录
            _notWonGameMoves.Add(0);
    }

    private static int Move(int[] beans, int pit, int homePosition)
    {
        int beansToMove = beans[pit];  // 获取选中坑中的豆子数量
        beans[pit] = 0;  // 将选中坑中的豆子数量置为0
        // 将在坑中的豆子添加到其他坑中，顺时针移动游戏板
        for (; beansToMove >= 1; beansToMove--)
        {
            // 如果坑超过13，则循环
            pit = (pit + 1) % 14;

            beans[pit]++;
        }

        if (beans[pit] == 1 // 如果最后一颗豆子播种在一个空坑中
         && pit is not _playerHome and not _computerHome // 并且不是任何玩家的家
         && beans[12 - pit] != 0) // 并且对面的坑不是空的
        {
            // 将最后一个播种的坑和对面坑中的豆子移动到玩家的家中
            beans[homePosition] = beans[homePosition] + beans[12 - pit] + 1;
            beans[pit] = 0;
            beans[12 - pit] = 0;
        }
        return pit;
    }
    # 返回 pit 变量的值

    private int ComputerMove()
    {
        int move = DetermineComputerMove();
        MoveAndRegister(move, homePosition: _computerHome);
        # 调用 DetermineComputerMove() 方法获取计算机的移动位置，并将其传递给 MoveAndRegister() 方法进行注册

        // the result is only used to return it to the application, so translate it from an array index (between 7 and
        // 12) to a pit number (between 1 and 6)
        # 结果仅用于返回给应用程序，因此将其从数组索引（介于7和12之间）转换为坑位编号（介于1和6之间）
        return move - 6;
    }

    private int DetermineComputerMove()
    {
        int bestScore = -99;
        int move = 0;
        # 初始化最佳分数和移动位置

        // for each of the computer's possible moves, simulate them to calculate a score and pick the best one
        # 对于计算机的每个可能移动，模拟它们以计算得分并选择最佳移动
        for (int j = 7; j < 13; j++)
        {
            # 如果豆子数量小于等于0，则跳过当前循环
            if (_beans[j] <= 0)
                continue;

            # 模拟移动，计算移动后的得分
            int score = SimulateMove(j);

            # 如果得分大于等于最佳得分，则更新最佳得分和移动位置
            if (score >= bestScore)
            {
                move = j;
                bestScore = score;
            }
        }

        # 返回最佳移动位置
        return move;
    }

    # 模拟移动并返回得分
    private int SimulateMove(int move)
    {
        # 复制当前状态的豆子数量，以便安全地进行操作
        var hypotheticalBeans = new int[14];
        _beans.CopyTo(hypotheticalBeans, 0); // 将_beans数组的内容复制到hypotheticalBeans数组中

        // 模拟在我们的复制中进行移动
        Move(hypotheticalBeans, move, homePosition: _computerHome); // 在复制的数组中模拟移动

        // 确定玩家在此之后可能做出的“最佳”移动（对他们来说最好，而不是对计算机来说）
        int score = ScoreBestNextPlayerMove(hypotheticalBeans); // 计算玩家下一步可能的最佳得分

        // 通过计算移动后我们将领先多少，并减去玩家的下一步得分来评分此移动
        score = hypotheticalBeans[_computerHome] - hypotheticalBeans[_playerHome] - score; // 计算得分

        // 我们之前在平局/输掉的游戏中见过当前的移动序列吗？在8次移动之后，我们不太可能找到任何匹配，因为游戏会分叉。而且我们也没有空间来存储那么多的移动。
        if (_moveCount < 8)
        {
            int translatedMove = move - 7;  // 将移动从7到12转换为0到5

            // 如果游戏中的前两个移动是1和2，而这个假设的第三个移动将是3，movesSoFar将是(1 * 36) + (2 * 6) + 3 = 51
            int movesSoFar = _notWonGameMoves[^1] * 6 + translatedMove;
            // 计算到目前为止的移动次数，将上一次的移动次数乘以6再加上当前移动的次数

            // 由于我们将移动存储为“基数6”数字，因此我们需要将存储的移动除以6的幂
            // 假设我们有一个存储的输掉游戏，其中连续的移动是1到8，存储的值将是：
            // 8 + (7 * 6) + (6 * 36) + (5 * 216) + (4 * 1296) + (3 * 7776) + (2 * 46656) + (1 * 279936) = 403106
            // 要找出前三个移动，我们需要除以7776，结果为51.839...
            double divisor = Math.Pow(6.0, 7 - _moveCount);

            foreach (int previousGameMoves in _notWonGameMoves)
                // 如果到目前为止的移动组合最终导致平局/失败，则给予较低的分数
                // 请注意，这可能会发生多次
                if (movesSoFar == (int) (previousGameMoves / divisor + 0.1))
                    score -= 2;
        }
        return score;
    }

    private static int ScoreBestNextPlayerMove(int[] hypotheticalBeans)
    {
        int bestScore = 0;  // 初始化最佳分数为0

        for (int i = 0; i < 6; i++)  // 循环6次，遍历hypotheticalBeans数组
        {
            if (hypotheticalBeans[i] <= 0)  // 如果hypotheticalBeans数组中第i个元素小于等于0，跳过当前循环
                continue;

            int score = ScoreNextPlayerMove(hypotheticalBeans, i);  // 调用ScoreNextPlayerMove方法计算当前玩家移动的得分

            if (score > bestScore)  // 如果当前得分大于最佳分数
                bestScore = score;  // 更新最佳分数为当前得分
        }

        return bestScore;  // 返回最佳分数
    }

    private static int ScoreNextPlayerMove(int[] hypotheticalBeans, int move)  // 定义ScoreNextPlayerMove方法，计算下一个玩家的移动得分
    {
        // figure out where the last bean will land
```
在这段代码中，注释解释了每个语句的作用，使得其他程序员能够更容易地理解代码的逻辑和功能。
        int target = hypotheticalBeans[move] + move;  // 计算目标位置，加上当前位置的豆子数
        int score = 0;  // 初始化得分为0

        // 如果目标位置超过13，表示玩家将在自己的坑中添加豆子，这是好事
        if (target > 13)
        {
            // 防止超出我们拥有的坑的数量
            target %= 14;  // 对14取模，防止超出范围
            score = 1;  // 得分加1
        }

        // 如果玩家的移动导致落入一个空坑，将对面的坑中的豆子数加到得分中
        if (hypotheticalBeans[target] == 0 && target is not _playerHome and not _computerHome)
            score += hypotheticalBeans[12 - target];  // 将对面坑中的豆子数加到得分中

        return score;  // 返回得分
    }

    private const int _playerHome = 6;  // 玩家的家的位置
    private const int _computerHome = 13;  // 电脑的家的位置
private const int _initialPitValue = 3;  // 设置初始坑的初始值为3

private readonly int[] _beans = new int[14];  // 创建一个包含14个整数的数组，用于存储每个坑中的豆子数量
private readonly List<int> _notWonGameMoves = new() { 0 };  // 创建一个整数列表，用于存储未赢得游戏的移动步数，初始值为0，表示平局或失败
private int _moveCount;  // 用于存储游戏中的移动步数

public enum GameState  // 定义游戏状态的枚举类型
{
    PlayerMove,  // 玩家移动
    PlayerSecondMove,  // 玩家第二次移动
    ComputerMove,  // 电脑移动
    ComputerSecondMove,  // 电脑第二次移动
    Done,  // 游戏结束
}

public enum GameWinner  // 定义游戏获胜者的枚举类型
{
    Player,  // 玩家获胜
    Computer,  // 电脑获胜
}
    Draw,  # 定义一个变量或者函数名为Draw

}

public record struct GameOutcome(GameWinner Winner, int Difference);  # 定义一个名为GameOutcome的记录结构，包含Winner和Difference两个属性
```
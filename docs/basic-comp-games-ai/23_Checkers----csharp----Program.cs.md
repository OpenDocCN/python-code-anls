# `basic-computer-games\23_Checkers\csharp\Program.cs`

```
/*********************************************************************************
 * CHECKERS
 * ported from BASIC https://www.atariarchives.org/basicgames/showpage.php?page=41
 *
 * Porting philosophy
 * 1) Adhere to the original as much as possible
 * 2) Attempt to be understandable by Novice progammers
 *
 * There are no classes or Object Oriented design patterns used in this implementation.
 * Everything is written procedurally, using only top-level functions. Hopefully, this
 * will be approachable for someone who wants to learn C# syntax without experience with
 * Object Oriented concepts. Similarly, basic data structures have been chosen over more
 * powerful collection types.  Linq/lambda syntax is also excluded.
 *
 * C# Concepts contained in this example:
 *    Loops (for, foreach, while, and do)
 *    Multidimensional arrays
 *    Tuples
 *    Nullables
 *    IEnumerable (yield return / yield break)
 *
 * The original had multiple implementations of logic, like determining valid jump locations.
 * This has been refactored to reduce unnecessary code duplication.
 *********************************************************************************/
#region Display functions
// 跳过指定行数的输出
void SkipLines(int count)
{
    for (int i = 0; i < count; i++)
    {
        Console.WriteLine();
    }
}

// 打印棋盘状态
void PrintBoard(int[,] state)
{
    SkipLines(3);
    // 从上到下遍历棋盘的每一行
    for (int y = 7; y >= 0; y--)
    {
        # 循环遍历 x 坐标，范围是 0 到 7
        for (int x = 0; x < 8; x++)
        {
            # 根据当前状态值输出不同的字符
            switch(state[x,y])
            {
                # 当状态为 -2 时输出 "X*"
                case -2:
                    Console.Write("X*");
                    break;
                # 当状态为 -1 时输出 "X "
                case -1:
                    Console.Write("X ");
                    break;
                # 当状态为 0 时输出 ". "
                case 0:
                    Console.Write(". ");
                    break;
                # 当状态为 1 时输出 "O "
                case 1:
                    Console.Write("O ");
                    break;
                # 当状态为 2 时输出 "O*"
                case 2:
                    Console.Write("O*");
                    break;
            }
            # 输出空格
            Console.Write("   ");
        }
        # 输出换行
        Console.WriteLine();
    }
}

// 将文本居中打印在控制台
void WriteCenter(string text)
{
    // 定义每行的长度
    const int LineLength = 80;
    // 计算需要添加的空格数，使得文本居中
    var spaces = (LineLength - text.Length) / 2;
    // 在控制台打印居中的文本
    Console.WriteLine($"{"".PadLeft(spaces)}{text}");
}

// 当计算机获胜时打印消息
void ComputerWins()
{
    Console.WriteLine("I WIN.");
}
// 当玩家获胜时打印消息
void PlayerWins()
{
    Console.WriteLine("YOU WIN.");
}

// 打印游戏介绍
void WriteIntroduction()
{
    // 居中打印游戏名称
    WriteCenter("CHECKERS");
    // 居中打印游戏信息
    WriteCenter("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
    // 跳过3行
    SkipLines(3);
    // 打印游戏规则和提示
    Console.WriteLine("THIS IS THE GAME OF CHECKERS. THE COMPUTER IS X,");
    Console.WriteLine("AND YOU ARE O.  THE COMPUTER WILL MOVE FIRST.");
    Console.WriteLine("SQUARES ARE REFERRED TO BY A COORDINATE SYSTEM.");
    Console.WriteLine("(0,0) IS THE LOWER LEFT CORNER");
    Console.WriteLine("(0,7) IS THE UPPER LEFT CORNER");
    Console.WriteLine("(7,0) IS THE LOWER RIGHT CORNER");
    Console.WriteLine("(7,7) IS THE UPPER RIGHT CORNER");
    Console.WriteLine("THE COMPUTER WILL TYPE '+TO' WHEN YOU HAVE ANOTHER");
    Console.WriteLine("JUMP.  TYPE TWO NEGATIVE NUMBERS IF YOU CANNOT JUMP.");
    // 跳过3行
    SkipLines(3);
}
#endregion

// 以下是状态验证函数

// 判断点是否超出边界
bool IsPointOutOfBounds(int x)
{
    return x < 0 || x > 7;
}

// 判断位置是否超出边界
bool IsOutOfBounds((int x, int y) position)
{
    return IsPointOutOfBounds(position.x) || IsPointOutOfBounds(position.y);
}

// 判断是否为跳跃移动
bool IsJumpMove((int x, int y) from, (int x, int y) to)
{
    return Math.Abs(from.y - to.y) == 2;
}

// 验证玩家移动是否有效
bool IsValidPlayerMove(int[,] state, (int x, int y) from, (int x, int y) to)
{
    if (state[to.x, to.y] != 0)
    {
        return false;
    }
    var deltaX = Math.Abs(to.x - from.x);
    var deltaY = Math.Abs(to.y - from.y);
    if (deltaX != 1 && deltaX != 2)
    {
        return false;
    }
    if (deltaX != deltaY)
    {
        return false;
    }
    if (state[from.x, from.y] == 1 && Math.Sign(to.y - from.y) <= 0)
    {
        // 只有国王可以向下移动
        return false;
    }
    if (deltaX == 2)
    {
        // 获取从起始位置到目标位置的跳跃棋子
        var jump = GetJumpedPiece(from, to);
        // 如果跳跃位置上有棋子
        if (state[jump.x, jump.y] >= 0)
        {
            // 没有有效的棋子可以跳
            return false;
        }
    }
    // 返回 true
    return true;
}

bool CheckForComputerWin(int[,] state)
{
    bool playerAlive = false;
    foreach (var piece in state)
    {
        if (piece > 0)
        {
            playerAlive = true;
            break;
        }
    }
    return !playerAlive;
}

bool CheckForPlayerWin(int[,] state)
{
    bool computerAlive = false;
    foreach (var piece in state)
    {
        if (piece < 0)
        {
            computerAlive = true;
            break;
        }
    }
    return !computerAlive;
}
#endregion

#region Board "arithmetic"
/// <summary>
/// Get the Coordinates of a jumped piece
/// </summary>
(int x, int y) GetJumpedPiece((int x, int y) from, (int x, int y) to)
{
    var midX = (to.x + from.x) / 2;
    var midY = (to.y + from.y) / 2;
    return (midX, midY);
}
/// <summary>
/// Apply a directional vector "direction" to location "from"
/// return resulting location
/// direction will contain: (-1,-1), (-1, 1), ( 1,-1), ( 1, 1)
/// /// </summary>
(int x, int y) GetLocation((int x , int y) from, (int x, int y) direction)
{
    return (x: from.x + direction.x, y: from.y + direction.y);
}
#endregion

#region State change functions
/// <summary>
/// Alter current "state" by moving a piece from "from" to "to"
/// This method does not verify that the move being made is valid
/// This method works for both player moves and computer moves
/// </summary>
int[,] ApplyMove(int[,] state, (int x, int y) from, (int x, int y) to)
{
    state[to.x, to.y] = state[from.x, from.y];  // 将起始位置的棋子移动到目标位置
    state[from.x, from.y] = 0;  // 将起始位置清空

    if (IsJumpMove(from, to))  // 如果是跳跃移动
    {
        // a jump was made
        // remove the jumped piece from the board
        var jump = GetJumpedPiece(from, to);  // 获取被跳跃的棋子的坐标
        state[jump.x, jump.y] = 0;  // 清空被跳跃的棋子
    }
    return state;  // 返回新的状态
}
/// <summary>
/// At the end of a turn (either player or computer) check to see if any pieces
/// reached the final row.  If so, change them to kings (crown)
/// </summary>
int[,] CrownKingPieces(int[,] state)
{
    for (int x = 0; x < 8; x++)
    {
        // 检查底部行是否计算机在其中有一个棋子
        if (state[x, 0] == -1)
        {
            state[x, 0] = -2;
        }
        // 检查顶部行是否玩家在其中有一个棋子
        if (state[x, 7] == 1)
        {
            state[x, 7] = 2;
        }
    }
    // 返回更新后的状态
    return state;
#endregion

#region Computer Logic
/// <summary>
/// 给定当前位置 "from"，确定在给定向量 "direction" 中是否存在移动
/// direction 将包含：(-1,-1), (-1, 1), ( 1,-1), ( 1, 1)
/// 如果在这个方向上没有移动，则返回 "null"
/// </summary>
(int x, int y)? GetCandidateMove(int[,] state, (int x, int y) from, (int x, int y) direction)
{
    // 获取目标位置
    var to = GetLocation(from, direction);
    // 如果目标位置超出边界，则返回 null
    if (IsOutOfBounds(to))
        return null;
    // 如果目标位置上有棋子
    if (state[to.x, to.y] > 0)
    {
        // 可能是跳跃
        to = GetLocation(to, direction);
        // 如果跳跃后位置超出边界，则返回 null
        if (IsOutOfBounds(to))
            return null;
    }
    // 如果目标位置已经被占据
    if (state[to.x, to.y] != 0)
        // 返回 null
        return null;

    // 返回目标位置
    return to;
}
/// <summary>
/// 计算给定潜在移动的等级
/// 等级值越高，移动被认为越好
/// </summary>
int RankMove(int[,] state, (int x, int y) from, (int x, int y) to)
{
    int rank = 0;

    // 如果目标位置的 y 坐标为 0，且当前位置的棋子为 -1
    if (to.y == 0 && state[from.x, from.y] == -1)
    {
        // 获取国王
        rank += 2;
    }
    // 如果是跳跃移动
    if (IsJumpMove(from, to))
    {
        // 进行跳跃
        rank += 5;
    }
    // 如果当前位置的 y 坐标为 7
    if (from.y == 7)
    {
        // 离开起始行
        rank -= 2;
    }
    // 如果目标位置的 x 坐标为 0 或 7
    if (to.x == 0 || to.x == 7)
    {
        // 移动到棋盘边缘
        rank += 1;
    }
    // 查看潜在目的地前一行的情况
    for (int c = -1; c <=1; c+=2)
    {
        // 获取目标位置前方的位置
        var inFront = GetLocation(to, (c, -1));
        // 如果超出边界，则继续下一次循环
        if (IsOutOfBounds(inFront))
            continue;
        // 如果目标位置前方的状态小于 0
        if (state[inFront.x, inFront.y] < 0)
        {
            // 受到我方棋子保护
            rank++;
            // 继续下一次循环
            continue;
        }
        // 获取目标位置后方的位置
        var inBack = GetLocation(to, (-c, 1));
        // 如果超出边界，则继续下一次循环
        if (IsOutOfBounds(inBack))
        {
            continue;
        }
        // 如果目标位置前方的状态大于 0，并且目标位置后方的状态等于 0，或者目标位置后方等于起始位置
        if ((state[inFront.x, inFront.y] > 0) &&
            (state[inBack.x, inBack.y] == 0) || (inBack == from))
        {
            // 对方可以跳过我们
            rank -= 2;
        }
    }
    // 返回计算出的 rank 值
    return rank;
// 返回给定棋子“from”可能移动的枚举
// 如果没有可移动的步骤，则枚举将为空
IEnumerable<(int x, int y)> GetPossibleMoves(int[,] state, (int x, int y) from)
{
    int maxB; // 最大的b值
    switch (state[from.x, from.y])
    {
        case -2:
            // 国王也可以后退
            maxB = 1;
            break;
        case -1:
            maxB = -1;
            break;
        default:
            // 不是我们的棋子
            yield break;
    }

    for (int a = -1; a <= 1; a += 2)
    {
        // a
        // -1 = 左
        // +1 = 右
        for (int b = -1; b <= maxB; b += 2)
        {
            // b
            // -1 = 前进
            // +1 = 后退（只有国王允许做这个动作）
            var to = GetCandidateMove(state, from, (a, b));
            if (to == null)
            {
                // 在这个方向上没有有效的移动
                continue;
            }
            yield return to.Value;
        }
    }
}

// 从候选移动列表“possibleMoves”中确定最佳移动
// 如果无法移动，则返回“null”
((int x, int y) from, (int x, int y) to)? GetBestMove(int[,] state, IEnumerable<((int x, int y) from, (int x, int y) to)> possibleMoves)
{
    int? bestRank = null; // 最佳等级
    ((int x, int y) from, (int x, int y) to)? bestMove = null; // 最佳移动

    foreach (var move in possibleMoves)
    {
        int rank = RankMove(state, move.from, move.to);

        if (bestRank == null || rank > bestRank)
        {
            bestRank = rank;
            bestMove = move;
        }
    }

    return bestMove;
}

// 检查整个棋盘并记录所有可能的移动
// 返回找到的最佳移动，如果存在的话
// 如果找不到移动，则返回“null”
((int x, int y) from, (int x, int y) to)? CalculateMove(int[,] state)
{
    # 创建一个存储可能移动的列表，每个元素是一个元组，包含起始位置和目标位置
    var possibleMoves = new List<((int x, int y) from, (int x, int y) to)>();
    # 遍历棋盘上的每一个位置
    for (int x = 0; x < 8; x++)
    {
        for (int y = 0; y < 8; y++)
        {
            # 记录当前位置作为起始位置
            var from = (x, y);
            # 获取从当前位置可能的所有目标位置
            foreach (var to in GetPossibleMoves(state, from))
            {
                # 将起始位置和目标位置组成元组，添加到可能移动的列表中
                possibleMoves.Add((from, to));
            }
        }
    }
    # 获取最佳移动
    var bestMove = GetBestMove(state, possibleMoves);
    # 返回最佳移动
    return bestMove;
}
/// <summary>
/// The logic behind the Computer's turn
/// Look for valid moves and possible subsequent moves
/// </summary>
(bool moveMade, int[,] state) ComputerTurn(int[,] state)
{
    // Get best move available
    var move = CalculateMove(state);
    if (move == null)
    {
        // No move can be made
        return (false, state);
    }
    var from = move.Value.from;
    Console.Write($"FROM {from.x} {from.y} ");
    // Continue to make moves until no more valid moves can be made
    while (move != null)
    {
        var to = move.Value.to;
        Console.WriteLine($"TO {to.x} {to.y}");
        state = ApplyMove(state, from, to);
        if (!IsJumpMove(from, to))
            break;

        // check for double / triple / etc. jump
        var possibleMoves = new List<((int x, int y) from, (int x, int y) to)>();
        from = to;
        foreach (var candidate in GetPossibleMoves(state, from))
        {
            if (IsJumpMove(from, candidate))
            {
                possibleMoves.Add((from, candidate));
            }
        }
        // Get best jump move
        move = GetBestMove(state, possibleMoves);
    }
    // apply crowns to any new Kings
    state = CrownKingPieces(state);
    return (true, state);
}
#endregion

#region Player Logic
/// <summary>
/// Get input from the player in the form "x,y" where x and y are integers
/// If invalid input is received, return null
/// If input is valid, return the coordinate of the location
/// </summary>
(int x, int y)? GetCoordinate(string prompt)
{
    Console.Write(prompt + "? ");
    var input = Console.ReadLine();
    // split the string into multiple parts
    var parts = input?.Split(",");
    if (parts?.Length != 2)
        // must be exactly 2 parts
        return null;
    int x;
    if (!int.TryParse(parts[0], out x))
        // first part is not a number
        return null;
    int y;
    # 如果无法将第二部分转换为整数，则返回空值
    if (!int.TryParse(parts[1], out y))
        //second part is not a number
        return null;
    
    # 返回包含 x 和 y 的元组
    return (x, y);
// 结束 GetPlayerMove 函数的定义
}

/// <summary>
/// 从玩家获取移动
/// 返回一个表示有效移动的“from”和“to”的元组
/// </summary>
((int x, int y) from, (int x,int y) to) GetPlayerMove(int[,] state)
{
    // 原始程序在用户输入方面存在一些问题
    // 1) 原始程序中存在最小的数据完整性检查：
    //    a) FROM 位置必须由玩家拥有
    //    b) TO 位置必须为空
    //    c) FROM 和 TO 的 x 坐标必须相差不超过2个方块
    //    d) FROM 和 TO 的 y 坐标必须与 x 坐标相同
    //    没有检查方向，是否跳跃有效，或者甚至是否棋子移动。
    // 2) 一旦选择了有效的 FROM，必须选择 TO。
    //    如果没有有效的 TO 位置，你将被软锁定
    // 这种方法故意与原始方法不同，但尽可能保持了原始意图
    // 1) 选择一个 FROM 位置
    // 2) 如果 FROM 无效，则返回步骤 1
    // 3) 选择一个 TO 位置
    // 4) 如果 TO 无效或者暗示的移动无效，则返回步骤 1

    // 目前仍然没有办法让玩家指示无法进行移动
    // 这符合原始逻辑，但是可以考虑重构

    do
    {
        // 从玩家获取“FROM”坐标
        var from = GetCoordinate("FROM");
        // 如果 FROM 不为空且不越界且 FROM 位置上有棋子
        if ((from != null)
            && !IsOutOfBounds(from.Value)
            && (state[from.Value.x, from.Value.y] > 0))
        {
            // 我们有一个有效的“FROM”位置
            // 从玩家获取“TO”坐标
            var to = GetCoordinate("TO");
            // 如果 TO 不为空且不越界且移动有效
            if ((to != null)
                && !IsOutOfBounds(to.Value)
                && IsValidPlayerMove(state, from.Value, to.Value))
            {
                // 我们有一个有效的“TO”位置
                // 返回移动的“from”和“to”
                return (from.Value, to.Value);
            }
        }
    } while (true);
}

/// <summary>
/// 如果玩家可以/想要进行跳跃，则从玩家获取后续跳跃
/// 如果玩家跳跃，则返回一个移动（“from”，“to”）
/// returns null if a player does not make another move
/// The player must input negative numbers for the coordinates to indicate
/// that no more moves are to be made.  This matches the original implementation
/// </summary>
((int x, int y) from, (int x, int y) to)? GetPlayerSubsequentJump(int[,] state, (int x, int y) from)
{
    do
    {
        var to = GetCoordinate("+TO");  // 获取玩家输入的目标位置坐标
        if ((to != null)  // 如果目标位置不为空
            && !IsOutOfBounds(to.Value)  // 并且目标位置不超出边界
            && IsValidPlayerMove(state, from, to.Value)  // 并且目标位置是有效的玩家移动
            && IsJumpMove(from, to.Value))  // 并且是跳跃移动
        {
            // we have a valid "to" location
            return (from, to.Value); ;  // 返回有效的移动坐标
        }

        if (to != null && to.Value.x < 0 && to.Value.y < 0)
        {
            // player has indicated to not make any more moves
            return null;  // 玩家指示不再进行移动
        }
    }
    while (true);
}

/// <summary>
/// The logic behind the Player's turn
/// Get the player input for a move
/// Get subsequent jumps, if possible
/// </summary>
int [,] PlayerTurn(int[,] state)
{
    var move = GetPlayerMove(state);  // 获取玩家的移动
    do
    {
        state = ApplyMove(state, move.from, move.to);  // 应用玩家的移动
        if (!IsJumpMove(move.from, move.to))
        {
            // If player doesn't make a jump move, no further moves are possible
            break;  // 如果玩家没有进行跳跃移动，则无法进行进一步移动
        }
        var nextMove = GetPlayerSubsequentJump(state, move.to);  // 获取玩家的后续跳跃
        if (nextMove == null)
        {
            // another jump is not made
            break;  // 没有进行另一个跳跃
        }
        move = nextMove.Value;  // 更新移动坐标
    }
    while (true);
    // check to see if any kings need crowning
    state = CrownKingPieces(state);  // 检查是否有需要加冕的国王
    return state;  // 返回状态
}
#endregion

/*****************************************************************************
 *
 * Main program starts here
 *
 ****************************************************************************/

WriteIntroduction();  // 输出游戏介绍

// initalize state -  empty spots initialize to 0
// set player pieces to 1, computer pieces to -1
// 创建一个8x8的二维数组表示棋盘状态，1代表玩家的棋子，-1代表电脑的棋子，0代表空位
int[,] state = new int[8, 8] {
    { 1, 0, 1, 0, 0, 0,-1, 0 },
    { 0, 1, 0, 0, 0,-1, 0,-1 },
    { 1, 0, 1, 0, 0, 0,-1, 0 },
    { 0, 1, 0, 0, 0,-1, 0,-1 },
    { 1, 0, 1, 0, 0, 0,-1, 0 },
    { 0, 1, 0, 0, 0,-1, 0,-1 },
    { 1, 0, 1, 0, 0, 0,-1, 0 },
    { 0, 1, 0, 0, 0,-1, 0,-1 },
};

// 无限循环，直到游戏结束
while (true)
{
    // 标记是否有棋子移动
    bool moveMade;
    // 电脑执行一步棋，并更新棋盘状态
    (moveMade, state) = ComputerTurn(state);
    // 如果电脑无法移动，则玩家获胜
    if (!moveMade)
    {
        // 在原始程序中，如果电脑无法移动则电脑获胜
        // 这里认为如果电脑无法移动，且玩家可以移动，则玩家获胜
        // 如果两方都无法移动，则游戏为平局
        // 暂时保留原始逻辑
        ComputerWins();
        break;
    }
    // 打印当前棋盘状态
    PrintBoard(state);
    // 检查电脑是否获胜
    if (CheckForComputerWin(state))
    {
        ComputerWins();
        break;
    }
    // 玩家执行一步棋，并更新棋盘状态
    state = PlayerTurn(state);
    // 检查玩家是否获胜
    if (CheckForPlayerWin(state))
    {
        PlayerWins();
        break;
    }
}
```
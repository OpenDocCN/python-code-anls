# `23_Checkers\csharp\Program.cs`

```
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 从文件名读取二进制数据，并封装成字节流
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流内容创建 ZIP 对象，以只读模式打开
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 文件中的文件名列表，读取文件数据，组成文件名到数据的字典
    # 关闭 ZIP 对象
    zip.close()  # 关闭 ZIP 对象
    # 返回结果字典
    return fdict  # 返回文件名到数据的字典
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
    SkipLines(3); // 调用跳过指定行数的输出函数
    for (int y = 7; y >= 0; y--) // 遍历棋盘的行
    {
        for (int x = 0; x < 8; x++) // 遍历棋盘的列
        {
# 根据状态矩阵中的值进行不同的输出
switch(state[x,y]):
    # 如果值为-2，输出"X*"
    case -2:
        Console.Write("X*")
        break
    # 如果值为-1，输出"X "
    case -1:
        Console.Write("X ")
        break
    # 如果值为0，输出". "
    case 0:
        Console.Write(". ")
        break
    # 如果值为1，输出"O "
    case 1:
        Console.Write("O ")
        break
    # 如果值为2，输出"O*"
    case 2:
        Console.Write("O*")
        break
# 输出空格
Console.Write("   ")
        Console.WriteLine();
    }
}

void WriteCenter(string text)
{
    // 定义变量 LineLength 为 80
    const int LineLength = 80;
    // 计算需要添加的空格数，使得文本居中显示
    var spaces = (LineLength - text.Length) / 2;
    // 在控制台输出居中显示的文本
    Console.WriteLine($"{"".PadLeft(spaces)}{text}");
}

void ComputerWins()
{
    // 在控制台输出 "I WIN."
    Console.WriteLine("I WIN.");
}
void PlayerWins()
{
    // 在控制台输出 "YOU WIN."
    Console.WriteLine("YOU WIN.");
}
# 定义一个函数，用于写入游戏介绍
def WriteIntroduction():
    # 在屏幕中央写入标题"CHECKERS"
    WriteCenter("CHECKERS")
    # 在屏幕中央写入"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
    WriteCenter("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    # 跳过3行
    SkipLines(3)
    # 在控制台打印游戏介绍信息
    Console.WriteLine("THIS IS THE GAME OF CHECKERS. THE COMPUTER IS X,")
    Console.WriteLine("AND YOU ARE O.  THE COMPUTER WILL MOVE FIRST.")
    Console.WriteLine("SQUARES ARE REFERRED TO BY A COORDINATE SYSTEM.")
    Console.WriteLine("(0,0) IS THE LOWER LEFT CORNER")
    Console.WriteLine("(0,7) IS THE UPPER LEFT CORNER")
    Console.WriteLine("(7,0) IS THE LOWER RIGHT CORNER")
    Console.WriteLine("(7,7) IS THE UPPER RIGHT CORNER")
    Console.WriteLine("THE COMPUTER WILL TYPE '+TO' WHEN YOU HAVE ANOTHER")
    Console.WriteLine("JUMP.  TYPE TWO NEGATIVE NUMBERS IF YOU CANNOT JUMP.")
    # 跳过3行
    SkipLines(3)

# 定义一个函数，用于验证坐标点是否超出边界
def IsPointOutOfBounds(int x):
{
    # 检查 x 是否小于 0 或大于 7，返回布尔值
    return x < 0 || x > 7;
}

bool IsOutOfBounds((int x, int y) position)
{
    # 检查位置的 x 坐标和 y 坐标是否越界，返回布尔值
    return IsPointOutOfBounds(position.x) || IsPointOutOfBounds(position.y);
}

bool IsJumpMove((int x, int y) from, (int x, int y) to)
{
    # 检查是否是跳跃移动，返回布尔值
    return Math.Abs(from.y - to.y) == 2;
}

bool IsValidPlayerMove(int[,] state, (int x, int y) from, (int x, int y) to)
{
    # 如果目标位置上已经有棋子，则移动无效，返回 false
    if (state[to.x, to.y] != 0)
    {
        return false;
    }
    # 计算横向和纵向的移动距离
    var deltaX = Math.Abs(to.x - from.x);
    var deltaY = Math.Abs(to.y - from.y);
    # 如果横向移动距离不是1或2，则返回false
    if (deltaX != 1 && deltaX != 2)
    {
        return false;
    }
    # 如果横向和纵向移动距离不相等，则返回false
    if (deltaX != deltaY)
    {
        return false;
    }
    # 如果起始位置是1（代表黑色棋子）并且纵向移动方向不是向下，则返回false
    if (state[from.x, from.y] == 1 && Math.Sign(to.y - from.y) <= 0)
    {
        # 只有国王可以向下移动
        return false;
    }
    # 如果横向移动距离为2，则检查是否有跳跃的棋子
    if (deltaX == 2)
    {
        # 获取被跳跃的棋子的位置
        var jump = GetJumpedPiece(from, to);
        # 如果被跳跃的位置上没有棋子，则返回false
        if (state[jump.x, jump.y] >= 0)
        {
// 检查是否存在玩家获胜的情况
bool CheckForComputerWin(int[,] state)
{
    // 初始化玩家是否存活的标志为假
    bool playerAlive = false;
    // 遍历状态数组中的每个棋子
    foreach (var piece in state)
    {
        // 如果棋子的值大于0，表示玩家还存活
        if (piece > 0)
        {
            // 更新玩家存活标志为真
            playerAlive = true;
            // 跳出循环
            break;
        }
    }
    // 返回玩家是否存活的结果
    return !playerAlive;
}
bool CheckForPlayerWin(int[,] state)
{
    // 检查玩家是否获胜的函数
    bool computerAlive = false;
    // 遍历棋盘状态数组
    foreach (var piece in state)
    {
        // 如果棋盘上存在计算机的棋子
        if (piece < 0)
        {
            // 将计算机存活状态设为真
            computerAlive = true;
            // 跳出循环
            break;
        }
    }
    // 返回玩家是否获胜的结果
    return !computerAlive;
}
#endregion

#region Board "arithmetic"
/// <summary>
/// Get the Coordinates of a jumped piece
/// </summary>
(int x, int y) GetJumpedPiece((int x, int y) from, (int x, int y) to)
{
    // 计算跳跃的棋子的位置
    var midX = (to.x + from.x) / 2;
    var midY = (to.y + from.y) / 2;
    return (midX, midY);
}
/// <summary>
/// 应用方向向量 "direction" 到位置 "from"
/// 返回结果位置
/// direction 包含: (-1,-1), (-1, 1), ( 1,-1), ( 1, 1)
/// </summary>
(int x, int y) GetLocation((int x , int y) from, (int x, int y) direction)
{
    return (x: from.x + direction.x, y: from.y + direction.y);
}
#endregion

#region State change functions
/// <summary>
/// 通过从 "from" 移动棋子到 "to" 来改变当前的 "state"
/// This method does not verify that the move being made is valid
/// This method works for both player moves and computer moves
/// </summary>
// 应用移动，更新棋盘状态
int[,] ApplyMove(int[,] state, (int x, int y) from, (int x, int y) to)
{
    // 将起始位置的棋子移动到目标位置
    state[to.x, to.y] = state[from.x, from.y];
    // 将起始位置的棋子状态置为0，表示空位置
    state[from.x, from.y] = 0;

    // 如果是跳跃移动
    if (IsJumpMove(from, to))
    {
        // 从棋盘上移除被跳过的棋子
        var jump = GetJumpedPiece(from, to);
        state[jump.x, jump.y] = 0;
    }
    // 返回更新后的棋盘状态
    return state;
}
/// <summary>
/// At the end of a turn (either player or computer) check to see if any pieces
/// reached the final row.  If so, change them to kings (crown)
/// </summary>
// 定义一个名为CrownKingPieces的函数，接受一个二维整数数组state作为参数，返回一个二维整数数组
int[,] CrownKingPieces(int[,] state)
{
    // 使用for循环遍历x的取值范围为0到7
    for (int x = 0; x < 8; x++)
    {
        // 如果state数组中第x行第0列的元素值为-1，将其修改为-2
        if (state[x, 0] == -1)
        {
            state[x, 0] = -2;
        }
        // 如果state数组中第x行第7列的元素值为1，将其修改为2
        if (state[x, 7] == 1)
        {
            state[x, 7] = 2;
        }
    }
    // 返回修改后的state数组
    return state;
}
#endregion
# Given a current location "from", determine if a move exists in a given vector, "direction"
# direction will contain: (-1,-1), (-1, 1), ( 1,-1), ( 1, 1)
# return "null" if no move is possible in this direction
def GetCandidateMove(state, from, direction):
    # Calculate the new location based on the current location and direction
    to = (from[0] + direction[0], from[1] + direction[1])
    # Check if the new location is out of bounds
    if IsOutOfBounds(to):
        return None
    # Check if the new location is occupied by another piece
    if state[to[0]][to[1]] > 0:
        # potential jump
        # Calculate the new location based on the potential jump
        to = (to[0] + direction[0], to[1] + direction[1])
        # Check if the new location after potential jump is out of bounds
        if IsOutOfBounds(to):
            return None
    # Check if the new location is already occupied by another piece
    if state[to[0]][to[1]] != 0:
        rank += 3;  // 增加跳跃移动的排名值
    }
    else
    {
        rank += 1;  // 增加普通移动的排名值
    }

    return rank;  // 返回移动的排名值
}
        rank += 5;  // 增加5个单位的分数
    }
    if (from.y == 7)
    {
        rank -= 2;  // 如果起始位置在棋盘的最上方，减少2个单位的分数
    }
    if (to.x == 0 || to.x == 7)
    {
        rank += 1;  // 如果目标位置在棋盘的边缘，增加1个单位的分数
    }
    // 在潜在目的地的前一行查找
    for (int c = -1; c <=1; c+=2)
    {
        var inFront = GetLocation(to, (c, -1));  // 获取潜在目的地前一行的位置
        if (IsOutOfBounds(inFront))  // 如果位置超出了棋盘范围，继续下一次循环
            continue;
        if (state[inFront.x, inFront.y] < 0)  // 如果潜在目的地前一行的位置上有敌方棋子
        {
            // protected by our piece in front
            // 如果前方有我方棋子保护，则增加rank值
            rank++;
            // 继续循环
            continue;
        }
        // 获取后方位置
        var inBack = GetLocation(to, (-c, 1));
        // 如果后方位置超出边界
        if (IsOutOfBounds(inBack))
        {
            // 继续循环
            continue;
        }
        // 如果前方有对方棋子，并且后方位置为空或者为起始位置
        if ((state[inFront.x, inFront.y] > 0) &&
            (state[inBack.x, inBack.y] == 0) || (inBack == from))
        {
            // 对方可以跳过我们
            // 减少rank值
            rank -= 2;
        }
    }
    // 返回rank值
    return rank;
};

/// <summary>
# 返回可能移动的位置的枚举，由给定位置的棋子“from”决定
# 如果没有可移动的位置，枚举将为空
def GetPossibleMoves(state, from):
    maxB = 0
    # 根据棋子类型确定可移动的最大步数
    if state[from[0], from[1]] == -2:
        # 国王可以向后移动
        maxB = 1
    elif state[from[0], from[1]] == -1:
        maxB = -1
    else:
        # 不是我们的棋子
        return
    for (int a = -1; a <= 1; a += 2)
    {
        // 循环变量 a，取值为 -1 和 1，分别代表向左和向右移动
        for (int b = -1; b <= maxB; b += 2)
        {
            // 循环变量 b，取值为 -1 和 maxB，分别代表向前和向后移动（只有国王可以向后移动）
            var to = GetCandidateMove(state, from, (a, b));
            // 调用 GetCandidateMove 函数，获取可能的移动位置
            if (to == null)
            {
                // 如果移动位置为空
                // 表示在该方向上没有有效的移动
                continue;
            }
            // 如果移动位置不为空
            // 返回该位置的值
            yield return to.Value;
        }
    }
}
/// <summary>
/// 从候选移动列表“possibleMoves”中确定最佳移动
/// 如果无法进行移动，则返回“null”
/// </summary>
((int x, int y) from, (int x, int y) to)? GetBestMove(int[,] state, IEnumerable<((int x, int y) from, (int x, int y) to)> possibleMoves)
{
    int? bestRank = null; // 初始化最佳等级为null
    ((int x, int y) from, (int x, int y) to)? bestMove = null; // 初始化最佳移动为null

    foreach (var move in possibleMoves) // 遍历可能的移动列表
    {
        int rank = RankMove(state, move.from, move.to); // 计算移动的等级

        if (bestRank == null || rank > bestRank) // 如果当前等级比最佳等级高
        {
            bestRank = rank; // 更新最佳等级
            bestMove = move; // 更新最佳移动
        }
    }
    return bestMove;  // 返回变量 bestMove 的值

}

/// <summary>  // 摘要注释，解释函数的作用
/// Examine the entire board and record all possible moves  // 检查整个棋盘并记录所有可能的移动
/// Return the best move found, if one exists  // 如果存在最佳移动，则返回找到的最佳移动
/// Returns "null" if no move found  // 如果没有找到移动，则返回 "null"
/// </summary>  // 摘要注释结束
((int x, int y) from, (int x, int y) to)? CalculateMove(int[,] state)  // 定义函数 CalculateMove，接受一个二维数组 state 作为参数，并返回一个元组类型的值
{
    var possibleMoves = new List<((int x, int y) from, (int x, int y) to)>();  // 创建一个空的可能移动列表
    for (int x = 0; x < 8; x++)  // 循环遍历 x 坐标
    {
        for (int y = 0; y < 8; y++)  // 在 x 坐标循环的基础上，循环遍历 y 坐标
        {
            var from = (x, y);  // 创建一个元组类型的变量 from，表示当前位置的坐标
            foreach (var to in GetPossibleMoves(state, from))  // 遍历从当前位置 from 可能的移动目标
            {
                possibleMoves.Add((from, to));  // 将当前位置 from 和移动目标 to 组成的元组添加到可能移动列表中
            }
    }
    // 获取最佳移动
    var bestMove = GetBestMove(state, possibleMoves);
    // 返回最佳移动
    return bestMove;
}

/// <summary>
/// 计算机回合的逻辑
/// 寻找有效的移动和可能的后续移动
/// </summary>
(bool moveMade, int[,] state) ComputerTurn(int[,] state)
{
    // 获取可用的最佳移动
    var move = CalculateMove(state);
    if (move == null)
    {
        // 无法进行移动
        return (false, state);
    }
    var from = move.Value.from;
    Console.Write($"FROM {from.x} {from.y} "); // 输出当前棋子的起始位置坐标
    // 继续进行移动，直到没有更多有效的移动可以进行
    while (move != null)
    {
        var to = move.Value.to; // 获取移动的目标位置
        Console.WriteLine($"TO {to.x} {to.y}"); // 输出当前棋子的目标位置坐标
        state = ApplyMove(state, from, to); // 应用移动，更新游戏状态
        if (!IsJumpMove(from, to)) // 如果不是跳跃移动，则跳出循环
            break;

        // 检查是否存在双跳/三跳等情况
        var possibleMoves = new List<((int x, int y) from, (int x, int y) to)>();
        from = to; // 更新起始位置为当前移动的目标位置
        foreach (var candidate in GetPossibleMoves(state, from)) // 遍历可能的移动
        {
            if (IsJumpMove(from, candidate)) // 如果是跳跃移动
            {
                possibleMoves.Add((from, candidate)); // 将起始位置和目标位置添加到可能的移动列表中
            }
        }
        // Get best jump move
        move = GetBestMove(state, possibleMoves); // 调用 GetBestMove 函数，获取最佳跳跃移动

    }
    // apply crowns to any new Kings
    state = CrownKingPieces(state); // 调用 CrownKingPieces 函数，将任何新的国王棋子加冠
    return (true, state); // 返回一个元组，包含布尔值 true 和状态对象

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
    Console.Write(prompt + "? "); // 在控制台中打印提示信息
    var input = Console.ReadLine(); // 从控制台中读取用户输入的内容
    // split the string into multiple parts // 将字符串分割成多个部分
    // 将输入字符串按逗号分割成数组
    var parts = input?.Split(",");
    // 如果数组长度不为2，返回空
    if (parts?.Length != 2)
        // 必须恰好有2部分
        return null;
    // 定义变量x
    int x;
    // 尝试将第一部分转换为整数，如果失败，返回空
    if (!int.TryParse(parts[0], out x))
        // 第一部分不是一个数字
        return null;
    // 定义变量y
    int y;
    // 尝试将第二部分转换为整数，如果失败，返回空
    if (!int.TryParse(parts[1], out y))
        // 第二部分不是一个数字
        return null;

    // 返回包含x和y的元组
    return (x, y);
}

/// <summary>
/// 从玩家获取移动
/// 返回一个表示有效移动的"from"和"to"的元组
/// </summary>
/// </summary>
// 定义一个名为 GetPlayerMove 的函数，接受一个名为 state 的二维数组作为参数，返回一个包含两个元组的元组
((int x, int y) from, (int x,int y) to) GetPlayerMove(int[,] state)
{
    // 原始程序在用户输入方面存在一些问题
    // 1) 原始程序中存在较少的数据合法性检查：
    //    a) FROM 位置必须由玩家拥有
    //    b) TO 位置必须为空
    //    c) FROM 和 TO 的 x 坐标必须相差不超过2个方块
    //    d) FROM 和 TO 的 y 坐标必须与 x 坐标的距离相同
    //    没有检查方向，是否跳跃有效，或者甚至是否棋子移动。
    // 2) 一旦选择了有效的 FROM 位置，必须选择 TO 位置。
    //    如果没有有效的 TO 位置，就会陷入软锁定状态
    // 这种方法有意与原始方法不同
    // 但尽可能保持了原始意图
    // 1) 选择一个 FROM 位置
    // 2)  如果 FROM 无效，则返回步骤 1
    // 3)  选择一个 TO 位置
    // 4)  如果 TO 无效或者暗示的移动无效，
    //     则返回步骤 1
    // There is still currently no way for the player to indicate that no move can be made
    // This matches the original logic, but is a candidate for a refactor

    do
    {
        // 获取玩家输入的起始坐标
        var from = GetCoordinate("FROM");
        // 如果起始坐标不为空且不超出边界且该位置有棋子
        if ((from != null)
            && !IsOutOfBounds(from.Value)
            && (state[from.Value.x, from.Value.y] > 0))
        {
            // we have a valid "from" location
            // 获取玩家输入的目标坐标
            var to = GetCoordinate("TO");
            // 如果目标坐标不为空且不超出边界且是有效的玩家移动
            if ((to != null)
                && !IsOutOfBounds(to.Value)
                && IsValidPlayerMove(state, from.Value, to.Value))
            {
                // we have a valid "to" location
                // 返回起始坐标和目标坐标
                return (from.Value, to.Value);
/// <summary>
/// Get a subsequent jump from the player if they can / want to
/// returns a move ("from", "to") if a player jumps
/// returns null if a player does not make another move
/// The player must input negative numbers for the coordinates to indicate
/// that no more moves are to be made.  This matches the original implementation
/// </summary>
((int x, int y) from, int x, int y) to)? GetPlayerSubsequentJump(int[,] state, (int x, int y) from)
{
    do
    {
        // 从玩家输入获取下一个跳跃的坐标
        var to = GetCoordinate("+TO");
        // 如果玩家输入了坐标，并且坐标不超出边界，并且是有效的玩家移动
        if ((to != null)
            && !IsOutOfBounds(to.Value)
            && IsValidPlayerMove(state, from, to.Value)
            && IsJumpMove(from, to.Value))
        {
            // 如果“to”位置有效
            return (from, to.Value); ;
        }

        if (to != null && to.Value.x < 0 && to.Value.y < 0)
        {
            // 玩家指示不再进行移动
            return null;
        }
    }
    while (true);
}

/// <summary>
/// 玩家回合的逻辑
/// 获取玩家移动的输入
/// 如果可能，获取后续的跳跃
/// </summary>
# 定义一个名为 PlayerTurn 的函数，接受一个二维整数数组 state 作为参数，并返回一个二维整数数组
int [,] PlayerTurn(int[,] state)
{
    # 调用 GetPlayerMove 函数获取玩家的移动
    var move = GetPlayerMove(state);
    # 使用 do...while 循环执行以下操作
    do
    {
        # 将玩家的移动应用到状态中
        state = ApplyMove(state, move.from, move.to);
        # 如果移动不是跳跃移动，则无法进行进一步的移动，跳出循环
        if (!IsJumpMove(move.from, move.to))
        {
            // If player doesn't make a jump move, no further moves are possible
            break;
        }
        # 获取玩家连续跳跃的下一个移动
        var nextMove = GetPlayerSubsequentJump(state, move.to);
        # 如果没有下一个跳跃移动，则跳出循环
        if (nextMove == null)
        {
            // another jump is not made
            break;
        }
        # 更新移动为下一个跳跃移动
        move = nextMove.Value;
    }
    # 循环条件为始终为真，即无限循环
    while (true);
// 初始化状态 - 空位初始化为0
// 将玩家棋子设置为1，将计算机棋子设置为-1
// 把头转向右边来想象棋盘
// 国王将用-2（代表计算机）和2（代表玩家）来表示
int[,] state = new int[8, 8] {
    { 1, 0, 1, 0, 0, 0,-1, 0 },
```
在这段代码中，我们初始化了一个8x8的二维数组state，表示棋盘的状态。其中1代表玩家的棋子，-1代表计算机的棋子，0代表空位。这个数组的初始状态是一个8x8的棋盘，其中有一些位置已经被玩家和计算机的棋子占据，其他位置为空。
    { 0, 1, 0, 0, 0,-1, 0,-1 }, // 初始化一个二维数组
    { 1, 0, 1, 0, 0, 0,-1, 0 }, // 初始化一个二维数组
    { 0, 1, 0, 0, 0,-1, 0,-1 }, // 初始化一个二维数组
    { 1, 0, 1, 0, 0, 0,-1, 0 }, // 初始化一个二维数组
    { 0, 1, 0, 0, 0,-1, 0,-1 }, // 初始化一个二维数组
    { 1, 0, 1, 0, 0, 0,-1, 0 }, // 初始化一个二维数组
    { 0, 1, 0, 0, 0,-1, 0,-1 }, // 初始化一个二维数组
};

while (true)
{
    bool moveMade; // 声明一个布尔变量
    (moveMade, state) = ComputerTurn(state); // 调用ComputerTurn函数，将返回的结果赋值给moveMade和state
    if (!moveMade) // 如果moveMade为假
    {
        // In the original program the computer wins if it cannot make a move
        // I believe the player should win in this case, assuming the player can make a move.
        // if neither player can make a move, the game should be draw.
        // I have left it as the original logic for now.
        ComputerWins(); // 调用ComputerWins函数
        break;  # 结束当前循环，跳出循环体
    }
    PrintBoard(state);  # 调用PrintBoard函数，打印游戏棋盘的当前状态
    if (CheckForComputerWin(state))  # 调用CheckForComputerWin函数，检查计算机是否获胜
    {
        ComputerWins();  # 调用ComputerWins函数，宣布计算机获胜
        break;  # 结束当前循环，跳出循环体
    }
    state = PlayerTurn(state);  # 调用PlayerTurn函数，轮到玩家进行下一步操作
    if (CheckForPlayerWin(state))  # 调用CheckForPlayerWin函数，检查玩家是否获胜
    {
        PlayerWins();  # 调用PlayerWins函数，宣布玩家获胜
        break;  # 结束当前循环，跳出循环体
    }
}
```
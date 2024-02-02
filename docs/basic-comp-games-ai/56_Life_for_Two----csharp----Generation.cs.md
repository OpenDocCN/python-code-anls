# `basic-computer-games\56_Life_for_Two\csharp\Generation.cs`

```py
internal class Generation
{
    private readonly Board _board;  // 声明私有变量_board，用于存储游戏板信息

    public Generation(Board board)  // 构造函数，接受一个Board对象作为参数
    {
        _board = board;  // 将传入的Board对象赋值给私有变量_board
        CountNeighbours();  // 调用CountNeighbours方法
    }

    public Board Board => _board;  // 公共属性，返回私有变量_board

    public int Player1Count => _board.Player1Count;  // 公共属性，返回玩家1的棋子数量
    public int Player2Count => _board.Player2Count;  // 公共属性，返回玩家2的棋子数量

    public string? Result =>  // 公共属性，返回游戏结果
        (Player1Count, Player2Count) switch  // 使用switch语句根据玩家1和玩家2的棋子数量进行判断
        {
            (0, 0) => Strings.Draw,  // 如果玩家1和玩家2的棋子数量都为0，则返回平局
            (_, 0) => string.Format(Formats.Winner, 1),  // 如果玩家2的棋子数量为0，则返回玩家1获胜
            (0, _) => string.Format(Formats.Winner, 2),  // 如果玩家1的棋子数量为0，则返回玩家2获胜
            _ => null  // 其他情况返回null
        };

    public static Generation Create(IReadWrite io)  // 静态方法，用于创建Generation对象，接受一个IReadWrite对象作为参数
    {
        var board = new Board();  // 创建一个新的Board对象

        SetInitialPieces(1, coord => board.AddPlayer1Piece(coord));  // 调用SetInitialPieces方法，设置玩家1的初始棋子
        SetInitialPieces(2, coord => board.AddPlayer2Piece(coord));  // 调用SetInitialPieces方法，设置玩家2的初始棋子

        return new Generation(board);  // 返回一个新的Generation对象，传入新创建的Board对象

        void SetInitialPieces(int player, Action<Coordinates> setPiece)  // 声明一个私有方法SetInitialPieces，接受一个玩家编号和一个Action<Coordinates>类型的委托作为参数
        {
            io.WriteLine(Formats.InitialPieces, player);  // 调用IReadWrite对象的WriteLine方法，输出初始棋子信息
            for (var i = 1; i <= 3; i++)  // 循环3次
            {
                setPiece(io.ReadCoordinates(board));  // 调用传入的委托，设置棋子的坐标
            }
        }
    }

    public Generation CalculateNextGeneration()  // 公共方法，用于计算下一代的Generation对象
    {
        var board = new Board();  // 创建一个新的Board对象

        foreach (var coordinates in _board)  // 遍历_board中的坐标集合
        {
            board[coordinates] = _board[coordinates].GetNext();  // 根据当前坐标的状态计算下一代的状态，并赋值给新的Board对象
        }

        return new(board);  // 返回一个新的Generation对象，传入新创建的Board对象
    }
    
    public void AddPieces(IReadWrite io)  // 公共方法，用于向游戏板中添加棋子，接受一个IReadWrite对象作为参数
    {
        var player1Coordinate = io.ReadCoordinates(1, _board);  // 从IReadWrite对象中读取玩家1的棋子坐标
        var player2Coordinate = io.ReadCoordinates(2, _board);  // 从IReadWrite对象中读取玩家2的棋子坐标

        if (player1Coordinate == player2Coordinate)  // 如果玩家1和玩家2的棋子坐标相同
        {
            io.Write(Streams.SameCoords);  // 输出相同坐标的提示信息
            // This is a bug existing in the original code. The line should be _board[_coordinates[_player]] = 0;
            _board.ClearCell(player1Coordinate + 1);  // 清空相同坐标的单元格
        }
        else  // 如果玩家1和玩家2的棋子坐标不同
        {
            _board.AddPlayer1Piece(player1Coordinate);  // 向游戏板中添加玩家1的棋子
            _board.AddPlayer2Piece(player2Coordinate);  // 向游戏板中添加玩家2的棋子
        }
    }
}
    # 计算每个格子周围的邻居数量
    private void CountNeighbours()
    {
        # 遍历棋盘上的每个坐标
        foreach (var coordinates in _board)
        {
            # 获取当前坐标上的棋子
            var piece = _board[coordinates];
            # 如果当前坐标上没有棋子，则跳过
            if (piece.IsEmpty) { continue; }

            # 遍历当前坐标的邻居坐标
            foreach (var neighbour in coordinates.GetNeighbors())
            {
                # 将当前邻居坐标上的棋子的邻居数量加一
                _board[neighbour] = _board[neighbour].AddNeighbour(piece);
            }
        }
    }

    # 重写 ToString 方法，返回棋盘的字符串表示
    public override string ToString() => _board.ToString();
# 闭合大括号，表示代码块的结束
```
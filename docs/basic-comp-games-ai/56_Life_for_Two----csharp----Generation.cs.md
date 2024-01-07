# `basic-computer-games\56_Life_for_Two\csharp\Generation.cs`

```

// 定义 Generation 类
internal class Generation
{
    // 私有成员变量 _board，用于存储棋盘信息
    private readonly Board _board;

    // Generation 类的构造函数，接受一个 Board 对象作为参数
    public Generation(Board board)
    {
        // 将传入的 Board 对象赋值给 _board
        _board = board;
        // 调用 CountNeighbours 方法，计算每个位置的邻居数量
        CountNeighbours();
    }

    // 只读属性，返回 _board 对象
    public Board Board => _board;

    // 只读属性，返回 _board 中 Player1 的棋子数量
    public int Player1Count => _board.Player1Count;
    // 只读属性，返回 _board 中 Player2 的棋子数量
    public int Player2Count => _board.Player2Count;

    // 只读属性，根据 Player1Count 和 Player2Count 返回游戏结果
    public string? Result => 
        (Player1Count, Player2Count) switch
        {
            (0, 0) => Strings.Draw,
            (_, 0) => string.Format(Formats.Winner, 1),
            (0, _) => string.Format(Formats.Winner, 2),
            _ => null
        };

    // 静态方法，用于创建 Generation 对象
    public static Generation Create(IReadWrite io)
    {
        // 创建一个新的 Board 对象
        var board = new Board();

        // 设置初始棋子位置，玩家1和玩家2各设置3个棋子
        SetInitialPieces(1, coord => board.AddPlayer1Piece(coord));
        SetInitialPieces(2, coord => board.AddPlayer2Piece(coord));

        // 返回一个新的 Generation 对象
        return new Generation(board);

        // 设置初始棋子位置的内部方法
        void SetInitialPieces(int player, Action<Coordinates> setPiece)
        {
            // 输出初始棋子位置的信息
            io.WriteLine(Formats.InitialPieces, player);
            // 循环3次，设置初始棋子位置
            for (var i = 1; i <= 3; i++)
            {
                setPiece(io.ReadCoordinates(board));
            }
        }
    }

    // 计算下一代的棋盘状态
    public Generation CalculateNextGeneration()
    {
        // 创建一个新的 Board 对象
        var board = new Board();

        // 遍历当前棋盘的每个位置，计算下一代的状态
        foreach (var coordinates in _board)
        {
            board[coordinates] = _board[coordinates].GetNext();
        }

        // 返回一个新的 Generation 对象
        return new(board);
    }
    
    // 添加棋子的方法
    public void AddPieces(IReadWrite io)
    {
        // 从输入中读取玩家1和玩家2的棋子位置
        var player1Coordinate = io.ReadCoordinates(1, _board);
        var player2Coordinate = io.ReadCoordinates(2, _board);

        // 如果两个玩家的棋子位置相同
        if (player1Coordinate == player2Coordinate)
        {
            // 输出错误信息
            io.Write(Streams.SameCoords);
            // 清空该位置的棋子
            _board.ClearCell(player1Coordinate + 1);
        }
        else
        {
            // 在棋盘上添加玩家1和玩家2的棋子
            _board.AddPlayer1Piece(player1Coordinate);
            _board.AddPlayer2Piece(player2Coordinate);
        }
    }

    // 计算每个位置的邻居数量
    private void CountNeighbours()
    {
        // 遍历棋盘上的每个位置
        foreach (var coordinates in _board)
        {
            var piece = _board[coordinates];
            // 如果当前位置为空，则跳过
            if (piece.IsEmpty) { continue; }

            // 遍历当前位置的邻居位置
            foreach (var neighbour in coordinates.GetNeighbors())
            {
                // 更新邻居位置的邻居数量
                _board[neighbour] = _board[neighbour].AddNeighbour(piece);
            }
        }
    }

    // 重写 ToString 方法，返回棋盘的字符串表示
    public override string ToString() => _board.ToString();
}

```
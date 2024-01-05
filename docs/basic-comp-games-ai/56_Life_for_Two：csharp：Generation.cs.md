# `d:/src/tocomm/basic-computer-games\56_Life_for_Two\csharp\Generation.cs`

```
internal class Generation
{
    private readonly Board _board; // 声明一个私有的 Board 类型的变量 _board

    public Generation(Board board) // Generation 类的构造函数，接受一个 Board 类型的参数 board
    {
        _board = board; // 将传入的 board 参数赋值给 _board
        CountNeighbours(); // 调用 CountNeighbours 方法
    }

    public Board Board => _board; // 声明一个公共的只读属性 Board，返回 _board

    public int Player1Count => _board.Player1Count; // 声明一个公共的只读属性 Player1Count，返回 _board 的 Player1Count 属性
    public int Player2Count => _board.Player2Count; // 声明一个公共的只读属性 Player2Count，返回 _board 的 Player2Count 属性

    public string? Result =>  // 声明一个公共的可空的字符串属性 Result
        (Player1Count, Player2Count) switch // 使用 Player1Count 和 Player2Count 进行模式匹配
        {
            (0, 0) => Strings.Draw, // 当 Player1Count 和 Player2Count 都为 0 时，返回 Strings.Draw
            (_, 0) => string.Format(Formats.Winner, 1), // 当 Player2Count 为 0 时，返回格式化后的字符串，其中 {0} 会被替换为 1
            (0, _) => string.Format(Formats.Winner, 2),  // 如果第一个参数为0，则返回格式化后的字符串，表示玩家2获胜
            _ => null  // 否则返回空值
        };

    public static Generation Create(IReadWrite io)  // 创建一个名为Create的静态方法，接受一个IReadWrite类型的参数，并返回一个Generation类型的对象
    {
        var board = new Board();  // 创建一个新的Board对象

        SetInitialPieces(1, coord => board.AddPlayer1Piece(coord));  // 调用SetInitialPieces方法，传入1和一个lambda表达式，将玩家1的棋子添加到棋盘上
        SetInitialPieces(2, coord => board.AddPlayer2Piece(coord));  // 调用SetInitialPieces方法，传入2和一个lambda表达式，将玩家2的棋子添加到棋盘上

        return new Generation(board);  // 返回一个新的Generation对象，传入之前创建的board对象

        void SetInitialPieces(int player, Action<Coordinates> setPiece)  // 创建一个名为SetInitialPieces的方法，接受一个整数类型的player参数和一个Action类型的setPiece参数
        {
            io.WriteLine(Formats.InitialPieces, player);  // 调用io对象的WriteLine方法，传入Formats.InitialPieces和player参数
            for (var i = 1; i <= 3; i++)  // 循环3次
            {
                setPiece(io.ReadCoordinates(board));  // 调用setPiece方法，传入io对象的ReadCoordinates方法的返回值，将棋子添加到棋盘上
            }
        }
    }  # 结束 CalculateNextGeneration 方法的定义

    public Generation CalculateNextGeneration()  # 定义 CalculateNextGeneration 方法
    {
        var board = new Board();  # 创建一个新的 Board 对象

        foreach (var coordinates in _board)  # 遍历 _board 中的坐标
        {
            board[coordinates] = _board[coordinates].GetNext();  # 将 _board 中每个坐标的下一个状态存储到 board 中
        }

        return new Generation(board);  # 返回一个新的 Generation 对象，传入 board 作为参数
    }
    
    public void AddPieces(IReadWrite io)  # 定义 AddPieces 方法，接受一个 IReadWrite 类型的参数 io
    {
        var player1Coordinate = io.ReadCoordinates(1, _board);  # 通过 io 读取玩家1的坐标信息
        var player2Coordinate = io.ReadCoordinates(2, _board);  # 通过 io 读取玩家2的坐标信息
        if (player1Coordinate == player2Coordinate)
        {
            // 如果玩家1和玩家2的坐标相同，向输出流写入相同坐标的消息
            io.Write(Streams.SameCoords);
            // 这是原始代码中存在的一个错误。这一行应该是 _board[_coordinates[_player]] = 0;
            // 清除玩家1坐标+1处的单元格
            _board.ClearCell(player1Coordinate + 1);
        }
        else
        {
            // 否则，向棋盘添加玩家1的棋子
            _board.AddPlayer1Piece(player1Coordinate);
            // 向棋盘添加玩家2的棋子
            _board.AddPlayer2Piece(player2Coordinate);
        }
    }

    private void CountNeighbours()
    {
        // 遍历棋盘上的每个坐标
        foreach (var coordinates in _board)
        {
            // 获取当前坐标处的棋子
            var piece = _board[coordinates];
            // 如果该位置为空，继续下一次循环
            if (piece.IsEmpty) { continue; }
            foreach (var neighbour in coordinates.GetNeighbors())
            {
                // 遍历当前坐标的邻居坐标
                // 将当前邻居坐标对应的棋子添加为当前坐标对应棋子的邻居
                _board[neighbour] = _board[neighbour].AddNeighbour(piece);
            }
        }
    }

    // 重写 ToString 方法，返回棋盘的字符串表示
    public override string ToString() => _board.ToString();
}
```
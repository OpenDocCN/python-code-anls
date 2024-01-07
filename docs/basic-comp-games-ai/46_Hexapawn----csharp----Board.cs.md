# `basic-computer-games\46_Hexapawn\csharp\Board.cs`

```

// 引入所需的命名空间
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;

// 使用 Hexapawn.Pawn 命名空间下的 Pawn 类
using static Hexapawn.Pawn;

// 声明 Hexapawn 命名空间下的 Board 类，并实现 IEnumerable<Pawn> 和 IEquatable<Board> 接口
namespace Hexapawn;

// 声明 Board 类，实现 IEnumerable<Pawn> 和 IEquatable<Board> 接口
internal class Board : IEnumerable<Pawn>, IEquatable<Board>
{
    // 声明只读的 Pawn 数组 _cells
    private readonly Pawn[] _cells;

    // 默认构造函数，初始化 _cells 数组
    public Board()
    {
        _cells = new[]
        {
            Black, Black, Black,
            None,  None,  None,
            White, White, White
        };
    }

    // 带参数的构造函数，初始化 _cells 数组
    public Board(params Pawn[] cells)
    {
        _cells = cells;
    }

    // 索引器，用于访问 _cells 数组中的元素
    public Pawn this[int index]
    {
        get => _cells[index - 1];
        set => _cells[index - 1] = value;
    }

    // 返回当前棋盘的镜像
    public Board Reflected => new(Cell.AllCells.Select(c => this[c.Reflected]).ToArray());

    // 实现 IEnumerable 接口的 GetEnumerator 方法
    public IEnumerator<Pawn> GetEnumerator() => _cells.OfType<Pawn>().GetEnumerator();

    // 实现 IEnumerable 接口的 GetEnumerator 方法
    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    // 重写 ToString 方法，返回当前棋盘的字符串表示
    public override string ToString()
    {
        var builder = new StringBuilder().AppendLine();
        for (int row = 0; row < 3; row++)
        {
            builder.Append("          ");
            for (int col = 0; col < 3; col++)
            {
                builder.Append(_cells[row * 3 + col]);
            }
            builder.AppendLine();
        }
        return builder.ToString();
    }

    // 判断当前棋盘是否与另一个棋盘相等
    public bool Equals(Board other) => other?.Zip(this).All(x => x.First == x.Second) ?? false;

    // 重写 Equals 方法，判断当前对象是否与另一个对象相等
    public override bool Equals(object obj) => Equals(obj as Board);

    // 重写 GetHashCode 方法，返回当前对象的哈希值
    public override int GetHashCode()
    {
        var hash = 19;

        for (int i = 0; i < 9; i++)
        {
            hash = hash * 53 + _cells[i].GetHashCode();
        }

        return hash;
    }
}

```
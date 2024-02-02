# `basic-computer-games\46_Hexapawn\csharp\Board.cs`

```py
// 引入所需的命名空间
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
// 使用 Hexapawn.Pawn 命名空间下的 Pawn 类
using static Hexapawn.Pawn;

// 声明 Hexapawn 命名空间下的 Board 类，并实现 IEnumerable 接口和 IEquatable 接口
namespace Hexapawn;

// 声明 Board 类为 internal 类
internal class Board : IEnumerable<Pawn>, IEquatable<Board>
{
    // 声明只读的 _cells 数组，用于存储棋盘上的棋子
    private readonly Pawn[] _cells;

    // 声明 Board 类的默认构造函数
    public Board()
    {
        // 初始化 _cells 数组，设置初始棋盘状态
        _cells = new[]
        {
            Black, Black, Black,
            None,  None,  None,
            White, White, White
        };
    }

    // 声明 Board 类的带参数的构造函数
    public Board(params Pawn[] cells)
    {
        // 将传入的 cells 数组赋值给 _cells 数组
        _cells = cells;
    }

    // 声明 Board 类的索引器，用于访问 _cells 数组中的元素
    public Pawn this[int index]
    {
        // 获取 _cells 数组中指定索引位置的元素
        get => _cells[index - 1];
        // 设置 _cells 数组中指定索引位置的元素
        set => _cells[index - 1] = value;
    }

    // 声明 Board 类的 Reflected 属性，用于获取当前棋盘的镜像
    public Board Reflected => new(Cell.AllCells.Select(c => this[c.Reflected]).ToArray());

    // 实现 IEnumerable 接口的 GetEnumerator 方法，用于遍历 _cells 数组中的元素
    public IEnumerator<Pawn> GetEnumerator() => _cells.OfType<Pawn>().GetEnumerator();

    // 实现 IEnumerable 接口的 GetEnumerator 方法，用于遍历 _cells 数组中的元素
    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    // 重写 ToString 方法，用于返回当前棋盘的字符串表示
    public override string ToString()
    {
        // 创建 StringBuilder 对象
        var builder = new StringBuilder().AppendLine();
        // 遍历 _cells 数组，构建棋盘的字符串表示
        for (int row = 0; row < 3; row++)
        {
            builder.Append("          ");
            for (int col = 0; col < 3; col++)
            {
                builder.Append(_cells[row * 3 + col]);
            }
            builder.AppendLine();
        }
        // 返回棋盘的字符串表示
        return builder.ToString();
    }

    // 实现 IEquatable 接口的 Equals 方法，用于比较两个棋盘是否相等
    public bool Equals(Board other) => other?.Zip(this).All(x => x.First == x.Second) ?? false;

    // 重写 Equals 方法，用于比较当前对象与其他对象是否相等
    public override bool Equals(object obj) => Equals(obj as Board);

    // 重写 GetHashCode 方法，用于获取当前对象的哈希码
    public override int GetHashCode()
    {
        // 初始化哈希码
        var hash = 19;
        // 遍历 _cells 数组，计算哈希码
        for (int i = 0; i < 9; i++)
        {
            hash = hash * 53 + _cells[i].GetHashCode();
        }
        // 返回哈希码
        return hash;
    }
}
```
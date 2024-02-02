# `basic-computer-games\46_Hexapawn\csharp\Cell.cs`

```py
// 引入 System 和 System.Collections.Generic 命名空间
using System;
using System.Collections.Generic;

// 定义 Hexapawn 命名空间
namespace Hexapawn
{
    // 表示棋盘上的一个单元格，编号从 1 到 9，支持查找关于棋盘中间列的镜像
    internal class Cell
    {
        // 静态只读字段，包含所有单元格
        private static readonly Cell[] _cells = new Cell[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        // 静态只读字段，包含所有单元格关于中间列的镜像
        private static readonly Cell[] _reflected = new Cell[] { 3, 2, 1, 6, 5, 4, 9, 8, 7 };
        // 只读字段，存储单元格的编号
        private readonly int _number;
        // 私有构造函数，用于创建单元格对象
        private Cell(int number)
        {
            // 如果编号不在 1 到 9 之间，抛出 ArgumentOutOfRangeException 异常
            if (number < 1 || number > 9)
            {
                throw new ArgumentOutOfRangeException(nameof(number), number, "Must be from 1 to 9");
            }
            _number = number;
        }
        // 用于枚举所有单元格
        public static IEnumerable<Cell> AllCells => _cells;
        // 接受用户输入的值，并尝试创建一个单元格引用
        public static bool TryCreate(float input, out Cell cell)
        {
            // 如果输入是整数且在 1 到 9 之间，创建单元格引用并返回 true
            if (IsInteger(input) && input >= 1 && input <= 9)
            {
                cell = (int)input;
                return true;
            }
            // 否则返回 false
            cell = default;
            return false;
            // 嵌套方法，用于检查值是否为整数
            static bool IsInteger(float value) => value - (int)value == 0;
        }
        // 返回单元格关于棋盘中间列的镜像
        public Cell Reflected => _reflected[_number - 1];
        // 允许将单元格引用用作 int 类型，例如在 Board 的索引器中
        public static implicit operator int(Cell c) => c._number;
        public static implicit operator Cell(int number) => new Cell(number);
        // 返回单元格编号的字符串表示形式
        public override string ToString() => _number.ToString();
    }
}
```
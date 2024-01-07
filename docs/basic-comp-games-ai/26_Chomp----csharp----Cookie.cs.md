# `basic-computer-games\26_Chomp\csharp\Cookie.cs`

```

// 使用 System.Text 命名空间
using System.Text;

// 命名空间 Chomp
namespace Chomp
{
    // Cookie 类
    internal class Cookie
    {
        // 私有字段，行数和列数
        private readonly int _rowCount;
        private readonly int _columnCount;
        // 二维字符数组
        private readonly char[][] _bits;

        // 构造函数，初始化行数和列数，并创建二维字符数组
        public Cookie(int rowCount, int columnCount)
        {
            _rowCount = rowCount;
            _columnCount = columnCount;

            // 使用 Math.Max 处理负值，创建二维字符数组
            _bits = new char[Math.Max(_rowCount, 1)][];
            for (int row = 0; row < _bits.Length; row++)
            {
                // 使用 Enumerable.Repeat 创建指定长度的字符数组
                _bits[row] = Enumerable.Repeat('*', Math.Max(_columnCount, 1)).ToArray();
            }
            // 将第一个元素设置为 'P'
            _bits[0][0] = 'P';
        }

        // 尝试吃掉指定位置的字符
        public bool TryChomp(int row, int column, out char chomped)
        {
            // 判断位置是否有效，如果无效则返回 false
            if (row < 1 || row > _rowCount || column < 1 || column > _columnCount || _bits[row - 1][column - 1] == ' ')
            {
                chomped = default;
                return false;
            }

            // 获取指定位置的字符
            chomped = _bits[row - 1][column - 1];

            // 将指定位置及其右下方的字符设置为空格
            for (int r = row; r <= _rowCount; r++)
            {
                for (int c = column; c <= _columnCount; c++)
                {
                    _bits[r - 1][c - 1] = ' ';
                }
            }

            return true;
        }

        // 重写 ToString 方法
        public override string ToString()
        {
            // 创建 StringBuilder 对象
            var builder = new StringBuilder().AppendLine("       1 2 3 4 5 6 7 8 9");
            for (int row = 1; row <= _bits.Length; row++)
            {
                // 拼接每行的字符数组
                builder.Append(' ').Append(row).Append("     ").AppendLine(string.Join(' ', _bits[row - 1]));
            }
            return builder.ToString();
        }
    }
}

```
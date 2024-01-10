# `basic-computer-games\26_Chomp\csharp\Cookie.cs`

```
// 使用 System.Text 命名空间
using System.Text;

// 命名空间 Chomp
namespace Chomp
{
    // Cookie 类，内部访问权限
    internal class Cookie
    {
        // 只读字段，行数和列数
        private readonly int _rowCount;
        private readonly int _columnCount;
        // 字符数组
        private readonly char[][] _bits;

        // 构造函数，初始化行数和列数
        public Cookie(int rowCount, int columnCount)
        {
            _rowCount = rowCount;
            _columnCount = columnCount;

            // 创建字符数组，处理负值情况
            _bits = new char[Math.Max(_rowCount, 1)][];
            for (int row = 0; row < _bits.Length; row++)
            {
                _bits[row] = Enumerable.Repeat('*', Math.Max(_columnCount, 1)).ToArray();
            }
            _bits[0][0] = 'P';  // 设置第一个元素为 'P'
        }

        // 尝试吃掉饼干的一部分
        public bool TryChomp(int row, int column, out char chomped)
        {
            // 判断是否越界或者已经被吃掉
            if (row < 1 || row > _rowCount || column < 1 || column > _columnCount || _bits[row - 1][column - 1] == ' ')
            {
                chomped = default;
                return false;
            }

            chomped = _bits[row - 1][column - 1];  // 记录被吃掉的部分

            // 将被吃掉的部分标记为空格
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
                // 拼接每一行的内容
                builder.Append(' ').Append(row).Append("     ").AppendLine(string.Join(' ', _bits[row - 1]));
            }
            return builder.ToString();  // 返回拼接好的字符串
        }
    }
}
```
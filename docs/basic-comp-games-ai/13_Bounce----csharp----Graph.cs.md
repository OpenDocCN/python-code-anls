# `13_Bounce\csharp\Graph.cs`

```
using System.Text; // 导入 System.Text 命名空间，提供对字符串操作的支持

namespace Bounce; // 命名空间声明，定义了类的作用域

/// <summary>
/// Provides support for plotting a graph of height vs time, and rendering it to a string.
/// </summary>
internal class Graph // 定义了一个名为 Graph 的内部类
{
    private readonly Dictionary<int, Row> _rows; // 声明了一个名为 _rows 的只读字典，键为整数，值为 Row 类型的对象

    public Graph(float maxHeight, float timeIncrement) // Graph 类的构造函数，接受 maxHeight 和 timeIncrement 两个参数
    {
        // 1 row == 1/2 foot + 1 row for zero
        var rowCount = 2 * (int)Math.Round(maxHeight, MidpointRounding.AwayFromZero) + 1; // 计算 rowCount 的值
        _rows = Enumerable.Range(0, rowCount) // 使用 Enumerable.Range 方法生成一个整数序列
            .ToDictionary(x => x, x => new Row(x % 2 == 0 ? $" {x / 2} " : "")); // 将整数序列转换为字典，键为整数，值为 Row 类型的对象
        TimeIncrement = timeIncrement; // 设置 TimeIncrement 属性的值为传入的 timeIncrement 参数
    }
    public float TimeIncrement { get; }  # 定义一个公共的浮点型属性 TimeIncrement，只读
    public float MaxTimePlotted { get; private set; }  # 定义一个公共的浮点型属性 MaxTimePlotted，可读写但只能在类内部进行写操作

    public void Plot(float time, float height)  # 定义一个公共的方法 Plot，接受两个浮点型参数 time 和 height
    {
        var rowIndex = (int)Math.Round(height * 2, MidpointRounding.AwayFromZero);  # 计算行索引，将 height 乘以 2 并四舍五入转换为整数
        var colIndex = (int)(time / TimeIncrement) + 1;  # 计算列索引，将 time 除以 TimeIncrement 转换为整数并加 1
        if (_rows.TryGetValue(rowIndex, out var row))  # 如果 _rows 中包含 rowIndex 对应的值，则将其赋值给 row
        {
            row[colIndex] = '0';  # 将 row 中的 colIndex 索引位置的值设为 '0'
        }
        MaxTimePlotted = Math.Max(time, MaxTimePlotted);  # 将 MaxTimePlotted 更新为 time 和 MaxTimePlotted 中的较大值
    }

    public override string ToString()  # 重写 ToString 方法
    {
        var sb = new StringBuilder().AppendLine("Feet").AppendLine();  # 创建一个 StringBuilder 对象，并添加 "Feet" 和换行符到其中
        foreach (var (_, row) in _rows.OrderByDescending(x => x.Key))  # 遍历 _rows 中按照键值降序排列的元素
        {
            sb.Append(row).AppendLine();  # 将 row 添加到 StringBuilder 对象中并添加换行符
        }
        sb.Append(new Axis(MaxTimePlotted, TimeIncrement));  // 将一个新的坐标轴对象添加到字符串构建器中

        return sb.ToString();  // 返回字符串构建器中的内容作为字符串
    }

    internal class Row  // 定义一个内部类 Row
    {
        public const int Width = 70;  // 定义一个常量 Width，值为 70

        private readonly char[] _chars = new char[Width + 2];  // 创建一个长度为 Width+2 的字符数组，并用 readonly 关键字标记为只读
        private int nextColumn = 0;  // 定义一个私有变量 nextColumn，初始值为 0

        public Row(string label)  // 定义一个构造函数，参数为 label
        {
            Array.Fill(_chars, ' ');  // 用空格填充字符数组
            Array.Copy(label.ToCharArray(), _chars, label.Length);  // 将 label 转换为字符数组，并复制到 _chars 数组中
            nextColumn = label.Length;  // 将 nextColumn 设置为 label 的长度
        }
        public char this[int column]
        {
            set
            {
                // 检查索引是否超出数组长度，如果是则返回
                if (column >= _chars.Length) { return; }
                // 如果要设置的列小于下一个列的索引，将列索引设置为下一个列的索引
                if (column < nextColumn) { column = nextColumn; }
                // 设置指定列的字符值
                _chars[column] = value;
                // 更新下一个列的索引
                nextColumn = column + 1;
            }
        }

        // 重写 ToString 方法，返回由字符数组构成的字符串
        public override string ToString() => new string(_chars);
    }

    internal class Axis
    {
        // 最大时间标记
        private readonly int _maxTimeMark;
        // 时间增量
        private readonly float _timeIncrement;
        // 标签
        private readonly Labels _labels;
        // 构造函数，初始化坐标轴的最大时间和时间增量
        internal Axis(float maxTimePlotted, float timeIncrement)
        {
            // 将最大时间向上取整并转换为整数赋值给 _maxTimeMark
            _maxTimeMark = (int)Math.Ceiling(maxTimePlotted);
            // 将时间增量赋值给 _timeIncrement
            _timeIncrement = timeIncrement;

            // 初始化标签对象
            _labels = new Labels();
            // 循环生成标签，i 从 1 到 _maxTimeMark
            for (var i = 1; i <= _maxTimeMark; i++)
            {
                // 将 (i / _timeIncrement) 转换为整数作为键，将 $" {i} " 作为值添加到标签对象中
                _labels.Add((int)(i / _timeIncrement), $" {i} ");
            }
        }

        // 重写 ToString 方法
        public override string ToString()
            // 返回一个包含坐标轴标记的字符串
            => new StringBuilder()
                // 添加一行由 '.' 组成的字符串，长度为 (_maxTimeMark / _timeIncrement) + 1
                .Append(' ').Append('.', (int)(_maxTimeMark / _timeIncrement) + 1).AppendLine()
                // 添加标签对象的字符串表示形式
                .Append(_labels).AppendLine()
                // 添加 "Seconds" 字符串，前面有一定数量的空格，数量为 (_maxTimeMark / _timeIncrement / 2 - 2)
                .Append(' ', (int)(_maxTimeMark / _timeIncrement / 2 - 2)).AppendLine("Seconds")
                // 转换为最终的字符串并返回
                .ToString();
    }
    internal class Labels : Row  # 创建一个名为 Labels 的内部类，继承自 Row 类
    {
        public Labels()  # Labels 类的构造函数
            : base(" 0")  # 调用基类 Row 的构造函数，传入参数 " 0"
        {
        }

        public void Add(int column, string label)  # 定义一个名为 Add 的方法，接受两个参数：column 和 label
        {
            for (var i = 0; i < label.Length; i++)  # 使用 for 循环遍历 label 字符串的每个字符
            {
                this[column + i] = label[i];  # 将 label 字符串中的每个字符赋值给 this 对象的指定列
            }
        }
    }
```
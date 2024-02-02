# `basic-computer-games\13_Bounce\csharp\Graph.cs`

```py
{
    /// <summary>
    /// 提供支持绘制高度 vs 时间的图表，并将其渲染为字符串。
    /// </summary>
    internal class Graph
    {
        private readonly Dictionary<int, Row> _rows;

        public Graph(float maxHeight, float timeIncrement)
        {
            // 1 行 == 1/2 英尺 + 1 行用于零点
            var rowCount = 2 * (int)Math.Round(maxHeight, MidpointRounding.AwayFromZero) + 1;
            // 创建行的字典，键为行号，值为行对象
            _rows = Enumerable.Range(0, rowCount)
                .ToDictionary(x => x, x => new Row(x % 2 == 0 ? $" {x / 2} " : ""));
            // 设置时间增量
            TimeIncrement = timeIncrement;
        }

        // 时间增量属性
        public float TimeIncrement { get; }
        // 最大绘制时间属性
        public float MaxTimePlotted { get; private set; }

        // 绘制方法，根据时间和高度绘制图表
        public void Plot(float time, float height)
        {
            // 计算行索引
            var rowIndex = (int)Math.Round(height * 2, MidpointRounding.AwayFromZero);
            // 计算列索引
            var colIndex = (int)(time / TimeIncrement) + 1;
            // 如果行字典中存在对应的行，则在对应位置绘制点
            if (_rows.TryGetValue(rowIndex, out var row))
            {
                row[colIndex] = '0';
            }
            // 更新最大绘制时间
            MaxTimePlotted = Math.Max(time, MaxTimePlotted);
        }

        // 覆盖 ToString 方法，将图表转换为字符串
        public override string ToString()
        {
            // 创建字符串构建器，添加标题
            var sb = new StringBuilder().AppendLine("Feet").AppendLine();
            // 遍历行字典，按行号降序添加行内容
            foreach (var (_, row) in _rows.OrderByDescending(x => x.Key))
            {
                sb.Append(row).AppendLine();
            }
            // 添加时间轴
            sb.Append(new Axis(MaxTimePlotted, TimeIncrement));

            return sb.ToString();
        }

        // 内部行类
        internal class Row
    }
}
    # 定义一个公共常量，表示行的宽度为70
    public const int Width = 70;

    # 定义一个只读的字符数组，长度为Width+2
    private readonly char[] _chars = new char[Width + 2];
    # 初始化下一个列的位置为0
    private int nextColumn = 0;

    # 定义一个行的构造函数，接受一个标签参数
    public Row(string label)
    {
        # 用空格填充字符数组
        Array.Fill(_chars, ' ');
        # 将标签转换为字符数组，并复制到_chars数组中
        Array.Copy(label.ToCharArray(), _chars, label.Length);
        # 更新下一个列的位置为标签的长度
        nextColumn = label.Length;
    }

    # 定义一个索引器，用于设置指定列的字符值
    public char this[int column]
    {
        set
        {
            # 如果列超出数组长度，则返回
            if (column >= _chars.Length) { return; }
            # 如果列小于下一个列的位置，则更新列为下一个列的位置
            if (column < nextColumn) { column = nextColumn; }
            # 设置指定列的字符值为给定值
            _chars[column] = value;
            # 更新下一个列的位置为当前列加1
            nextColumn = column + 1;
        }
    }

    # 重写ToString方法，返回_chars数组转换为字符串的结果
    public override string ToString() => new string(_chars);



    # 定义一个内部类Axis
    internal class Axis
    {
        # 定义只读字段_maxTimeMark、_timeIncrement和_labels
        private readonly int _maxTimeMark;
        private readonly float _timeIncrement;
        private readonly Labels _labels;

        # 定义Axis类的构造函数，接受最大时间和时间增量两个参数
        internal Axis(float maxTimePlotted, float timeIncrement)
        {
            # 将最大时间向上取整并转换为整数，赋值给_maxTimeMark
            _maxTimeMark = (int)Math.Ceiling(maxTimePlotted);
            # 将时间增量赋值给_timeIncrement
            _timeIncrement = timeIncrement;

            # 创建Labels对象，并初始化标签
            _labels = new Labels();
            for (var i = 1; i <= _maxTimeMark; i++)
            {
                # 将时间除以时间增量并转换为整数，作为列号，时间作为标签，添加到_labels中
                _labels.Add((int)(i / _timeIncrement), $" {i} ");
            }
        }

        # 重写ToString方法，返回绘制的坐标轴字符串
        public override string ToString()
            => new StringBuilder()
                .Append(' ').Append('.', (int)(_maxTimeMark / _timeIncrement) + 1).AppendLine()
                .Append(_labels).AppendLine()
                .Append(' ', (int)(_maxTimeMark / _timeIncrement / 2 - 2)).AppendLine("Seconds")
                .ToString();
    }



    # 定义一个继承自Row的内部类Labels
    internal class Labels : Row
    {
        # 定义Labels类的构造函数，调用基类的构造函数，传入" 0"作为标签
        public Labels()
            : base(" 0")
        {
        }

        # 定义一个方法Add，用于向_labels中添加标签
        public void Add(int column, string label)
        {
            # 遍历标签字符串，将每个字符设置到对应的列
            for (var i = 0; i < label.Length; i++)
            {
                this[column + i] = label[i];
            }
        }
    }
# 闭合前面的函数定义
```
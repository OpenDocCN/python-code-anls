# `69_Pizza\csharp\CustomerMap.cs`

```
using System.Text;  // 导入 System.Text 命名空间

namespace Pizza  // 命名空间 Pizza
{
    internal class CustomerMap  // 内部类 CustomerMap
    {
        private readonly int _mapSize;  // 只读整型变量 _mapSize
        private readonly string[,] _customerMap;  // 只读二维字符串数组 _customerMap

        public CustomerMap(int mapSize)  // 构造函数，参数为 mapSize
        {
            _mapSize = mapSize;  // 将参数值赋给 _mapSize
            _customerMap = GenerateCustomerMap();  // 调用 GenerateCustomerMap 方法并将返回值赋给 _customerMap
        }

        /// <summary>
        /// Gets customer on position X, Y.
        /// </summary>
        /// <param name="x">Represents X position.</param>
        /// <param name="y">Represents Y position.</param>
        /// <returns>If positions is valid then returns customer name otherwise returns empty string.</returns>
        public string GetCustomerOnPosition(int x, int y)
        {
            // 检查位置是否超出范围
            if(IsPositionOutOfRange(x, y))
            {
                // 如果位置超出范围，返回空字符串
                return string.Empty;
            }

            // 返回指定位置的客户名称
            return _customerMap[y, x];
        }

        /// <summary>
        /// Overridden ToString for getting text representation of customers map.
        /// </summary>
        /// <returns>Text representation of customers map.</returns>
        public override string ToString()
        {
            // 设置垂直间距和水平间距
            int verticalSpace = 4;
            int horizontalSpace = 5;
            // 创建一个 StringBuilder 对象，用于存储地图的显示内容
            var mapToDisplay = new StringBuilder();

            // 在地图顶部添加一行水平空白
            AppendXLine(mapToDisplay, horizontalSpace);

            // 从地图的最后一行开始向上遍历
            for (int i = _customerMap.GetLength(0) - 1; i >= 0; i--)
            {
                // 在地图的每一行之上添加一条水平分隔线
                mapToDisplay.AppendLine("-", verticalSpace);
                // 添加当前行的行号
                mapToDisplay.Append($"{i + 1}");
                mapToDisplay.Append(' ', horizontalSpace);

                // 遍历当前行的每个元素，并添加到 mapToDisplay 中
                for (var j = 0; j < _customerMap.GetLength(1); j++)
                {
                    mapToDisplay.Append($"{_customerMap[i, j]}");
                    mapToDisplay.Append(' ', horizontalSpace);
                }

                // 添加当前行的行号
                mapToDisplay.Append($"{i + 1}");
                mapToDisplay.Append(' ', horizontalSpace);
                // 添加换行符
                mapToDisplay.Append(Environment.NewLine);
            }
            mapToDisplay.AppendLine("-", verticalSpace);
            // 在地图显示中添加一行分隔符，用于分隔不同行的地图数据

            AppendXLine(mapToDisplay, horizontalSpace);
            // 调用自定义函数AppendXLine，向地图显示中添加一行X字符，用于分隔不同列的地图数据

            return mapToDisplay.ToString();
            // 返回地图显示的字符串结果
        }

        /// <summary>
        /// Checks if position is out of range or not.
        /// </summary>
        /// <param name="x">Represents X position.</param>
        /// <param name="y">Represents Y position.</param>
        /// <returns>True if position is out of range otherwise false.</returns>
        private bool IsPositionOutOfRange(int x, int y)
        {
            return
                x < 0 || x > _mapSize - 1 ||
                y < 0 || y > _mapSize - 1;
            // 检查给定的位置是否超出地图范围，如果超出范围则返回true，否则返回false
        }
/// <summary>
/// 生成表示客户地图的数组。
/// </summary>
/// <returns>返回客户地图。</returns>
private string[,] GenerateCustomerMap()
{
    // 创建一个指定大小的字符串数组，用于表示客户地图
    string[,] customerMap = new string[_mapSize, _mapSize];
    // 获取指定数量的客户名称
    string[] customerNames = GetCustomerNames(_mapSize * _mapSize);
    // 当前客户名称索引
    int currentCustomerNameIndex = 0;

    // 遍历客户地图数组
    for (int i = 0; i < customerMap.GetLength(0); i++)
    {
        for (int j = 0; j < customerMap.GetLength(1); j++)
        {
            // 将客户名称填充到客户地图数组中
            customerMap[i, j] = customerNames[currentCustomerNameIndex++].ToString();
        }
    }

    // 返回客户地图数组
    return customerMap;
}
        }

        /// <summary>
        /// 生成客户名称。名称由字母数字表示，从'A'开始。最后一个客户的名称取决于传入的参数。
        /// </summary>
        /// <param name="numberOfCustomers">需要生成多少个客户。</param>
        /// <returns>客户名称的列表。</returns>
        private static string[] GetCustomerNames(int numberOfCustomers)
        {
            // 返回 ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P"];
            return Enumerable.Range(65, numberOfCustomers).Select(c => ((Char)c).ToString()).ToArray();
        }

        /// <summary>
        /// 添加带有X坐标的行。
        /// </summary>
        /// <param name="mapToDisplay">将添加新行的当前地图。</param>
        /// <param name="horizontalSpace">在每个坐标之间将添加的水平分隔符的数量。</param>
        private void AppendXLine(StringBuilder mapToDisplay, int horizontalSpace)
        {
            mapToDisplay.Append(' ');  // 在地图显示字符串中添加一个空格
            mapToDisplay.Append('-', horizontalSpace);  // 在地图显示字符串中添加横线，横线的数量由 horizontalSpace 决定
            for (var i = 0; i < _customerMap.GetLength(0); i++)  // 循环遍历 _customerMap 的第一维度
            {
                mapToDisplay.Append($"{i + 1}");  // 在地图显示字符串中添加当前行数加一的数字
                mapToDisplay.Append('-', horizontalSpace);  // 在地图显示字符串中添加横线，横线的数量由 horizontalSpace 决定
            }
            mapToDisplay.Append(Environment.NewLine);  // 在地图显示字符串中添加一个换行符
        }
    }
}
```
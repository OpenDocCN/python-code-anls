# `27_Civil_War\csharp\ConsoleUtils.cs`

```
# 引入命名空间
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

# 创建静态类 ConsoleUtils
namespace CivilWar
{
    static class ConsoleUtils
    {
        # 创建公共静态方法 InputYesOrNo
        public static bool InputYesOrNo()
        {
            # 创建无限循环
            while (true)
            {
                # 读取用户输入
                var answer = Console.ReadLine();
                # 使用 switch 语句判断用户输入
                switch (answer?.ToLower())
                {
                    # 如果用户输入为 "no"，返回 false
                    case "no":
                        return false;
                    # 如果用户输入为 "yes"，返回 true
                    case "yes":
                        return true;
                    default:
                        Console.WriteLine("(Answer Yes or No)");  # 输出提示信息，要求用户回答Yes或No
                        break;  # 结束switch语句
                }
            }
        }

        public static void WriteWordWrap(string text)
        {
            var line = new StringBuilder(Console.WindowWidth);  # 创建一个StringBuilder对象，用于存储每行的文本
            foreach (var paragraph in text.Split(Environment.NewLine))  # 遍历文本的每个段落
            {
                line.Clear();  # 清空StringBuilder对象的内容，准备存储新的段落
                foreach (var word in paragraph.Split(' '))  # 遍历段落中的每个单词
                {
                    if (line.Length + word.Length < Console.WindowWidth)  # 判断当前行加上新单词的长度是否超过了控制台的宽度
                    {
                        if (line.Length > 0)  # 如果当前行已经有内容
                            line.Append(' ');  # 在当前行末尾添加一个空格
                        line.Append(word);  # 将新单词添加到当前行
                    }
                    else
                    {
                        // 打印当前行的内容
                        Console.WriteLine(line.ToString());
                        // 清空当前行的内容，并添加新的单词
                        line.Clear().Append(word);
                    }
                }
                // 打印最后一行的内容
                Console.WriteLine(line.ToString());
            }
        }

        // 写入表格数据
        public static void WriteTable<T>(ICollection<T> items, List<TableRow<T>> rows, bool transpose = false)
        {
            // 计算表格的列数
            int cols = items.Count + 1;
            // 格式化表格内容
            var content = rows.Select(r => r.Format(items)).ToList();
            // 如果需要转置表格
            if (transpose)
            {
                // 转置表格内容
                content = Enumerable.Range(0, cols).Select(col => content.Select(r => r[col]).ToArray()).ToList();
                // 更新列数为行数
                cols = rows.Count;
            }
            // 计算每列的最大宽度
            var colWidths = Enumerable.Range(0, cols).Select(col => content.Max(c => c[col].Length)).ToArray();

            // 遍历内容，根据每列的最大宽度进行对齐处理
            foreach (var row in content)
            {
                for (int col = 0; col < cols; col++)
                {
                    // 计算需要添加的空格数，实现左对齐和右对齐
                    var space = new string(' ', colWidths[col] - row[col].Length);
                    row[col] = col == 0 ? row[col] + space : space + row[col]; // left-align first col; right-align other cols
                }
            }

            // 创建 StringBuilder 对象
            var sb = new StringBuilder();
            // 根据每列的最大宽度创建水平分隔线
            var horizBars = colWidths.Select(w => new string('═', w)).ToArray();

            // 定义一个函数，用于将一行内容添加到 StringBuilder 中
            void OneRow(string[] cells, char before, char between, char after)
            {
                sb.Append(before);
                sb.AppendJoin(between, cells);
                sb.Append(after);
                sb.AppendLine();
            }

            // 使用水平线字符和角落字符创建一行
            OneRow(horizBars, '╔', '╦', '╗');
            bool first = true;
            // 遍历内容中的每一行
            foreach (var row in content)
            {
                // 如果是第一行，则不添加交叉线
                if (first)
                    first = false;
                else
                    // 添加交叉线
                    OneRow(horizBars, '╠', '╬', '╣');
                // 添加一行数据
                OneRow(row, '║', '║', '║');
            }
            // 添加底部边框
            OneRow(horizBars, '╚', '╩', '╝');

            // 打印表格
            Console.WriteLine(sb.ToString());
        }

        // 定义一个包含行数据的记录
        public record TableRow<T>(string Name, Func<T, object> Data, string Before = "", string After = "")
        {
            // 格式化每个项目
            private string FormatItem(T item) => $" {Before}{Data(item)}{After} ";
# 定义一个公共的字符串数组格式化方法，接受一个泛型集合作为参数，使用FormatItem方法对集合中的每个元素进行格式化，然后在数组开头添加一个以Name属性为值的字符串，最后将结果转换为数组并返回
public string[] Format(IEnumerable<T> items) => items.Select(FormatItem).Prepend($" {Name} ").ToArray();
```
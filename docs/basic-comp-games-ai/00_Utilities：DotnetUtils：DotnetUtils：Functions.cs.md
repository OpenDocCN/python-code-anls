# `d:/src/tocomm/basic-computer-games\00_Utilities\DotnetUtils\DotnetUtils\Functions.cs`

```
// 引入 System.Xml.Linq 命名空间
using System.Xml.Linq;
// 引入 System.Console 类
using static System.Console;

// 声明 DotnetUtils 命名空间
namespace DotnetUtils;

// 声明 Functions 类为静态类
public static class Functions {
    // 声明 getValue 方法，接受一个路径和一个或多个字符串参数，返回一个字符串
    public static string? getValue(string path, params string[] names) {
        // 如果参数 names 的长度为 0，则抛出 InvalidOperationException 异常
        if (names.Length == 0) { throw new InvalidOperationException(); }
        // 从指定路径加载 XML 文档，并获取根元素为 "Project" 的元素，再获取其子元素为 "PropertyGroup" 的元素
        var parent = XDocument.Load(path).Element("Project")?.Element("PropertyGroup");
        // 调用另一个 getValue 方法，传入 parent 和 names 参数
        return getValue(parent, names);
    }

    // 声明 getValue 方法，接受一个 XElement 类型的参数和一个或多个字符串参数，返回一个字符串
    public static string? getValue(XElement? parent, params string[] names) {
        // 如果参数 names 的长度为 0，则抛出 InvalidOperationException 异常
        if (names.Length == 0) { throw new InvalidOperationException(); }
        // 声明一个 XElement 类型的变量 elem，并初始化为 null
        XElement? elem = null;
        // 遍历 names 数组中的每个字符串
        foreach (var name in names) {
            // 获取 parent 元素的子元素为当前遍历到的字符串的元素
            elem = parent?.Element(name);
            // 如果 elem 不为 null，则跳出循环
            if (elem != null) { break; }
        }
        // 返回 elem 元素的值，如果 elem 为 null 则返回 null
        return elem?.Value;
    }
}
    }  # 结束 getChoice 方法的定义

    public static int getChoice(int maxValue) => getChoice(0, maxValue);  # 定义一个参数为最大值的 getChoice 方法，并调用另一个 getChoice 方法

    public static int getChoice(int minValue, int maxValue) {  # 定义一个参数为最小值和最大值的 getChoice 方法
        int result;  # 声明一个整型变量 result
        do {
            Write("? ");  # 输出问号
        } while (!int.TryParse(ReadLine(), out result) || result < minValue || result > maxValue);  # 循环直到用户输入的值为整数且在最小值和最大值范围内
        //WriteLine();  # 输出空行
        return result;  # 返回用户输入的值
    }
}  # 结束类的定义
```
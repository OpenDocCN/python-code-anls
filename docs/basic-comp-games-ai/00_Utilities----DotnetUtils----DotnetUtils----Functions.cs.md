# `basic-computer-games\00_Utilities\DotnetUtils\DotnetUtils\Functions.cs`

```py
// 引入 System.Xml.Linq 命名空间和 System.Console 类
using System.Xml.Linq;
using static System.Console;

// 声明 DotnetUtils 命名空间和 Functions 类
namespace DotnetUtils;

// 声明 Functions 类为静态类
public static class Functions {
    // 声明 getValue 方法，接受文件路径和参数名数组，返回字符串
    public static string? getValue(string path, params string[] names) {
        // 如果参数名数组长度为 0，则抛出 InvalidOperationException 异常
        if (names.Length == 0) { throw new InvalidOperationException(); }
        // 加载 XML 文档，并获取根元素为 "Project" 的元素下的 "PropertyGroup" 元素
        var parent = XDocument.Load(path).Element("Project")?.Element("PropertyGroup");
        // 调用另一个 getValue 方法，传入 parent 和参数名数组，返回结果
        return getValue(parent, names);
    }

    // 声明 getValue 方法，接受 XElement 和参数名数组，返回字符串
    public static string? getValue(XElement? parent, params string[] names) {
        // 如果参数名数组长度为 0，则抛出 InvalidOperationException 异常
        if (names.Length == 0) { throw new InvalidOperationException(); }
        // 声明 XElement 变量 elem，并初始化为 null
        XElement? elem = null;
        // 遍历参数名数组，获取 parent 元素下的对应元素
        foreach (var name in names) {
            elem = parent?.Element(name);
            // 如果找到元素，则跳出循环
            if (elem != null) { break; }
        }
        // 返回找到的元素的值
        return elem?.Value;
    }

    // 声明 getChoice 方法，接受最大值参数，返回整数
    public static int getChoice(int maxValue) => getChoice(0, maxValue);

    // 声明 getChoice 方法，接受最小值和最大值参数，返回整数
    public static int getChoice(int minValue, int maxValue) {
        // 声明整数变量 result
        int result;
        // 循环提示用户输入，直到输入合法的整数
        do {
            Write("? ");
        } while (!int.TryParse(ReadLine(), out result) || result < minValue || result > maxValue);
        // 返回用户输入的合法整数
        return result;
    }
}
```
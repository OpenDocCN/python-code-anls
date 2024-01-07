# `basic-computer-games\00_Utilities\DotnetUtils\DotnetUtils\Functions.cs`

```

// 引入 System.Xml.Linq 命名空间和 System.Console 类
using System.Xml.Linq;
using static System.Console;

// 定义 DotnetUtils 命名空间和 Functions 静态类
namespace DotnetUtils;

public static class Functions {
    // 从 XML 文件中获取指定路径下的元素值
    public static string? getValue(string path, params string[] names) {
        // 如果参数 names 的长度为 0，则抛出 InvalidOperationException 异常
        if (names.Length == 0) { throw new InvalidOperationException(); }
        // 加载 XML 文件，并获取指定路径下的元素
        var parent = XDocument.Load(path).Element("Project")?.Element("PropertyGroup");
        return getValue(parent, names);
    }

    // 从指定的 XElement 对象中获取指定名称的元素值
    public static string? getValue(XElement? parent, params string[] names) {
        // 如果参数 names 的长度为 0，则抛出 InvalidOperationException 异常
        if (names.Length == 0) { throw new InvalidOperationException(); }
        // 初始化 XElement 对象
        XElement? elem = null;
        // 遍历参数 names，获取指定名称的元素
        foreach (var name in names) {
            elem = parent?.Element(name);
            if (elem != null) { break; }
        }
        return elem?.Value;
    }

    // 获取用户输入的选择，范围为 0 到 maxValue
    public static int getChoice(int maxValue) => getChoice(0, maxValue);

    // 获取用户输入的选择，范围为 minValue 到 maxValue
    public static int getChoice(int minValue, int maxValue) {
        int result;
        // 循环直到用户输入合法的选择
        do {
            Write("? ");
        } while (!int.TryParse(ReadLine(), out result) || result < minValue || result > maxValue);
        // 返回用户输入的选择
        return result;
    }
}

```
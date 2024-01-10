# `basic-computer-games\00_Utilities\DotnetUtils\DotnetUtils\Globals.cs`

```
# 声明 DotnetUtils 命名空间
namespace DotnetUtils;

# 声明静态类 Globals
public static class Globals {
    # 声明静态只读的 LangData 字典，键为字符串，值为元组
    public static readonly Dictionary<string, (string codefileExtension, string projExtension)> LangData = new() {
        # 向 LangData 字典添加键值对，键为 "csharp"，值为元组 ("cs", "csproj")
        { "csharp", ("cs", "csproj") },
        # 向 LangData 字典添加键值对，键为 "vbnet"，值为元组 ("vb", "vbproj")
        { "vbnet", ("vb", "vbproj") }
    };
}
```
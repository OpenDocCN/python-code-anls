# `basic-computer-games\00_Utilities\DotnetUtils\DotnetUtils\PortInfo.cs`

```py
// 使用静态导入来简化代码，可以直接使用 Directory 和 Path 类的方法
using static System.IO.Directory;
using static System.IO.Path;
// 使用静态导入来简化代码，可以直接使用 DotnetUtils.Globals 类的方法
using static DotnetUtils.Globals;

// 命名空间 DotnetUtils
namespace DotnetUtils
{
    // 定义记录类型 PortInfo，包含多个属性
    public record PortInfo(
        string GamePath, string FolderName, int Index, string GameName,
        string LangPath, string Lang, string Ext, string ProjExt,
        string[] CodeFiles, string[] Slns, string[] Projs
    ) {
        // 创建静态只读的枚举选项对象，用于设置文件遍历的选项
        private static readonly EnumerationOptions enumerationOptions = new() {
            RecurseSubdirectories = true,
            MatchType = MatchType.Simple,
            MatchCasing = MatchCasing.CaseInsensitive
        };

        // 创建静态只读的特殊游戏名称字典，用于将以数字开头的游戏名称映射到特定字符串
        private static readonly Dictionary<string, string> specialGameNames = new() {
            { "3-D_Plot", "Plot" },
            { "3-D_Tic-Tac-Toe", "ThreeDTicTacToe" },
            { "23_Matches", "TwentyThreeMatches"}
        };
    }
}
    // 创建一个静态方法，用于根据游戏路径和语言关键词创建 PortInfo 对象
    public static PortInfo? Create(string gamePath, string langKeyword) {
        // 获取游戏路径中的文件夹名称
        var folderName = GetFileName(gamePath);
        // 使用下划线分割文件夹名称，获取索引和游戏名称
        var parts = folderName.Split('_', 2);

        // 如果分割后的部分长度小于等于1，则返回空
        if (parts.Length <= 1) { return null; }

        // 使用元组解构，尝试解析索引和游戏名称
        var (index, gameName) = (
            int.TryParse(parts[0], out var n) && n > 0 ? // 忽略工具文件夹
                n :
                (int?)null,
            specialGameNames.TryGetValue(parts[1], out var specialName) ?
                specialName :
                parts[1].Replace("_", "").Replace("-", "")
        );

        // 如果索引或游戏名称为空，则返回空
        if (index is null || gameName is null) { return null; }

        // 使用元组解构，获取语言数据中的扩展名和项目文件扩展名
        var (ext, projExt) = LangData[langKeyword];
        // 组合游戏路径和语言路径
        var langPath = Combine(gamePath, langKeyword);
        // 获取语言路径下的代码文件
        var codeFiles =
            GetFiles(langPath, $"*.{ext}", enumerationOptions)
                .Where(x => !x.Contains("\\bin\\") && !x.Contains("\\obj\\"))
                .ToArray();

        // 返回一个新的 PortInfo 对象
        return new PortInfo(
            gamePath, folderName, index.Value, gameName,
            langPath, langKeyword, ext, projExt,
            codeFiles,
            GetFiles(langPath, "*.sln", enumerationOptions),
            GetFiles(langPath, $"*.{projExt}", enumerationOptions)
        );
    }
# 闭合前面的函数定义
```
# `basic-computer-games\00_Utilities\DotnetUtils\DotnetUtils\PortInfo.cs`

```

// 引入静态类，简化代码中对 Directory 和 Path 的调用
using static System.IO.Directory;
using static System.IO.Path;
// 引入全局变量
using static DotnetUtils.Globals;

// 命名空间 DotnetUtils
namespace DotnetUtils;

// 定义 PortInfo 记录类型，包含游戏路径、文件夹名称、索引、游戏名称、语言路径、语言、扩展名、项目扩展名、代码文件数组、解决方案数组、项目数组
public record PortInfo(
    string GamePath, string FolderName, int Index, string GameName,
    string LangPath, string Lang, string Ext, string ProjExt,
    string[] CodeFiles, string[] Slns, string[] Projs
) {
    // 定义枚举选项
    private static readonly EnumerationOptions enumerationOptions = new() {
        RecurseSubdirectories = true,
        MatchType = MatchType.Simple,
        MatchCasing = MatchCasing.CaseInsensitive
    };

    // 特殊游戏名称映射表
    private static readonly Dictionary<string, string> specialGameNames = new() {
        { "3-D_Plot", "Plot" },
        { "3-D_Tic-Tac-Toe", "ThreeDTicTacToe" },
        { "23_Matches", "TwentyThreeMatches"}
    };

    // 创建 PortInfo 实例的静态方法
    public static PortInfo? Create(string gamePath, string langKeyword) {
        // 获取游戏路径中的文件夹名称
        var folderName = GetFileName(gamePath);
        // 将文件夹名称按下划线分割成两部分
        var parts = folderName.Split('_', 2);

        // 如果分割后的部分数量小于等于1，则返回空
        if (parts.Length <= 1) { return null; }

        // 解析索引和游戏名称
        var (index, gameName) = (
            int.TryParse(parts[0], out var n) && n > 0 ? // 忽略 utilities 文件夹
                n :
                (int?)null,
            specialGameNames.TryGetValue(parts[1], out var specialName) ?
                specialName :
                parts[1].Replace("_", "").Replace("-", "")
        );

        // 如果索引或游戏名称为空，则返回空
        if (index is null || gameName is null) { return null; }

        // 获取语言关键字对应的扩展名和项目扩展名
        var (ext, projExt) = LangData[langKeyword];
        // 拼接语言路径
        var langPath = Combine(gamePath, langKeyword);
        // 获取语言路径下的代码文件数组
        var codeFiles =
            GetFiles(langPath, $"*.{ext}", enumerationOptions)
                .Where(x => !x.Contains("\\bin\\") && !x.Contains("\\obj\\"))
                .ToArray();

        // 返回 PortInfo 实例
        return new PortInfo(
            gamePath, folderName, index.Value, gameName,
            langPath, langKeyword, ext, projExt,
            codeFiles,
            GetFiles(langPath, "*.sln", enumerationOptions),
            GetFiles(langPath, $"*.{projExt}", enumerationOptions)
        );
    }
}

```
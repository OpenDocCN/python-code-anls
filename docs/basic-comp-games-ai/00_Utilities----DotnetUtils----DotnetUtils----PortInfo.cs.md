# `00_Utilities\DotnetUtils\DotnetUtils\PortInfo.cs`

```
// 使用静态类别名引入 System.IO.Directory 命名空间
using static System.IO.Directory;
// 使用静态类别名引入 System.IO.Path 命名空间
using static System.IO.Path;
// 使用静态类别名引入 DotnetUtils.Globals 命名空间
using static DotnetUtils.Globals;

// 声明 DotnetUtils 命名空间
namespace DotnetUtils;

// 声明 PortInfo 记录类型，包含多个属性
public record PortInfo(
    string GamePath, string FolderName, int Index, string GameName,
    string LangPath, string Lang, string Ext, string ProjExt,
    string[] CodeFiles, string[] Slns, string[] Projs
) {
    // 声明静态只读的 EnumerationOptions 对象，设置 RecurseSubdirectories、MatchType 和 MatchCasing 属性
    private static readonly EnumerationOptions enumerationOptions = new() {
        RecurseSubdirectories = true,
        MatchType = MatchType.Simple,
        MatchCasing = MatchCasing.CaseInsensitive
    };

    // .NET 命名空间不能以数字开头
    // 对于以数字开头的游戏名称，将名称映射到特定的字符串
    private static readonly Dictionary<string, string> specialGameNames = new() {
        { "3-D_Plot", "Plot" }, // 将特殊游戏名称"3-D_Plot"映射为"Plot"
        { "3-D_Tic-Tac-Toe", "ThreeDTicTacToe" }, // 将特殊游戏名称"3-D_Tic-Tac-Toe"映射为"ThreeDTicTacToe"
        { "23_Matches", "TwentyThreeMatches"} // 将特殊游戏名称"23_Matches"映射为"TwentyThreeMatches"
    };

    public static PortInfo? Create(string gamePath, string langKeyword) {
        var folderName = GetFileName(gamePath); // 获取游戏路径中的文件夹名称
        var parts = folderName.Split('_', 2); // 使用下划线分割文件夹名称，最多分割成两部分

        if (parts.Length <= 1) { return null; } // 如果分割后的部分数量小于等于1，则返回空值

        var (index, gameName) = (
            int.TryParse(parts[0], out var n) && n > 0 ? // 尝试将第一个部分转换为整数，如果大于0则赋值给index
                n :
                (int?)null, // 否则赋值为null
            specialGameNames.TryGetValue(parts[1], out var specialName) ? // 尝试从特殊游戏名称字典中获取对应的值
                specialName :
                parts[1].Replace("_", "").Replace("-", "") // 如果不存在对应的特殊名称，则将下划线和破折号替换为空字符串
        );
        # 检查 index 和 gameName 是否为空，如果有一个为空则返回空
        if (index is null || gameName is null) { return null; }

        # 从 LangData 中获取 langKeyword 对应的 ext 和 projExt
        var (ext, projExt) = LangData[langKeyword];
        
        # 组合 gamePath 和 langKeyword 得到 langPath
        var langPath = Combine(gamePath, langKeyword);
        
        # 获取 langPath 下所有扩展名为 ext 的文件，并排除包含 "\\bin\\" 和 "\\obj\\" 的文件
        var codeFiles =
            GetFiles(langPath, $"*.{ext}", enumerationOptions)
                .Where(x => !x.Contains("\\bin\\") && !x.Contains("\\obj\\"))
                .ToArray();

        # 返回一个新的 PortInfo 对象，包括 gamePath、folderName、index、gameName、langPath、langKeyword、ext、projExt、codeFiles、以及扩展名为 .sln 和 projExt 的文件
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
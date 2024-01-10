# `basic-computer-games\00_Utilities\find-missing-implementations.js`

```
/**
 * Program to find games that are missing solutions in a given language
 *
 * Scan each game folder, check for a folder for each language, and also make
 * sure there's at least one file of the expected extension and not just a
 * readme or something
 */

const fs = require("fs"); // 引入文件系统模块
const glob = require("glob"); // 引入文件匹配模块

// relative path to the repository root
const ROOT_PATH = "../."; // 仓库根目录的相对路径

const languages = [ // 支持的编程语言及其对应的文件扩展名
  { name: "csharp", extension: "cs" },
  { name: "java", extension: "java" },
  { name: "javascript", extension: "html" },
  { name: "pascal", extension: "pas" },
  { name: "perl", extension: "pl" },
  { name: "python", extension: "py" },
  { name: "ruby", extension: "rb" },
  { name: "vbnet", extension: "vb" },
];

const getFilesRecursive = async (path, extension) => { // 异步函数，递归获取指定路径下指定扩展名的文件
  return new Promise((resolve, reject) => {
    glob(`${path}/**/*.${extension}`, (err, matches) => {
      if (err) {
        reject(err);
      }
      resolve(matches);
    });
  });
};

const getPuzzleFolders = () => { // 获取谜题文件夹列表
  return fs
    .readdirSync(ROOT_PATH, { withFileTypes: true }) // 同步读取指定路径下的文件夹
    .filter((dirEntry) => dirEntry.isDirectory()) // 过滤出文件夹
    .filter(
      (dirEntry) =>
        ![".git", "node_modules", "00_Utilities"].includes(dirEntry.name) // 过滤掉特定的文件夹
    )
    .map((dirEntry) => dirEntry.name); // 映射出文件夹名列表
};

(async () => { // 异步自执行函数
  let missingGames = {}; // 未找到解决方案的游戏
  let missingLanguageCounts = {}; // 未找到解决方案的语言计数
  languages.forEach((l) => (missingLanguageCounts[l.name] = 0)); // 初始化未找到解决方案的语言计数
  const puzzles = getPuzzleFolders(); // 获取谜题文件夹列表
  for (const puzzle of puzzles) { // 遍历谜题文件夹列表
    for (const { name: language, extension } of languages) { // 遍历支持的编程语言
      const files = await getFilesRecursive( // 获取指定路径下指定扩展名的文件
        `${ROOT_PATH}/${puzzle}/${language}`,
        extension
      );
      if (files.length === 0) { // 如果文件列表为空
        if (!missingGames[puzzle]) missingGames[puzzle] = []; // 如果未找到解决方案的游戏中没有当前游戏，则添加
        missingGames[puzzle].push(language); // 将当前语言添加到未找到解决方案的游戏中
        missingLanguageCounts[language]++; // 未找到解决方案的语言计数加一
      }
    }
  }
  const missingCount = Object.values(missingGames).flat().length; // 计算未找到解决方案的游戏总数
  if (missingCount === 0) { // 如果未找到解决方案的游戏总数为零
    # 打印所有游戏都有所有语言的解决方案的消息
    console.log("All games have solutions for all languages");
  } else {
    # 打印缺少实现的游戏数量
    console.log(`Missing ${missingCount} implementations:`);

    # 遍历缺少实现的游戏对象，将其值转换为逗号分隔的字符串
    Object.entries(missingGames).forEach(
      ([p, ls]) => (missingGames[p] = ls.join(", "))
    );

    # 打印缺少语言的游戏对象的表格
    console.log(`\nMissing languages by game:`);
    console.table(missingGames);

    # 打印缺少语言的计数对象的表格
    console.log(`\nBy language:`);
    console.table(missingLanguageCounts);
  }
// 调用立即执行函数并结束函数的执行
})();
// 返回并结束函数的执行
return;
```
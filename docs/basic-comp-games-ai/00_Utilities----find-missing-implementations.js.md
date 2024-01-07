# `basic-computer-games\00_Utilities\find-missing-implementations.js`

```
/**
 * Program to find games that are missing solutions in a given language
 * 用于查找缺少特定语言解决方案的游戏的程序
 *
 * Scan each game folder, check for a folder for each language, and also make
 * sure there's at least one file of the expected extension and not just a
 * readme or something
 * 扫描每个游戏文件夹，检查每种语言的文件夹，并确保至少有一个符合预期扩展名的文件，而不仅仅是一个自述文件或其他内容
 */

const fs = require("fs"); // 引入文件系统模块
const glob = require("glob"); // 引入文件匹配模块

// relative path to the repository root
// 仓库根目录的相对路径
const ROOT_PATH = "../.";

const languages = [ // 支持的语言列表
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
    .readdirSync(ROOT_PATH, { withFileTypes: true }) // 同步读取指定目录下的文件
    .filter((dirEntry) => dirEntry.isDirectory()) // 过滤出文件夹
    .filter(
      (dirEntry) =>
        ![".git", "node_modules", "00_Utilities"].includes(dirEntry.name) // 过滤掉特定文件夹
    )
    .map((dirEntry) => dirEntry.name); // 返回文件夹名称列表
};

(async () => { // 异步函数
  let missingGames = {}; // 缺失游戏的对象
  let missingLanguageCounts = {}; // 缺失语言计数的对象
  languages.forEach((l) => (missingLanguageCounts[l.name] = 0)); // 初始化缺失语言计数对象
  const puzzles = getPuzzleFolders(); // 获取谜题文件夹列表
  for (const puzzle of puzzles) { // 遍历谜题文件夹
    for (const { name: language, extension } of languages) { // 遍历语言列表
      const files = await getFilesRecursive( // 获取指定路径下指定扩展名的文件列表
        `${ROOT_PATH}/${puzzle}/${language}`,
        extension
      );
      if (files.length === 0) { // 如果文件列表为空
        if (!missingGames[puzzle]) missingGames[puzzle] = []; // 如果缺失游戏对象中没有当前谜题，则添加
        missingGames[puzzle].push(language); // 将缺失的语言添加到缺失游戏对象中
        missingLanguageCounts[language]++; // 缺失语言计数加一
      }
    }
  }
  const missingCount = Object.values(missingGames).flat().length; // 计算缺失的语言数量
  if (missingCount === 0) { // 如果缺失数量为0
    console.log("All games have solutions for all languages"); // 输出所有游戏都有所有语言的解决方案
  } else {
    console.log(`Missing ${missingCount} implementations:`); // 输出缺失的实现数量

    Object.entries(missingGames).forEach( // 遍历缺失游戏对象
      ([p, ls]) => (missingGames[p] = ls.join(", ")) // 将缺失的语言列表转换为字符串
    );

    console.log(`\nMissing languages by game:`); // 输出缺失语言按游戏分类
    console.table(missingGames); // 以表格形式输出缺失游戏对象

    console.log(`\nBy language:`); // 输出按语言分类
    console.table(missingLanguageCounts); // 以表格形式输出缺失语言计数对象
  }
})();

return; // 返回结果


```
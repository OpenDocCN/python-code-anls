# `basic-computer-games\00_Utilities\find-unimplemented.js`

```
/**
 * Program to show unimplemented games by language, optionally filtered by
 * language
 *
 * Usage: node find-unimplemented.js [[[lang1] lang2] ...]
 *
 * Adapted from find-missing-implementtion.js
 */

const fs = require("fs"); // 引入文件系统模块
const glob = require("glob"); // 引入文件匹配模块

// relative path to the repository root
const ROOT_PATH = "../."; // 仓库根目录的相对路径

let languages = [ // 支持的编程语言及其文件扩展名
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
    .readdirSync(ROOT_PATH, { withFileTypes: true }) // 同步读取根目录下的文件和文件夹
    .filter((dirEntry) => dirEntry.isDirectory()) // 过滤出文件夹
    .filter(
      (dirEntry) =>
        ![".git", "node_modules", "00_Utilities", "buildJvm"].includes(dirEntry.name) // 过滤掉特定的文件夹
    )
    .map((dirEntry) => dirEntry.name); // 映射出文件夹名列表
};

(async () => { // 异步自执行函数
  const result = {}; // 存储结果的对象
  if (process.argv.length > 2) { // 如果命令行参数大于2
    languages = languages.filter((language) => process.argv.slice(2).includes(language.name)); // 根据命令行参数过滤语言列表
  }
  for (const { name: language } of languages) { // 遍历语言列表
    result[language] = []; // 初始化结果对象
  }

  const puzzleFolders = getPuzzleFolders(); // 获取谜题文件夹列表
  for (const puzzleFolder of puzzleFolders) { // 遍历谜题文件夹列表
    for (const { name: language, extension } of languages) { // 遍历语言列表
      const files = await getFilesRecursive( // 获取指定路径下指定扩展名的文件列表
        `${ROOT_PATH}/${puzzleFolder}/${language}`, extension
      );
      if (files.length === 0) { // 如果文件列表为空
        result[language].push(puzzleFolder); // 将文件夹名加入对应语言的结果列表
      }
    }
  }
  console.log('Unimplementation by language:') // 输出提示信息
  console.dir(result); // 输出结果对象
})();

return; // 返回
```
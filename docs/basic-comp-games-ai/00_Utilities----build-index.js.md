# `basic-computer-games\00_Utilities\build-index.js`

```py
#!/usr/bin/env node
/**
 * This script creates an "index.html" file in the root of the directory.
 * 
 * Call this from the root of the project with 
 *  `node ./00_Utilities/build-index.js`
 * 
 * @author Alexander Wunschik <https://github.com/mojoaxel>
 */

const fs = require('fs'); // 引入文件系统模块
const path = require('path'); // 引入路径模块

const TITLE = 'BASIC Computer Games'; // 定义标题常量
const JAVASCRIPT_FOLDER = 'javascript'; // 定义 JavaScript 文件夹常量
const IGNORE_FOLDERS_START_WITH = ['.', '00_', 'buildJvm', 'Sudoku']; // 定义需要忽略的文件夹列表
const IGNORE_FILES = [
    // "84 Super Star Trek"  has it's own node/js implementation (using xterm)
    'cli.mjs', 'superstartrek.mjs'
];

function createGameLinks(game) {
    const creatFileLink = (file, name = path.basename(file)) => { // 创建游戏文件链接的函数
        if (file.endsWith('.html')) { // 如果文件以 .html 结尾
            return `
                <li><a href="${file}">${name.replace('.html', '')}</a></li>
            `;
        } else if (file.endsWith('.mjs')) { // 如果文件以 .mjs 结尾
            return `
                <li><a href="./00_Common/javascript/WebTerminal/terminal.html#${file}">${name.replace('.mjs', '')} (node.js)</a></li>
            `;
        } else {
            throw new Error(`Unknown file-type found: ${file}`); // 抛出错误，未知的文件类型
        }
    }

    if (game.files.length > 1) { // 如果游戏文件数量大于1
        const entries = game.files.map(file => {
            return creatFileLink(file);
        });
        return `
            <li>
                <span>${game.name}</span>
                <ul>${entries.map(e => `\t\t\t${e}`).join('\n')}</ul>
            </li>
        `;
    } else {
        return creatFileLink(game.files[0], game.name);
    }
}

function createIndexHtml(title, games) {
    const listEntries = games.map(game => 
        createGameLinks(game)
    ).map(entry => `\t\t\t${entry}`).join('\n'); // 创建游戏链接的 HTML 内容

    const head = `
        <head>
            <meta charset="UTF-8">
            <title>${title}</title>
            <link rel="stylesheet" href="./00_Utilities/javascript/style_terminal.css" />
        </head>
    `;
    # 创建 HTML 页面的主体内容
    const body = `
        <body>
            <article id="output">
                <header>
                    <h1>${title}</h1>  # 在文章标题标签中插入变量 title 的内容
                </header>
                <main>
                    <ul>
                        ${listEntries}  # 在列表中插入变量 listEntries 的内容
                    </ul>
                </main>
            </article>
        </body>
    `;

    # 返回完整的 HTML 页面
    return `
        <!DOCTYPE html>
        <html lang="en">
        ${head}  # 在 HTML 中插入变量 head 的内容
        ${body}  # 在 HTML 中插入变量 body 的内容
        </html>
    `.trim().replace(/\s\s+/g, '');  # 去除多余的空白字符
// 查找文件夹中的所有 JavaScript 文件
function findJSFilesInFolder(folder) {
    // 过滤掉不包含名为 "javascript" 的子文件夹的文件夹
    const hasJavascript = fs.existsSync(`${folder}/${JAVASCRIPT_FOLDER}`);
    if (!hasJavascript) {
        throw new Error(`Game "${folder}" is missing a javascript folder`);
    }

    // 获取 javascript 文件夹中的所有文件
    const files = fs.readdirSync(`${folder}/${JAVASCRIPT_FOLDER}`);

    // 过滤出只允许 .html 文件
    const htmlFiles = files.filter(file => file.endsWith('.html'));
    const mjsFiles = files.filter(file => file.endsWith('.mjs'));
    const entries = [
        ...htmlFiles,
        ...mjsFiles
    ].filter(file => !IGNORE_FILES.includes(file));

    if (entries.length == 0) {
        throw new Error(`Game "${folder}" is missing a HTML or node.js file in the folder "${folder}/${JAVASCRIPT_FOLDER}"`);
    }

    return entries.map(file => path.join(folder, JAVASCRIPT_FOLDER, file));
}

// 主函数
function main() {
    // 获取当前目录中所有文件夹的列表
    let folders = fs.readdirSync(process.cwd());

    // 过滤出只允许文件夹
    folders = folders.filter(folder => fs.statSync(folder).isDirectory());

    // 过滤掉以点或 00_ 开头的文件夹
    folders = folders.filter(folder => {
        return !IGNORE_FOLDERS_START_WITH.some(ignore => folder.startsWith(ignore));
    });

    // 按字母顺序（按数字）对文件夹进行排序
    folders = folders.sort();

    // 从文件夹中获取名称和 JavaScript 文件
    const games = folders.map(folder => {
        const name = folder.replace('_', ' ');
        let files;
        
        try {
            files = findJSFilesInFolder(folder);
        } catch (error) {
            console.warn(`Game "${name}" is missing a javascript implementation: ${error.message}`);
            return null;
        }

        return {
            name,
            files
        }
    }).filter(game => game !== null);
}
    // 使用createIndexHtml函数创建一个包含所有游戏列表的index.html文件内容
    const htmlContent = createIndexHtml(TITLE, games);
    // 将htmlContent写入index.html文件中
    fs.writeFileSync('index.html', htmlContent);
    // 打印成功创建index.html的消息
    console.log(`index.html successfully created!`);
# 调用名为main的函数
main();
```
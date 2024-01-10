# `basic-computer-games\01_Acey_Ducey\javascript\aceyducey.js`

```
// UTILITY VARIABLES

// 默认情况下：
// — 浏览器有一个 window 对象
// — Node.js 没有
// 检查未定义的 window 对象是一种宽松的检查，以支持浏览器和 Node.js
const isRunningInBrowser = typeof window !== 'undefined';

// 用于使用实用函数轻松验证输入字符串
const validLowerCaseYesStrings = ['yes', 'y'];
const validLowerCaseNoStrings = ['no', 'n'];
const validLowerCaseYesAndNoStrings = [
    ...validLowerCaseYesStrings,
    ...validLowerCaseNoStrings,
];
// UTILITY VARIABLES

// 获取随机数（牌）2-14（ACE 是 14）
function getRandomCard() {
    // 在我们的游戏中，ACE 的值大于花色牌；
    // 我们不希望 ACE 的值为 1，而是希望它为 14。
    // 因此，我们希望将随机数的范围从 1-13 转换为 2-14
    let min = 2;
    let max = 14;
    // 返回两个值之间的随机整数，包括这两个值
    return Math.floor(Math.random() * (max - min + 1) + min);
}

function newGameCards() {
    let cardOne = getRandomCard();
    let cardTwo = getRandomCard();
    let cardThree = getRandomCard();
    // 我们希望：
    // 1. cardOne 和 cardTwo 是不同的牌
    // 2. cardOne 小于 cardTwo
    // 因此，当 cardOne 大于或等于 cardTwo 时，
    // 我们将继续生成随机牌。
    while (cardOne >= cardTwo) {
        cardOne = getRandomCard();
        cardTwo = getRandomCard();
    }
    return [cardOne, cardTwo, cardThree];
}

// 获取牌值的函数
function getCardValue(card) {
    let faceOrAce = {
        11: 'JACK',
        12: 'QUEEN',
        13: 'KING',
        14: 'ACE',
    };
    // 如果牌值与 faceOrAce 中的键匹配，则使用 faceOrAce 的值；
    // 否则，返回 undefined 并使用空值合并运算符（??）处理，并默认为牌值。
    let cardValue = faceOrAce[card] ?? card;
    return cardValue;
}

print(spaces(26) + 'ACEY DUCEY CARD GAME');
# 打印标题和地址
print(spaces(15) + 'CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n');
# 打印游戏规则
print('ACEY-DUCEY IS PLAYED IN THE FOLLOWING MANNER');
print('THE DEALER (COMPUTER) DEALS TWO CARDS FACE UP');
print('YOU HAVE AN OPTION TO BET OR NOT BET DEPENDING');
print('ON WHETHER OR NOT YOU FEEL THE CARD WILL HAVE');
print('A VALUE BETWEEN THE FIRST TWO.');
print("IF YOU DO NOT WANT TO BET, INPUT '0'");

# 调用主函数
main();

# 主函数
async function main() {
    # 初始化赌注和可用资金
    let bet;
    let availableDollars = 100;

    # 游戏循环
    while (true) {
        # 待添加游戏逻辑
    }
}

# 实用函数
# 检查输入字符串是否为有效的yes或no字符串
function isValidYesNoString(string) {
    return validLowerCaseYesAndNoStrings.includes(string.toLowerCase());
}

# 检查输入字符串是否为有效的yes字符串
function isValidYesString(string) {
    return validLowerCaseYesStrings.includes(string.toLowerCase());
}

# 检查输入字符串是否为有效的no字符串
function isValidNoString(string) {
    return validLowerCaseNoStrings.includes(string.toLowerCase());
}

# 打印函数
function print(string) {
    if (isRunningInBrowser) {
        # 在浏览器中打印，添加换行符以匹配console.log的行为
        document
            .getElementById('output')
            .appendChild(document.createTextNode(string + '\n'));
    } else {
        # 在控制台中打印
        console.log(string);
    }
}

# 输入函数
function input() {
    if (isRunningInBrowser) {
        # 从浏览器DOM输入接受输入
        return new Promise((resolve) => {
            const outputElement = document.querySelector('#output');
            const inputElement = document.createElement('input');
            outputElement.append(inputElement);
            inputElement.focus();

            inputElement.addEventListener('keydown', (event) => {
                if (event.key === 'Enter') {
                    const result = inputElement.value;
                    inputElement.remove();
                    print(result);
                    print('');
                    resolve(result);
                }
            });
        });
    }
}
    } else {
        // 如果不是浏览器环境，则从命令行接受输入
        // 参考：https://nodejs.dev/learn/accept-input-from-the-command-line-in-nodejs
        // 返回一个 Promise 对象
        return new Promise(function (resolve) {
            // 创建一个接口，用于读取命令行输入
            const readline = require('readline').createInterface({
                input: process.stdin,  // 设置输入流
                output: process.stdout,  // 设置输出流
            });
            // 询问用户输入，并在用户输入后解析并返回结果
            readline.question('', function (input) {
                resolve(input);  // 解析并返回用户输入
                readline.close();  // 关闭接口
            });
        });
    }
# 结束当前的函数定义
}

# 定义一个名为printInline的函数，用于在浏览器或终端中输出字符串
function printInline(string) {
    # 如果代码在浏览器中运行，则将字符串添加到id为'output'的元素中
    if (isRunningInBrowser) {
        document
            .getElementById('output')
            .appendChild(document.createTextNode(string));
    } else {
        # 如果代码在终端中运行，则使用process.stdout.write输出字符串
        process.stdout.write(string);
    }
}

# 定义一个名为spaces的函数，用于生成指定数量的空格字符串
function spaces(numberOfSpaces) {
    return ' '.repeat(numberOfSpaces);
}

# 工具函数部分结束
```
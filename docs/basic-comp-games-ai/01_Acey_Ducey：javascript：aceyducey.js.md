# `01_Acey_Ducey\javascript\aceyducey.js`

```
// UTILITY VARIABLES

// By default:
// — Browsers have a window object
// — Node.js does not
// Checking for an undefined window object is a loose check
// to enable browser and Node.js support
const isRunningInBrowser = typeof window !== 'undefined';

// To easily validate input strings with utility functions
const validLowerCaseYesStrings = ['yes', 'y'];
const validLowerCaseNoStrings = ['no', 'n'];
const validLowerCaseYesAndNoStrings = [
    ...validLowerCaseYesStrings,
    ...validLowerCaseNoStrings,
];
// UTILITY VARIABLES

// Function to get a random number (card) 2-14 (ACE is 14)
function getRandomCard() {
    // In our game, the value of ACE is greater than face cards;
    // instead of having the value of ACE be 1, we’ll have it be 14.
    // So, we want to shift the range of random numbers from 1-13 to 2-14
    let min = 2; // 设置随机数范围的最小值为2
    let max = 14; // 设置随机数范围的最大值为14
    // Return random integer between two values, inclusive
    return Math.floor(Math.random() * (max - min + 1) + min); // 生成2到14之间的随机整数
}

function newGameCards() {
    let cardOne = getRandomCard(); // 获取第一张随机卡牌
    let cardTwo = getRandomCard(); // 获取第二张随机卡牌
    let cardThree = getRandomCard(); // 获取第三张随机卡牌
    // We want:
    // 1. cardOne and cardTwo to be different cards
    // 2. cardOne to be lower than cardTwo
    // So, while cardOne is greater than or equal too cardTwo
    // we will continue to generate random cards.
    while (cardOne >= cardTwo) { // 当第一张卡牌大于等于第二张卡牌时
        cardOne = getRandomCard(); // 继续生成随机卡牌，直到满足条件
        cardTwo = getRandomCard();  # 从随机生成的卡片中获取第二张卡片
    }
    return [cardOne, cardTwo, cardThree];  # 返回三张卡片的列表
}

// Function to get card value
function getCardValue(card) {
    let faceOrAce = {  # 创建一个对象，用于存储特殊卡片值和对应的名称
        11: 'JACK',
        12: 'QUEEN',
        13: 'KING',
        14: 'ACE',
    };
    // If card value matches a key in faceOrAce, use faceOrAce value;  # 如果卡片值在faceOrAce对象中有对应的键，使用对应的值
    // Else, return undefined and handle with the Nullish Coalescing Operator (??)  # 否则返回undefined，并使用Nullish Coalescing Operator (??)进行处理
    // and default to card value.  # 并默认使用卡片的值
    let cardValue = faceOrAce[card] ?? card;  # 获取卡片的值，如果在faceOrAce对象中有对应的值则使用，否则使用卡片的原始值
    return cardValue;  # 返回卡片的值
}
# 打印游戏标题
print(spaces(26) + 'ACEY DUCEY CARD GAME');
# 打印游戏信息
print(spaces(15) + 'CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n');
print('ACEY-DUCEY IS PLAYED IN THE FOLLOWING MANNER');
print('THE DEALER (COMPUTER) DEALS TWO CARDS FACE UP');
print('YOU HAVE AN OPTION TO BET OR NOT BET DEPENDING');
print('ON WHETHER OR NOT YOU FEEL THE CARD WILL HAVE');
print('A VALUE BETWEEN THE FIRST TWO.');
print("IF YOU DO NOT WANT TO BET, INPUT '0'");

# 调用主函数
main();

# 异步主函数
async function main() {
    # 初始化赌注和可用资金
    let bet;
    let availableDollars = 100;

    # 游戏循环
    while (true) {
        # 获取新一轮游戏的三张牌
        let [cardOne, cardTwo, cardThree] = newGameCards();

        # 打印玩家当前可用资金
        print(`YOU NOW HAVE ${availableDollars} DOLLARS.\n`);
        // 打印提示信息，显示玩家的下两张牌
        print('HERE ARE YOUR NEXT TWO CARDS: ');
        // 打印第一张牌的点数
        print(getCardValue(cardOne));
        // 打印第二张牌的点数
        print(getCardValue(cardTwo));
        // 打印空行
        print('');

        // 循环直到接收到有效的赌注
        let validBet = false;
        while (!validBet) {
            // 提示玩家输入赌注
            print('\nWHAT IS YOUR BET? ');
            // 将输入的赌注转换为整数
            bet = parseInt(await input(), 10);
            // 设置最低要求的赌注
            let minimumRequiredBet = 0;
            // 如果赌注大于等于最低要求的赌注
            if (bet >= minimumRequiredBet) {
                // 如果赌注大于可用的资金
                if (bet > availableDollars) {
                    // 提示玩家赌注过高
                    print('SORRY, MY FRIEND, BUT YOU BET TOO MUCH.');
                    // 显示玩家可用的资金
                    print(`YOU HAVE ONLY ${availableDollars} DOLLARS TO BET.`);
                } else {
                    // 赌注有效，退出循环
                    validBet = true;
                }
            }
        }
        if (bet == 0)
        {
            // 如果玩家选择不下注。
            print('CHICKEN!!');  // 打印“CHICKEN!!”
            print('');  // 打印空行
            // 不抽取第三张牌，重新抽取一组2张牌。
            continue;  // 继续循环
        }

        print('\n\nHERE IS THE CARD WE DREW: ');  // 打印“这是我们抽取的牌：”
        print(getCardValue(cardThree));  // 打印抽取的第三张牌的值

        // 确定玩家是赢了还是输了
        if (cardThree > cardOne && cardThree < cardTwo) {  // 如果第三张牌的值在第一张牌和第二张牌之间
            print('YOU WIN!!!');  // 打印“你赢了！！！”
            availableDollars = availableDollars + bet;  // 可用资金增加下注金额
        } else {
            print('SORRY, YOU LOSE');  // 打印“对不起，你输了”
            # 如果赌注大于等于可用金额
            if (bet >= availableDollars) {
                # 打印空行
                print('');
                print('');
                # 打印提示信息
                print('SORRY, FRIEND, BUT YOU BLEW YOUR WAD.');
                print('');
                print('');
                # 打印提示信息
                print('TRY AGAIN (YES OR NO)');

                # 等待用户输入
                let tryAgainInput = await input();

                # 打印空行
                print('');
                print('');

                # 如果用户输入有效的"YES"字符串
                if (isValidYesString(tryAgainInput)) {
                    # 重置可用金额为100
                    availableDollars = 100;
                } else {
                    # 打印提示信息
                    print('O.K., HOPE YOU HAD FUN!');
                    # 跳出循环
                    break;
                }
            } else {
// 减去赌注金额
availableDollars = availableDollars - bet;
}

// 实用函数
// 检查输入字符串是否为有效的“是”或“否”
function isValidYesNoString(string) {
    return validLowerCaseYesAndNoStrings.includes(string.toLowerCase());
}

// 检查输入字符串是否为有效的“是”
function isValidYesString(string) {
    return validLowerCaseYesStrings.includes(string.toLowerCase());
}

// 检查输入字符串是否为有效的“否”
function isValidNoString(string) {
    return validLowerCaseNoStrings.includes(string.toLowerCase());
}

// 打印输出字符串
function print(string) {
    if (isRunningInBrowser) {
        // 如果代码在浏览器中运行，将字符串添加到输出元素中，并在末尾添加换行符以匹配console.log的行为
        document
            .getElementById('output')
            .appendChild(document.createTextNode(string + '\n'));
    } else {
        // 如果代码不在浏览器中运行，使用console.log输出字符串
        console.log(string);
    }
}

function input() {
    if (isRunningInBrowser) {
        // 如果代码在浏览器中运行，从浏览器DOM输入中接受输入
        return new Promise((resolve) => {
            const outputElement = document.querySelector('#output');
            const inputElement = document.createElement('input');
            outputElement.append(inputElement);
            inputElement.focus();

            // 监听输入元素的键盘事件
            inputElement.addEventListener('keydown', (event) => {
                if (event.key === 'Enter') {  # 如果按下的键是回车键
                    const result = inputElement.value;  # 获取输入框中的值
                    inputElement.remove();  # 移除输入框
                    print(result);  # 打印输入的结果
                    print('');  # 打印空行
                    resolve(result);  # 返回输入的结果
                }
            });
        });
    } else {
        // Accept input from the command line in Node.js  # 从 Node.js 的命令行接受输入
        // See: https://nodejs.dev/learn/accept-input-from-the-command-line-in-nodejs  # 参考链接：https://nodejs.dev/learn/accept-input-from-the-command-line-in-nodejs
        return new Promise(function (resolve) {  # 返回一个 Promise 对象
            const readline = require('readline').createInterface({  # 创建一个接口用于读取输入
                input: process.stdin,  # 输入来自标准输入流
                output: process.stdout,  # 输出到标准输出流
            });
            readline.question('', function (input) {  # 询问用户输入
                resolve(input);  # 返回用户输入的结果
                readline.close();  # 关闭接口
// 结束了之前的函数定义
            });
        });
    }
}

// 在浏览器中打印输出
function printInline(string) {
    if (isRunningInBrowser) {
        // 在输出元素中添加文本节点
        document
            .getElementById('output')
            .appendChild(document.createTextNode(string));
    } else {
        // 在控制台中输出
        process.stdout.write(string);
    }
}

// 返回指定数量的空格字符串
function spaces(numberOfSpaces) {
    return ' '.repeat(numberOfSpaces);
}

// 实用函数
bio = BytesIO(open(fname, 'rb').read())  # 根据 ZIP 文件名读取其二进制，封装成字节流
zip = zipfile.ZipFile(bio, 'r')  # 使用字节流里面内容创建 ZIP 对象
fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
zip.close()  # 关闭 ZIP 对象
```
# `basic-computer-games\94_War\javascript\war.js`

```
// WAR
//
// Original conversion from BASIC to Javascript by Oscar Toledo G. (nanochess)
//

// 定义一个打印函数，将字符串添加到输出元素中
function print(str) {
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义一个制表符函数，返回指定数量的空格字符串
function tab(space) {
    let str = "";
    while (space-- > 0) {
        str += " ";
    }
    return str;
}

// 定义一个输入函数，返回一个 Promise 对象
function input() {
    return new Promise(function (resolve) {
        // 创建一个输入元素
        const input_element = document.createElement("INPUT");

        // 在输出元素中添加问号提示
        print("? ");
        // 设置输入元素的类型和长度
        input_element.setAttribute("type", "text");
        input_element.setAttribute("length", "50");
        document.getElementById("output").appendChild(input_element);
        // 让输入元素获得焦点
        input_element.focus();
        // 监听键盘按下事件
        input_element.addEventListener("keydown", function (event) {
            // 如果按下的是回车键
            if (event.keyCode == 13) {
                // 获取输入的字符串
                const input_str = input_element.value;
                // 从输出元素中移除输入元素
                document.getElementById("output").removeChild(input_element);
                // 打印输入的字符串并换行
                print(input_str);
                print("\n");
                // 解析 Promise 对象
                resolve(input_str);
            }
        });
    });
}

// 定义一个异步函数，询问用户是或否的问题
async function askYesOrNo(question) {
    while (1) {
        // 打印问题
        print(question);
        // 获取用户输入的字符串并转换为大写
        const str = (await input()).toUpperCase();
        // 如果输入是 "YES"，返回 true
        if (str === "YES") {
            return true;
        }
        // 如果输入是 "NO"，返回 false
        else if (str === "NO") {
            return false;
        }
        // 如果输入既不是 "YES" 也不是 "NO"，提示用户重新输入
        else {
            print("YES OR NO, PLEASE.  ");
        }
    }
}

// 定义一个异步函数，询问用户是否需要游戏说明
async function askAboutInstructions() {
    // 获取用户是否需要游戏说明的答案
    const playerWantsInstructions = await askYesOrNo("DO YOU WANT DIRECTIONS");
    // 如果用户需要游戏说明，打印游戏说明
    if (playerWantsInstructions) {
        print("THE COMPUTER GIVES YOU AND IT A 'CARD'.  THE HIGHER CARD\n");
        print("(NUMERICALLY) WINS.  THE GAME ENDS WHEN YOU CHOOSE NOT TO\n");
        print("CONTINUE OR WHEN YOU HAVE FINISHED THE PACK.\n");
    }
    // 打印空行
    print("\n");
    print("\n");
}

// 定义一个创建游戏牌组的函数
function createGameDeck(cards, gameSize) {
    // 创建一个空的牌组数组
    const deck = [];
    // 获取牌的总数
    const deckSize = cards.length;
    # 遍历游戏大小范围内的索引
    for (let j = 0; j < gameSize; j++) {
        # 声明一个变量用于存储卡片索引
        let card;

        # 计算一个新的卡片索引，直到找到一个不在新牌组中的索引
        do {
            # 生成一个随机的卡片索引
            card = Math.floor(deckSize * Math.random());
        } while (deck.includes(card));  # 检查新生成的卡片索引是否已经在新牌组中，如果是则继续生成新的索引
        # 将新生成的卡片索引添加到新牌组中
        deck.push(card);
    }
    # 返回新牌组
    return deck;
// 计算卡片的值，每4张卡片对应一个值
function computeCardValue(cardIndex) {
    return Math.floor(cardIndex / 4);
}

// 打印游戏结束信息，包括玩家和电脑的最终得分
function printGameOver(playerScore, computerScore) {
    print("\n");
    print("\n");
    print(`WE HAVE RUN OUT OF CARDS.  FINAL SCORE:  YOU: ${playerScore}  THE COMPUTER: ${computerScore}\n`);
    print("\n");
}

// 打印游戏标题
function printTitle() {
    print(tab(33) + "WAR\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("THIS IS THE CARD GAME OF WAR.  EACH CARD IS GIVEN BY SUIT-#\n");
    print("AS S-7 FOR SPADE 7.  ");
}

// 打印玩家和电脑的卡片
function printCards(playerCard, computerCard) {
    print("\n");
    print(`YOU: ${playerCard}\tCOMPUTER: ${computerCard}\n`);
}

// 定义卡片的数组
const cards = [
    "S-2", "H-2", "C-2", "D-2",
    "S-3", "H-3", "C-3", "D-3",
    "S-4", "H-4", "C-4", "D-4",
    "S-5", "H-5", "C-5", "D-5",
    "S-6", "H-6", "C-6", "D-6",
    "S-7", "H-7", "C-7", "D-7",
    "S-8", "H-8", "C-8", "D-8",
    "S-9", "H-9", "C-9", "D-9",
    "S-10", "H-10", "C-10", "D-10",
    "S-J", "H-J", "C-J", "D-J",
    "S-Q", "H-Q", "C-Q", "D-Q",
    "S-K", "H-K", "C-K", "D-K",
    "S-A", "H-A", "C-A", "D-A"
];

// 主控制部分
async function main() {
    printTitle();  // 打印游戏标题
    await askAboutInstructions();  // 等待用户输入指令

    let computerScore = 0;  // 初始化电脑得分
    let playerScore = 0;  // 初始化玩家得分

    // 生成一个随机的牌堆
    const gameSize = cards.length;  // 牌堆中要洗入游戏的卡片数量，可以小于等于总卡片数量
    const deck = createGameDeck(cards, gameSize);  // 创建游戏牌堆
    let shouldContinuePlaying = true;  // 标记游戏是否继续进行
    # 当牌堆还有牌且游戏应该继续时执行循环
    while (deck.length > 0 && shouldContinuePlaying) {
        # 从牌堆中取出一张牌作为玩家的牌
        const playerCard = deck.shift();    // Take a card
        # 从牌堆中取出一张牌作为电脑的牌
        const computerCard = deck.shift();    // Take a card
        # 打印玩家和电脑的牌面
        printCards(cards[playerCard], cards[computerCard]);

        # 计算玩家牌的值
        const playerCardValue = computeCardValue(playerCard);
        # 计算电脑牌的值
        const computerCardValue = computeCardValue(computerCard);
        # 根据牌的值判断输赢，并更新分数
        if (playerCardValue < computerCardValue) {
            computerScore++;
            print("THE COMPUTER WINS!!! YOU HAVE " + playerScore + " AND THE COMPUTER HAS " + computerScore + "\n");
        } else if (playerCardValue > computerCardValue) {
            playerScore++;
            print("YOU WIN. YOU HAVE " + playerScore + " AND THE COMPUTER HAS " + computerScore + "\n");
        } else {
            print("TIE.  NO SCORE CHANGE.\n");
        }

        # 如果牌堆中没有牌了，打印游戏结束信息
        if (deck.length === 0) {
            printGameOver(playerScore, computerScore);
        }
        # 否则询问玩家是否继续游戏
        else {
            shouldContinuePlaying = await askYesOrNo("DO YOU WANT TO CONTINUE");
        }
    }
    # 打印游戏结束信息
    print("THANKS FOR PLAYING.  IT WAS FUN.\n");
    print("\n");
# 结束当前的函数定义
}

# 调用名为main的函数
main();
```
# `94_War\javascript\war.js`

```
// 定义一个名为print的函数，用于将字符串输出到页面上
// 参数str：要输出的字符串
function print(str) {
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义一个名为tab的函数，用于生成指定数量的空格字符串
// 参数space：要生成的空格数量
function tab(space) {
    let str = "";
    while (space-- > 0) {
        str += " ";
    }
    return str;
}

// 定义一个名为input的函数，返回一个Promise对象
function input() {
    return new Promise(function (resolve) {
        // 创建一个input元素
        const input_element = document.createElement("INPUT");
        # 打印问号
        print("? ");
        # 创建一个输入元素，并设置类型为文本，长度为50
        input_element.setAttribute("type", "text");
        input_element.setAttribute("length", "50");
        # 将输入元素添加到输出元素中
        document.getElementById("output").appendChild(input_element);
        # 让输入元素获得焦点
        input_element.focus();
        # 添加事件监听器，当按下回车键时执行相应操作
        input_element.addEventListener("keydown", function (event) {
            if (event.keyCode == 13) {
                # 获取输入的字符串
                const input_str = input_element.value;
                # 从输出元素中移除输入元素
                document.getElementById("output").removeChild(input_element);
                # 打印输入的字符串
                print(input_str);
                # 打印换行符
                print("\n");
                # 返回输入的字符串
                resolve(input_str);
            }
        });
    });
}

# 异步函数，用于询问是或否的问题
async function askYesOrNo(question) {
    # 循环直到得到有效的输入
    while (1) {
        print(question);  # 打印问题
        const str = (await input()).toUpperCase();  # 等待用户输入并将输入转换为大写
        if (str === "YES") {  # 如果输入是"YES"
            return true;  # 返回true
        }
        else if (str === "NO") {  # 如果输入是"NO"
            return false;  # 返回false
        }
        else {  # 如果输入既不是"YES"也不是"NO"
            print("YES OR NO, PLEASE.  ");  # 打印提示信息
        }
    }
}

async function askAboutInstructions() {
    const playerWantsInstructions = await askYesOrNo("DO YOU WANT DIRECTIONS");  # 等待用户输入是否需要游戏说明
    if (playerWantsInstructions) {  # 如果玩家需要游戏说明
        print("THE COMPUTER GIVES YOU AND IT A 'CARD'.  THE HIGHER CARD\n");  # 打印游戏说明
        print("(NUMERICALLY) WINS.  THE GAME ENDS WHEN YOU CHOOSE NOT TO\n");
        print("CONTINUE OR WHEN YOU HAVE FINISHED THE PACK.\n");
    }
    # 打印两个空行
    print("\n");
    print("\n");
}

# 创建游戏牌组
function createGameDeck(cards, gameSize) {
    # 创建一个空的牌组
    const deck = [];
    # 获取牌的总数
    const deckSize = cards.length;
    # 循环游戏所需的次数
    for (let j = 0; j < gameSize; j++) {
        let card;

        # 计算一个新的牌索引，直到找到一个不在新牌组中的牌
        do {
            card = Math.floor(deckSize * Math.random());
        } while (deck.includes(card));  # 如果新牌组中已经包含了这张牌，则继续计算新的牌索引
        deck.push(card);  # 将新的牌索引添加到新牌组中
    }
    return deck;  # 返回新牌组
}
# 计算卡片的值，通过卡片索引除以4取整得到
function computeCardValue(cardIndex) {
    return Math.floor(cardIndex / 4);
}

# 打印游戏结束信息，包括玩家分数和计算机分数
function printGameOver(playerScore, computerScore) {
    print("\n");  # 打印空行
    print("\n");  # 打印空行
    print(`WE HAVE RUN OUT OF CARDS.  FINAL SCORE:  YOU: ${playerScore}  THE COMPUTER: ${computerScore}\n`);  # 打印游戏结束信息和玩家、计算机的分数
    print("\n");  # 打印空行
}

# 打印游戏标题
function printTitle() {
    print(tab(33) + "WAR\n");  # 打印游戏标题
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  # 打印创意计算机的信息
    print("\n");  # 打印空行
    print("\n");  # 打印空行
    print("\n");  # 打印空行
    print("THIS IS THE CARD GAME OF WAR.  EACH CARD IS GIVEN BY SUIT-#\n");  # 打印游戏说明
    print("AS S-7 FOR SPADE 7.  ");  # 打印卡片的表示方式
}
function printCards(playerCard, computerCard) {
    // 打印玩家和电脑的卡牌
    print("\n");
    print(`YOU: ${playerCard}\tCOMPUTER: ${computerCard}\n`);
}

const cards = [
    // 一副扑克牌的所有牌的数组
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
// Main control section
async function main() {
    printTitle(); // 调用打印标题的函数

    await askAboutInstructions(); // 等待用户输入指令

    let computerScore = 0; // 初始化电脑得分
    let playerScore = 0; // 初始化玩家得分

    // Generate a random deck
    const gameSize = cards.length;    // 获取卡牌的数量作为游戏牌堆的大小
    const deck = createGameDeck(cards, gameSize); // 创建游戏牌堆
    let shouldContinuePlaying = true; // 初始化游戏继续标志

    while (deck.length > 0 && shouldContinuePlaying) { // 当牌堆不为空且游戏继续标志为真时执行循环
        const playerCard = deck.shift();    // 从牌堆中取出一张牌作为玩家的牌
        const computerCard = deck.shift();    // 从牌堆中取出一张牌作为电脑的牌
        printCards(cards[playerCard], cards[computerCard]); // 打印玩家和电脑的牌
        // 计算玩家卡牌的值
        const playerCardValue = computeCardValue(playerCard);
        // 计算电脑卡牌的值
        const computerCardValue = computeCardValue(computerCard);
        // 如果玩家卡牌值小于电脑卡牌值，电脑得分加一，并打印结果
        if (playerCardValue < computerCardValue) {
            computerScore++;
            print("THE COMPUTER WINS!!! YOU HAVE " + playerScore + " AND THE COMPUTER HAS " + computerScore + "\n");
        } 
        // 如果玩家卡牌值大于电脑卡牌值，玩家得分加一，并打印结果
        else if (playerCardValue > computerCardValue) {
            playerScore++;
            print("YOU WIN. YOU HAVE " + playerScore + " AND THE COMPUTER HAS " + computerScore + "\n");
        } 
        // 如果玩家卡牌值等于电脑卡牌值，打印平局
        else {
            print("TIE.  NO SCORE CHANGE.\n");
        }

        // 如果牌堆中没有剩余卡牌，打印游戏结束并显示最终得分
        if (deck.length === 0) {
            printGameOver(playerScore, computerScore);
        }
        // 否则询问玩家是否继续游戏
        else {
            shouldContinuePlaying = await askYesOrNo("DO YOU WANT TO CONTINUE");
        }
    }
    // 打印感谢信息
    print("THANKS FOR PLAYING.  IT WAS FUN.\n");
    print("\n");  # 打印一个换行符

}

main();  # 调用名为main的函数
```
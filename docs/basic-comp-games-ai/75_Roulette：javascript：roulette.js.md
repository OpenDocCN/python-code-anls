# `d:/src/tocomm/basic-computer-games\75_Roulette\javascript\roulette.js`

```
# 定义一个名为print的函数，用于将字符串输出到指定的HTML元素中
def print(str):
    document.getElementById("output").appendChild(document.createTextNode(str))

# 定义一个名为input的函数，用于获取用户输入的字符串
def input():
    var input_element
    var input_str

    # 返回一个Promise对象，用于处理异步操作
    return new Promise(function (resolve) {
                       # 创建一个input元素
                       input_element = document.createElement("INPUT")

                       # 在HTML元素中输出提示符
                       print("? ")

                       # 设置input元素的类型为文本
                       input_element.setAttribute("type", "text")
# 设置输入元素的长度为50
input_element.setAttribute("length", "50");
# 将输入元素添加到 id 为 "output" 的元素中
document.getElementById("output").appendChild(input_element);
# 让输入元素获得焦点
input_element.focus();
# 初始化输入字符串为未定义
input_str = undefined;
# 添加键盘按下事件监听器，当按下回车键时执行相应操作
input_element.addEventListener("keydown", function (event) {
    # 如果按下的键是回车键
    if (event.keyCode == 13) {
        # 将输入元素的值赋给输入字符串
        input_str = input_element.value;
        # 从 id 为 "output" 的元素中移除输入元素
        document.getElementById("output").removeChild(input_element);
        # 打印输入字符串
        print(input_str);
        # 打印换行符
        print("\n");
        # 解析输入字符串
        resolve(input_str);
    }
});
# 结束键盘按下事件监听器的添加
});
# 结束函数定义

# 定义一个名为 tab 的函数，参数为 space
function tab(space)
{
    # 初始化一个空字符串
    var str = "";
    # 当 space 大于 0 时执行循环
    while (space-- > 0)
```
        str += " ";  # 将空格字符添加到字符串末尾
    return str;  # 返回修改后的字符串

var ba = [];  # 创建一个空数组 ba
var ca = [];  # 创建一个空数组 ca
var ta = [];  # 创建一个空数组 ta
var xa = [];  # 创建一个空数组 xa
var aa = [];  # 创建一个空数组 aa

var numbers = [1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36];  # 创建一个包含数字的数组

// Main program  # 主程序开始
async function main()  # 异步函数 main 开始
{
    print(tab(32) + "ROULETTE\n");  # 打印带有制表符的字符串 "ROULETTE"
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  # 打印带有制表符的字符串 "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
    print("\n");  # 打印一个空行
    print("\n");  # 打印一个空行
    print("\n");  # 打印一个空行
}
    # 打印欢迎信息
    print("WELCOME TO THE ROULETTE TABLE\n")
    print("\n")
    # 询问是否需要游戏说明
    print("DO YOU WANT INSTRUCTIONS")
    str = await input()
    # 如果输入的第一个字符不是"N"，则打印下面的内容
    if (str.substr(0, 1) != "N"):
        print("\n")
        print("THIS IS THE BETTING LAYOUT\n")
        print("  (*=RED)\n")
        print("\n")
        # 打印赌注布局
        print(" 1*    2     3*\n")
        print(" 4     5*    6 \n")
        print(" 7*    8     9*\n")
        print("10    11    12*\n")
        print("---------------\n")
        print("13    14*   15 \n")
        print("16*   17    18*\n")
        print("19*   20    21*\n")
        print("22    23*   24 \n")
        # 打印分隔线
        print("---------------\n");
        # 打印特定的数字
        print("25*   26    27*\n");
        print("28    29    30*\n");
        print("31    32*   33 \n");
        print("34*   35    36*\n");
        # 打印分隔线
        print("---------------\n");
        # 打印特定的数字和文字
        print("    00    0    \n");
        print("\n");
        print("TYPES OF BETS\n");
        print("\n");
        # 打印赌注类型的说明
        print("THE NUMBERS 1 TO 36 SIGNIFY A STRAIGHT BET\n");
        print("ON THAT NUMBER.\n");
        print("THESE PAY OFF 35:1\n");
        print("\n");
        # 打印2:1赌注类型的说明
        print("THE 2:1 BETS ARE:\n");
        print(" 37) 1-12     40) FIRST COLUMN\n");
        print(" 38) 13-24    41) SECOND COLUMN\n");
        print(" 39) 25-36    42) THIRD COLUMN\n");
        print("\n");
        # 打印1:1赌注类型的说明
        print("THE EVEN MONEY BETS ARE:\n");
        # 打印赌注选项
        print(" 43) 1-18     46) ODD\n");
        print(" 44) 19-36    47) RED\n");
        print(" 45) EVEN     48) BLACK\n");
        print("\n");
        print(" 49)0 AND 50)00 PAY OFF 35:1\n");
        print(" NOTE: 0 AND 00 DO NOT COUNT UNDER ANY\n");
        print("       BETS EXCEPT THEIR OWN.\n");
        print("\n");
        print("WHEN I ASK FOR EACH BET, TYPE THE NUMBER\n");
        print("AND THE AMOUNT, SEPARATED BY A COMMA.\n");
        print("FOR EXAMPLE: TO BET $500 ON BLACK, TYPE 48,500\n");
        print("WHEN I ASK FOR A BET.\n");
        print("\n");
        print("THE MINIMUM BET IS $5, THE MAXIMUM IS $500.\n");
        print("\n");
    }
    // 程序从这里开始
    // 初始化赌注数组
    for (i = 1; i <= 100; i++) {
        ba[i] = 0;
        ca[i] = 0;  // 将 ca 数组的第 i 个元素赋值为 0
        ta[i] = 0;  // 将 ta 数组的第 i 个元素赋值为 0
    }
    for (i = 1; i <= 38; i++)
        xa[i] = 0;  // 将 xa 数组的第 i 个元素赋值为 0
    p = 1000;  // 将变量 p 赋值为 1000
    d = 100000;  // 将变量 d 赋值为 100000
    while (1) {  // 进入无限循环
        do {
            print("HOW MANY BETS");  // 打印提示信息
            y = parseInt(await input());  // 从输入中获取整数值并赋给变量 y
        } while (y < 1) ;  // 当 y 小于 1 时重复执行上述代码块
        for (i = 1; i <= 50; i++) {
            aa[i] = 0;  // 将 aa 数组的第 i 个元素赋值为 0
        }
        for (c = 1; c <= y; c++) {  // 循环 y 次
            while (1) {  // 进入无限循环
                print("NUMBER " + c + " ");  // 打印提示信息
                str = await input();  // 从输入中获取字符串值并赋给变量 str
                x = parseInt(str);  // 将字符串转换为整数并赋给变量 x
                # 将字符串中逗号后的部分转换为整数，并赋值给变量z
                z = parseInt(str.substr(str.indexOf(",") + 1));
                # 将z赋值给数组ba的第c个元素
                ba[c] = z;
                # 将x赋值给数组ta的第c个元素
                ta[c] = x;
                # 如果x小于1或大于50，则继续循环
                if (x < 1 || x > 50)
                    continue;
                # 如果z小于1，则继续循环
                if (z < 1)
                    continue;
                # 如果z小于5或大于500，则继续循环
                if (z < 5 || z > 500)
                    continue;
                # 如果数组aa中第x个元素不为0，则打印提示信息并继续循环
                if (aa[x] != 0) {
                    print("YOU MADE THAT BET ONCE ALREADY,DUM-DUM\n");
                    continue;
                }
                # 将数组aa中第x个元素赋值为1
                aa[x] = 1;
                # 跳出循环
                break;
            }
        }
        # 打印提示信息
        print("SPINNING\n");
        # 打印换行符
        print("\n");
        # 打印换行符
        print("\n");
        # 生成一个 1 到 100 之间的随机整数 s，直到 s 不等于 0 且不大于 38 为止
        do {
            s = Math.floor(Math.random() * 100);
        } while (s == 0 || s > 38) ;
        xa[s]++;    # 未使用
        # 如果 s 大于 37，则打印 "00"
        if (s > 37) {
            print("00\n");
        } 
        # 如果 s 等于 37，则打印 "0"
        else if (s == 37) {
            print("0\n");
        } 
        # 否则，遍历 numbers 列表，如果 s 等于 numbers 中的某个值，则打印 s 和 "RED"，否则打印 s 和 "BLACK"
        else {
            for (i1 = 1; i1 <= 18; i1++) {
                if (s == numbers[i1 - 1])
                    break;
            }
            if (i1 <= 18)
                print(s + " RED\n");
            else
                print(s + " BLACK\n");
        }
        # 打印一个空行
        print("\n");
        # 循环打印 y 次
        for (c = 1; c <= y; c++) {
            won = 0;  // 初始化赢得的筹码数量为0
            switch (ta[c]) {  // 根据ta数组中的值进行不同的情况判断
                case 37:    // 如果ta数组中的值为37，表示下注的是1-12范围内的数字，赔率为2:1
                    if (s > 12) {  // 如果开奖号码大于12
                        won = -ba[c];  // 则赢得的筹码数量为下注的筹码数量的负值
                    } else {
                        won = ba[c] * 2;  // 否则赢得的筹码数量为下注的筹码数量的两倍
                    }
                    break;  // 结束当前case的判断
                case 38:    // 如果ta数组中的值为38，表示下注的是13-24范围内的数字，赔率为2:1
                    if (s > 12 && s < 25) {  // 如果开奖号码大于12且小于25
                        won = ba[c] * 2;  // 则赢得的筹码数量为下注的筹码数量的两倍
                    } else {
                        won = -ba[c];  // 否则赢得的筹码数量为下注的筹码数量的负值
                    }
                    break;  // 结束当前case的判断
                case 39:    // 如果ta数组中的值为39，表示下注的是25-36范围内的数字，赔率为2:1
                    if (s > 24 && s < 37) {  // 如果开奖号码大于24且小于37
                        won = ba[c] * 2;  // 则赢得的筹码数量为下注的筹码数量的两倍
                    } else {
                    }
                    break;
                case 40:    // First column (40) 2:1
                    if (s < 37 && s % 3 == 1) {  # 如果 s 小于 37 并且 s 除以 3 的余数为 1
                        won = ba[c] * 2;  # 赢得的金额为当前下注金额的两倍
                    } else {
                        won = -ba[c];  # 输掉当前下注金额
                    }
                    break;
                case 41:    // Second column (41) 2:1
                    if (s < 37 && s % 3 == 2) {  # 如果 s 小于 37 并且 s 除以 3 的余数为 2
                        won = ba[c] * 2;  # 赢得的金额为当前下注金额的两倍
                    } else {
                        won = -ba[c];  # 输掉当前下注金额
                    }
                    break;
                case 42:    // Third column (42) 2:1
                    if (s < 37 && s % 3 == 0) {  # 如果 s 小于 37 并且 s 除以 3 的余数为 0
                        won = ba[c] * 2;  # 赢得的金额为当前下注金额的两倍
                    } else {
                        won = -ba[c];  # 输掉当前下注金额
                    }
                    } else {
                        won = -ba[c];
                    }
                    break;
```

这段代码是一个 switch 语句，根据不同的情况执行不同的操作。每个 case 后面的数字代表了不同的情况，注释中的数字表示了赌注的类型和赔率。在每个 case 下面是一个条件语句，根据条件判断赌注是否中奖，并给出相应的奖金。最后的 break 语句用于跳出 switch 语句。
                    } else {
                        won = -ba[c];
                    }
                    break;
                case 48:    // Black (48) 1:1
                    for (i = 19; i <= 36; i++) {
                        if (s == numbers[i - 1])
                            break;
                    }
                    if (i <= 36) {
                        won = ba[c];
                    } else {
                        won = -ba[c];
                    }
                    break;
                case 49:    // 1 to 18 (49) 1:1
                    if (s >= 1 && s <= 18) {
                        won = ba[c];
                    } else {
                        won = -ba[c];
                    }
                    break;
                case 50:    // 19 to 36 (50) 1:1
                    if (s >= 19 && s <= 36) {
                        won = ba[c];
                    } else {
                        won = -ba[c];
                    }
                    break;
                case 51:    // 1st 12 (51) 2:1
                    if (s >= 1 && s <= 12) {
                        won = ba[c] * 2;
                    } else {
                        won = -ba[c];
                    }
                    break;
                case 52:    // 2nd 12 (52) 2:1
                    if (s >= 13 && s <= 24) {
                        won = ba[c] * 2;
                    } else {
                        won = -ba[c];
                    }
                    break;
                case 53:    // 3rd 12 (53) 2:1
                    if (s >= 25 && s <= 36) {
                        won = ba[c] * 2;
                    } else {
                        won = -ba[c];
                    }
                    break;
                case 54:    // 1st column (54) 2:1
                    if (s % 3 == 1) {
                        won = ba[c] * 2;
                    } else {
                        won = -ba[c];
                    }
                    break;
                case 55:    // 2nd column (55) 2:1
                    if (s % 3 == 2) {
                        won = ba[c] * 2;
                    } else {
                        won = -ba[c];
                    }
                    break;
                case 56:    // 3rd column (56) 2:1
                    if (s % 3 == 0) {
                        won = ba[c] * 2;
                    } else {
                        won = -ba[c];
                    }
                    break;
                case 57:    // 1st dozen (57) 2:1
                    if (s >= 1 && s <= 12) {
                        won = ba[c] * 3;
                    } else {
                        won = -ba[c];
                    }
                    break;
                case 58:    // 2nd dozen (58) 2:1
                    if (s >= 13 && s <= 24) {
                        won = ba[c] * 3;
                    } else {
                        won = -ba[c];
                    }
                    break;
                case 59:    // 3rd dozen (59) 2:1
                    if (s >= 25 && s <= 36) {
                        won = ba[c] * 3;
                    } else {
                        won = -ba[c];
                    }
                    break;
                case 60:    // Even (60) 1:1
                    if (s % 2 == 0 && s != 0) {
                        won = ba[c];
                    } else {
                        won = -ba[c];
                    }
                    break;
                case 61:    // Odd (61) 1:1
                    if (s % 2 != 0) {
                        won = ba[c];
                    } else {
                        won = -ba[c];
                    }
                    break;
                case 62:    // Red (62) 1:1
                    for (i = 1; i <= 18; i++) {
                        if (s == numbers[i - 1])
                            break;
                    }
                    if (i <= 18) {
                        won = ba[c];
                    } else {
                        won = -ba[c];
                    }
                    break;
                case 63:    // Black (63) 1:1
                    for (i = 19; i <= 36; i++) {
                        if (s == numbers[i - 1])
                            break;
                    }
                    if (i <= 36) {
                        won = ba[c];
                    } else {
                        won = -ba[c];
                    }
                    break;
                default:
                    won = -ba[c];
                    break;
            }
        }
    }
    return won;
}
                    won = -ba[c];
                    }
                    break;
```

解释：

- `won = -ba[c];`：将变量`won`赋值为`-ba[c]`
- `break;`：跳出当前循环或者`switch`语句的执行

```python
                case 48:    // Black (48) 1:1
                    for (i = 1; i <= 18; i++) {
                        if (s == numbers[i - 1])
                            break;
                    }
                    if (i <= 18 || s > 36) {
                        won = -ba[c];
                    } else {
                        won = ba[c];
                    }
                    break;
```

解释：

- `case 48:`：`switch`语句的一个分支，表示当`switch`表达式的值等于48时执行以下代码
- `for (i = 1; i <= 18; i++) { if (s == numbers[i - 1]) break; }`：循环遍历`numbers`数组，如果找到`s`的值，则跳出循环
- `if (i <= 18 || s > 36) { won = -ba[c]; } else { won = ba[c]; }`：如果`i`小于等于18或者`s`大于36，则`won`赋值为`-ba[c]`，否则赋值为`ba[c]`
- `break;`：跳出`switch`语句的执行

```python
                default:    // 1-36,0,00 (1-36,49,50) 35:1
                    if (ta[c] < 49 && ta[c] == s
                        || ta[c] == 49 && s == 37
                        || ta[c] == 50 && s == 38) {
                        won = ba[c] * 35;
                    } else {
                        won = -ba[c];
                    }
                    break;
```

解释：

- `default:`：`switch`语句的默认分支，表示当`switch`表达式的值不匹配任何`case`时执行以下代码
- `if (ta[c] < 49 && ta[c] == s || ta[c] == 49 && s == 37 || ta[c] == 50 && s == 38) { won = ba[c] * 35; } else { won = -ba[c]; }`：根据条件判断，如果满足条件则`won`赋值为`ba[c] * 35`，否则赋值为`-ba[c]`
- `break;`：跳出`switch`语句的执行
                    won = -ba[c];  # 从玩家的赌注中减去庄家的赌注
                }
                break;  # 退出循环
        }
        d -= won;  # 玩家的赌注减去赢得的赌注
        p += won;  # 玩家的总金额增加赢得的赌注
        if (won < 0):  # 如果赢得的赌注小于0
            print("YOU LOSE " + -won + " DOLLARS ON BET " + c + "\n");  # 打印玩家输掉的赌注和赌注编号
        else:  # 否则
            print("YOU WIN " + won + " DOLLARS ON BET " + c + "\n");  # 打印玩家赢得的赌注和赌注编号
    }
    print("\n");  # 打印空行
    print("TOTALS:\tME\tYOU\n");  # 打印总计表头
    print(" \t" + d + "\t" + p + "\n");  # 打印玩家和庄家的总金额
    if (p <= 0):  # 如果玩家的总金额小于等于0
        print("OOPS! YOU JUST SPENT YOUR LAST DOLLAR!\n");  # 打印玩家已经花光最后一美元
        break;  # 退出循环
    else if (d <= 0):  # 否则如果庄家的总金额小于等于0
        print("YOU BROKE THE HOUSE!\n");  # 打印庄家破产
# 设置变量 p 的值为 101000
p = 101000;
# 打印 "AGAIN"
print("AGAIN");
# 等待用户输入，并将输入的字符串赋值给变量 str
str = await input();
# 如果输入字符串的第一个字符不是 "Y"，则跳出循环
if (str.substr(0, 1) != "Y")
    break;
# 结束循环

# 如果变量 p 的值小于 1
if (p < 1) {
    # 打印 "THANKS FOR YOUR MONEY."
    print("THANKS FOR YOUR MONEY.\n");
    # 打印 "I'LL USE IT TO BUY A SOLID GOLD ROULETTE WHEEL"
    print("I'LL USE IT TO BUY A SOLID GOLD ROULETTE WHEEL\n");
# 否则
} else {
    # 打印 "TO WHOM SHALL I MAKE THE CHECK"
    print("TO WHOM SHALL I MAKE THE CHECK");
    # 等待用户输入，并将输入的字符串赋值给变量 str
    str = await input();
    # 打印换行符
    print("\n");
    # 打印 72 个连字符
    for (i = 1; i <= 72; i++)
        print("-");
    # 打印换行符
    print("\n");
    # 打印 tab(50) + "CHECK NO. " + 0 到 100 之间的随机整数
    print(tab(50) + "CHECK NO. " + Math.floor(Math.random() * 100) + "\n");
    # 打印换行符
    print("\n");
    # 打印 tab(40) + 当前日期的字符串表示形式
    print(tab(40) + new Date().toDateString());
        print("\n");  # 打印空行
        print("\n");  # 打印空行
        print("PAY TO THE ORDER OF-----" + str + "-----$ " + p + "\n");  # 打印付款信息
        print("\n");  # 打印空行
        print("\n");  # 打印空行
        print(tab(10) + "\tTHE MEMORY BANK OF NEW YORK\n");  # 打印银行信息
        print("\n");  # 打印空行
        print(tab(40) + "\tTHE COMPUTER\n");  # 打印计算机信息
        print(tab(40) + "----------X-----\n");  # 打印分隔线
        print("\n");  # 打印空行
        for (i = 1; i <= 72; i++)  # 循环打印横线
            print("-");
        print("\n");  # 打印换行
        print("COME BACK SOON!\n");  # 打印提示信息
    }
    print("\n");  # 打印空行
}

main();  # 调用主函数
```
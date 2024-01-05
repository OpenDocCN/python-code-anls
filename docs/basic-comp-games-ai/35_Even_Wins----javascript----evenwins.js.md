# `35_Even_Wins\javascript\evenwins.js`

```
// 创建一个新的 Promise 对象，用于处理输入操作
// 创建一个 INPUT 元素，用于用户输入
// 在页面上显示提示符 "? "
// 设置 INPUT 元素的类型为文本输入
# 设置输入元素的长度为50
input_element.setAttribute("length", "50");
# 将输入元素添加到 id 为 "output" 的元素中
document.getElementById("output").appendChild(input_element);
# 让输入元素获得焦点
input_element.focus();
# 初始化输入字符串为 undefined
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
        # 解析并返回输入字符串
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
    # 当 space 大于 0 时，执行循环
    while (space-- > 0)
        str += " ";  # 将空格添加到字符串末尾
    return str;  # 返回处理后的字符串

var ma = [];  # 创建一个空数组ma
var ya = [];  # 创建一个空数组ya

// Main program  # 主程序开始
async function main()  # 异步函数main开始
{
    print(tab(31) + "EVEN WINS\n");  # 打印带有31个空格的字符串" EVEN WINS"
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  # 打印带有15个空格的字符串"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
    print("\n");  # 打印一个空行
    print("\n");  # 打印一个空行
    y1 = 0;  # 初始化变量y1为0
    m1 = 0;  # 初始化变量m1为0
    print("     THIS IS A TWO PERSON GAME CALLED 'EVEN WINS.'\n");  # 打印游戏介绍信息
    print("TO PLAY THE GAME, THE PLAYERS NEED 27 MARBLES OR\n");  # 打印游戏玩法信息
    print("OTHER OBJECTS ON A TABLE.\n");  # 打印游戏玩法信息
    print("\n");  # 打印一个空行
    # 打印空行
    print("\n");
    # 打印游戏规则说明
    print("     THE 2 PLAYERS ALTERNATE TURNS, WITH EACH PLAYER\n");
    print("REMOVING FROM 1 TO 4 MARBLES ON EACH MOVE.  THE GAME\n");
    print("ENDS WHEN THERE ARE NO MARBLES LEFT, AND THE WINNER\n");
    print("IS THE ONE WITH AN EVEN NUMBER OF MARBLES.\n");
    print("\n");
    print("\n");
    # 打印游戏规则说明
    print("     THE ONLY RULES ARE THAT (1) YOU MUST ALTERNATE TURNS,\n");
    print("(2) YOU MUST TAKE BETWEEN 1 AND 4 MARBLES EACH TURN,\n");
    print("AND (3) YOU CANNOT SKIP A TURN.\n");
    print("\n");
    print("\n");
    print("\n");
    # 进入游戏循环
    while (1) {
        # 提示玩家选择先手或后手
        print("     TYPE A '1' IF YOU WANT TO GO FIRST, AND TYPE\n");
        print("A '0' IF YOU WANT ME TO GO FIRST.\n");
        # 获取玩家选择
        c = parseInt(await input());
        print("\n");
        # 如果玩家选择先手
        if (c != 0) {
            # 设定初始的总数目为27
            t = 27;
            # 打印空行
            print("\n");
            print("\n");
            print("\n");
            # 打印总数
            print("TOTAL= " + t + "\n");
            print("\n");
            print("\n");
            # 打印提示信息
            print("WHAT IS YOUR FIRST MOVE");
            # 初始化变量 m
            m = 0;
        } else {
            # 设置总数为 27
            t = 27;
            # 设置 m 为 2
            m = 2;
            # 打印空行
            print("\n");
            # 打印总数
            print("TOTAL= " + t + "\n");
            print("\n");
            # 将 m 加到 m1 上
            m1 += m;
            # 从总数中减去 m
            t -= m;
        }
        # 进入循环
        while (1) {
            # 如果 m 不为 0
            if (m) {
                # 打印拾起的数量
                print("I PICK UP " + m + " MARBLES.\n");
                if (t == 0)  # 如果 t 的值为 0，则跳出循环
                    break;
                print("\n");  # 打印空行
                print("TOTAL= " + t + "\n");  # 打印总数 t 的值
                print("\n");  # 打印空行
                print("     AND WHAT IS YOUR NEXT MOVE, MY TOTAL IS " + m1 + "\n");  # 打印提示信息和 m1 的值
            }
            while (1) {  # 进入无限循环
                y = parseInt(await input());  # 从输入中获取一个整数值赋给变量 y
                print("\n");  # 打印空行
                if (y < 1 || y > 4) {  # 如果 y 的值小于 1 或大于 4
                    print("\n");  # 打印空行
                    print("THE NUMBER OF MARBLES YOU MUST TAKE BE A POSITIVE\n");  # 打印提示信息
                    print("INTEGER BETWEEN 1 AND 4.\n");  # 打印提示信息
                    print("\n");  # 打印空行
                    print("     WHAT IS YOUR NEXT MOVE?\n");  # 打印提示信息
                    print("\n");  # 打印空行
                } else if (y > t) {  # 如果 y 的值大于 t
                    print("     YOU HAVE TRIED TO TAKE MORE MARBLES THAN THERE ARE\n");  # 打印提示信息
                    print("LEFT.  TRY AGAIN.\n");  # 打印提示信息
            } else {
                break;  # 如果条件不满足，则跳出循环
            }
        }

        y1 += y;  # 将 y1 值增加 y
        t -= y;   # 将 t 减去 y
        if (t == 0):  # 如果 t 等于 0
            break;  # 跳出循环
        print("TOTAL= " + t + "\n");  # 打印 t 的值
        print("\n");  # 打印空行
        print("YOUR TOTAL IS " + y1 + "\n");  # 打印 y1 的值
        if (t < 0.5):  # 如果 t 小于 0.5
            break;  # 跳出循环
        r = t % 6;  # 计算 t 除以 6 的余数
        if (y1 % 2 != 0):  # 如果 y1 除以 2 的余数不等于 0
            if (t >= 4.2):  # 如果 t 大于等于 4.2
                if (r <= 3.4):  # 如果 r 小于等于 3.4
                    m = r + 1;  # 计算 m 的值为 r 加 1
                    m1 += m;  # 将 m 加到 m1 上
                    t -= m;  # 从 t 中减去 m 的值
                    } else if (r < 4.7 || r > 3.5) {  # 如果 r 的值小于 4.7 或者大于 3.5
                        m = 4;  # 将 m 的值设为 4
                        m1 += m;  # 将 m 加到 m1 上
                        t -= m;  # 从 t 中减去 m 的值
                    } else {  # 否则
                        m = 1;  # 将 m 的值设为 1
                        m1 += m;  # 将 m 加到 m1 上
                        t -= m;  # 从 t 中减去 m 的值
                    }
                } else {  # 如果不满足上述条件
                    m = t;  # 将 m 的值设为 t
                    t -= m;  # 从 t 中减去 m 的值
                    print("I PICK UP " + m + " MARBLES.\n");  # 打印出 "I PICK UP " + m + " MARBLES."
                    print("\n");  # 打印一个空行
                    print("TOTAL = 0\n");  # 打印 "TOTAL = 0"
                    m1 += m;  # 将 m 加到 m1 上
                    break;  # 跳出循环
                }
            } else {  # 如果不满足上述条件
# 如果r小于1.5或者大于5.3
if (r < 1.5 || r > 5.3) {
    # 将m设为1
    m = 1;
    # 将m加到m1上
    m1 += m;
    # 从t中减去m
    t -= m;
} else {
    # 将m设为r减1
    m = r - 1;
    # 将m加到m1上
    m1 += m;
    # 从t中减去m
    t -= m;
    # 如果t小于0.2
    if (t < 0.2) {
        # 打印"I PICK UP " + m + " MARBLES."
        print("I PICK UP " + m + " MARBLES.\n");
        # 打印换行符
        print("\n");
        # 跳出循环
        break;
    }
}
# 打印"THAT IS ALL OF THE MARBLES."
print("THAT IS ALL OF THE MARBLES.\n");
# 打印换行符
print("\n");
# 打印" MY TOTAL IS " + m1 + ", YOUR TOTAL IS " + y1 +"\n"
print(" MY TOTAL IS " + m1 + ", YOUR TOTAL IS " + y1 +"\n");
# 打印换行符
print("\n");
        # 如果m1除以2的余数不等于0，打印“YOU WON.  DO YOU WANT TO PLAY”
        if (m1 % 2 != 0) {
            print("     YOU WON.  DO YOU WANT TO PLAY\n");
        } else {
            # 否则，打印“I WON.  DO YOU WANT TO PLAY”
            print("     I WON.  DO YOU WANT TO PLAY\n");
        }
        # 打印“AGAIN?  TYPE 1 FOR YES AND 0 FOR NO.”
        print("AGAIN?  TYPE 1 FOR YES AND 0 FOR NO.\n");
        # 将用户输入的值转换为整数并赋给a1
        a1 = parseInt(await input());
        # 如果a1等于0，跳出循环
        if (a1 == 0)
            break;
        # 重置m1和y1的值为0
        m1 = 0;
        y1 = 0;
    }
    # 打印空行
    print("\n");
    # 打印“OK.  SEE YOU LATER”
    print("OK.  SEE YOU LATER\n");
}

main();
```
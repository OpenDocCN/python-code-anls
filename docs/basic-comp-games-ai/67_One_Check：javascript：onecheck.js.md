# `d:/src/tocomm/basic-computer-games\67_One_Check\javascript\onecheck.js`

```
// 创建一个名为print的函数，用于在页面上输出文本
// 创建一个名为input的函数，用于获取用户输入的文本
// 使用Promise对象来异步获取用户输入
// 创建一个input元素，并设置其类型为文本
// 将input元素添加到页面上，并显示提示符“? ”
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
}

# 定义一个函数，用于生成指定数量的空格
function tab(space)
{
    var str = "";
    # 当 space 大于 0 时，循环添加空格到 str 中
    while (space-- > 0)
        str += " ";  // 将空格添加到字符串末尾
    return str;  // 返回处理后的字符串

var a = [];  // 创建一个空数组

// Main program
async function main()
{
    print(tab(30) + "ONE CHECK\n");  // 在控制台打印带有缩进的字符串
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  // 在控制台打印带有缩进的字符串
    print("\n");  // 在控制台打印空行
    print("\n");  // 在控制台打印空行
    print("\n");  // 在控制台打印空行
    for (i = 0; i <= 64; i++)  // 循环，将数组a的前65个元素初始化为0
        a[i] = 0;
    print("SOLITAIRE CHECKER PUZZLE BY DAVID AHL\n");  // 在控制台打印字符串
    print("\n");  // 在控制台打印空行
    print("48 CHECKERS ARE PLACED ON THE 2 OUTSIDE SPACES OF A\n");  // 在控制台打印字符串
    print("STANDARD 64-SQUARE CHECKERBOARD.  THE OBJECT IS TO\n");  // 在控制台打印字符串
    # 打印游戏规则和提示信息
    print("REMOVE AS MANY CHECKERS AS POSSIBLE BY DIAGONAL JUMPS\n");
    print("(AS IN STANDARD CHECKERS).  USE THE NUMBERED BOARD TO\n");
    print("INDICATE THE SQUARE YOU WISH TO JUMP FROM AND TO.  ON\n");
    print("THE BOARD PRINTED OUT ON EACH TURN '1' INDICATES A\n");
    print("CHECKER AND '0' AN EMPTY SQUARE.  WHEN YOU HAVE NO\n");
    print("POSSIBLE JUMPS REMAINING, INPUT A '0' IN RESPONSE TO\n");
    print("QUESTION 'JUMP FROM ?'\n");
    print("\n");
    print("HERE IS THE NUMERICAL BOARD:\n");
    print("\n");
    # 进入游戏循环
    while (1) {
        # 打印数字棋盘
        for (j = 1; j <= 57; j += 8) {
            str = "";
            for (i = 0; i <= 7; i++) {
                # 确保每个数字占据4个字符的位置
                while (str.length < 4 * i)
                    str += " ";
                # 添加数字到字符串
                str += " " + (j + i);
            }
            # 打印每一行的数字
            print(str + "\n");
        }
        print("\n");  # 打印空行
        print("AND HERE IS THE OPENING POSITION OF THE CHECKERS.\n");  # 打印提示信息
        print("\n");  # 打印空行
        for (j = 1; j <= 64; j++)  # 循环遍历1到64
            a[j] = 1;  # 将数组a的每个元素初始化为1
        for (j = 19; j <= 43; j += 8)  # 循环遍历19到43，步长为8
            for (i = j; i <= j + 3; i++)  # 嵌套循环，遍历j到j+3
                a[i] = 0;  # 将数组a的指定元素初始化为0
        m = 0;  # 初始化变量m为0
        while (1) {  # 进入无限循环
            // Print board  # 打印棋盘
            for (j = 1; j <= 57; j += 8) {  # 循环遍历1到57，步长为8
                str = "";  # 初始化字符串str为空
                for (i = j; i <= j + 7; i++) {  # 嵌套循环，遍历j到j+7
                    str += " " + a[i] + " ";  # 将数组a的元素拼接成字符串
                }
                print(str + "\n");  # 打印字符串并换行
            }
            print("\n");  # 打印空行
            while (1) {  # 进入内部无限循环
                // 打印提示信息，要求输入起始位置
                print("JUMP FROM");
                // 读取用户输入的起始位置
                f = parseInt(await input());
                // 如果用户输入的起始位置为0，则跳出循环
                if (f == 0)
                    break;
                // 打印提示信息，要求输入目标位置
                print("TO");
                // 读取用户输入的目标位置
                t = parseInt(await input());
                // 打印换行符
                print("\n");
                // 检查移动的合法性
                f1 = Math.floor((f - 1) / 8);
                f2 = f - 8 * f1;
                t1 = Math.floor((t - 1) / 8);
                t2 = t - 8 * t1;
                // 如果起始位置或目标位置超出棋盘范围，或者移动不符合规则，或者中间位置没有对方棋子，或者起始位置没有自己的棋子，则打印提示信息并继续循环
                if (f1 > 7 || t1 > 7 || f2 > 8 || t2 > 8 || Math.abs(f1 - t1) != 2 || Math.abs(f2 - t2) != 2 || a[(t + f) / 2] == 0 || a[f] == 0 || a[t] == 1) {
                    print("ILLEGAL MOVE.  TRY AGAIN...\n");
                    continue;
                }
                // 跳出循环
                break;
            }
            // 如果用户输入的起始位置为0，则跳出循环
            if (f == 0)
                break;
            // 更新棋盘
            a[t] = 1; // 将目标位置设为1，表示有棋子
            a[f] = 0; // 将起始位置设为0，表示没有棋子
            a[(t + f) / 2] = 0; // 将跳跃位置设为0，表示跳过的棋子被移除
            m++; // 移动次数加一
        }
        // 游戏结束总结
        s = 0; // 初始化棋盘上棋子数量
        for (i = 1; i <= 64; i++)
            s += a[i]; // 统计棋盘上的棋子数量
        print("\n");
        print("YOU MADE " + m + " JUMPS AND HAD " + s + " PIECES\n"); // 打印移动次数和剩余棋子数量
        print("REMAINING ON THE BOARD.\n"); // 打印剩余在棋盘上的棋子数量
        print("\n");
        while (1) { // 无限循环，直到满足条件跳出
            print("TRY AGAIN"); // 提示玩家再试一次
            str = await input(); // 等待玩家输入
            if (str == "YES") // 如果玩家输入YES
                break; // 跳出循环
            if (str == "NO") // 如果玩家输入NO
                break;  # 结束当前循环，跳出循环体
            print("PLEASE ANSWER 'YES' OR 'NO'.\n");  # 打印提示信息，要求输入'YES'或'NO'
        }
        if (str == "NO")  # 如果输入的字符串是'NO'
            break;  # 结束当前循环，跳出循环体
    }
    print("\n");  # 打印空行
    print("O.K.  HOPE YOU HAD FUN!!\n");  # 打印提示信息，表示程序结束
}

main();  # 调用主函数
```
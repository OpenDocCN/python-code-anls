# `basic-computer-games\26_Chomp\javascript\chomp.js`

```
// CHOMP 游戏的Javascript版本，由Oscar Toledo G. (nanochess) 从BASIC转换而来

// 打印输出到页面
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 获取用户输入
function input()
{
    var input_element;
    var input_str;

    return new Promise(function (resolve) {
                       input_element = document.createElement("INPUT");

                       // 打印提示符
                       print("? ");
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       document.getElementById("output").appendChild(input_element);
                       input_element.focus();
                       input_str = undefined;
                       // 监听用户输入
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      input_str = input_element.value;
                                                      document.getElementById("output").removeChild(input_element);
                                                      print(input_str);
                                                      print("\n");
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

// 生成指定数量的空格字符串
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 初始化游戏板
var a = [];
var r;
var c;
function init_board()
{
    for (i = 1; i <= r; i++)
        for (j = 1; j <= c; j++)
            a[i][j] = 1;
    a[1][1] = -1;
}

// 显示游戏板
function show_board()
{
    print("\n");
    print(tab(7) + "1 2 3 4 5 6 7 8 9\n");
    # 遍历行数范围内的每一行
    for (i = 1; i <= r; i++) {
        # 将当前行数和6个空格组成字符串
        str = i + tab(6);
        # 遍历列数范围内的每一列
        for (j = 1; j <= c; j++) {
            # 如果当前位置的值为-1，则在字符串后添加"P "
            if (a[i][j] == -1)
                str += "P ";
            # 如果当前位置的值为0，则跳出当前循环
            else if (a[i][j] == 0)
                break;
            # 否则在字符串后添加"* "
            else
                str += "* ";
        }
        # 打印当前行的字符串并换行
        print(str + "\n");
    }
    # 打印空行
    print("\n");
// 主程序
async function main()
{
    // 打印 CHOMP
    print(tab(33) + "CHOMP\n");
    // 打印 CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    // 打印三个空行
    print("\n");
    print("\n");
    print("\n");
    // 初始化数组 a 的元素为数组
    for (i = 1; i <= 10; i++)
        a[i] = [];
    // *** THE GAME OF CHOMP *** COPYRIGHT PCC 1973 ***
    // 打印游戏信息
    print("\n");
    print("THIS IS THE GAME OF CHOMP (SCIENTIFIC AMERICAN, JAN 1973)\n");
    // 询问是否需要游戏规则
    print("DO YOU WANT THE RULES (1=YES, 0=NO!)");
    // 将输入转换为整数
    r = parseInt(await input());
    if (r != 0) {
        // 设置 f 为 1
        f = 1;
        // 设置 r 为 5
        r = 5;
        // 设置 c 为 7
        c = 7;
        // 打印游戏规则
        print("CHOMP IS FOR 1 OR MORE PLAYERS (HUMANS ONLY).\n");
        print("\n");
        print("HERE'S HOW A BOARD LOOKS (THIS ONE IS 5 BY 7):\n");
        // 初始化游戏板
        init_board();
        // 显示游戏板
        show_board();
        print("\n");
        print("THE BOARD IS A BIG COOKIE - R ROWS HIGH AND C COLUMNS\n");
        print("WIDE. YOU INPUT R AND C AT THE START. IN THE UPPER LEFT\n");
        print("CORNER OF THE COOKIE IS A POISON SQUARE (P). THE ONE WHO\n");
        print("CHOMPS THE POISON SQUARE LOSES. TO TAKE A CHOMP, TYPE THE\n");
        print("ROW AND COLUMN OF ONE OF THE SQUARES ON THE COOKIE.\n");
        print("ALL OF THE SQUARES BELOW AND TO THE RIGHT OF THAT SQUARE\n");
        print("INCLUDING THAT SQUARE, TOO) DISAPPEAR -- CHOMP!!\n");
        print("NO FAIR CHOMPING SQUARES THAT HAVE ALREADY BEEN CHOMPED,\n");
        print("OR THAT ARE OUTSIDE THE ORIGINAL DIMENSIONS OF THE COOKIE.\n");
        print("\n");
    }
}
    # 进入无限循环
    while (1) {
        # 打印提示信息
        print("HERE WE GO...\n");
        # 初始化变量 f 为 0
        f = 0;
        # 循环遍历 i 从 1 到 10
        for (i = 1; i <= 10; i++) {
            # 初始化 a[i] 为一个空数组
            a[i] = [];
            # 循环遍历 j 从 1 到 10
            for (j = 1; j <= 10; j++) {
                # 初始化 a[i][j] 为 0
                a[i][j] = 0;
            }
        }
        # 打印换行符
        print("\n");
        # 打印提示信息
        print("HOW MANY PLAYERS");
        # 将输入转换为整数并赋值给变量 p
        p = parseInt(await input());
        # 初始化变量 i1 为 0
        i1 = 0;
        # 进入无限循环
        while (1) {
            # 打印提示信息
            print("HOW MANY ROWS");
            # 将输入转换为整数并赋值给变量 r
            r = parseInt(await input());
            # 如果 r 小于等于 9，则跳出循环
            if (r <= 9)
                break;
            # 打印提示信息
            print("TOO MANY ROWS (9 IS MAXIMUM). NOW ");
        }
        # 进入无限循环
        while (1) {
            # 打印提示信息
            print("HOW MANY COLUMNS");
            # 将输入转换为整数并赋值给变量 c
            c = parseInt(await input());
            # 如果 c 小于等于 9，则跳出循环
            if (c <= 9)
                break;
            # 打印提示信息
            print("TOO MANY COLUMNS (9 IS MAXIMUM). NOW ");
        }
        # 打印换行符
        print("\n");
        # 调用初始化棋盘的函数
        init_board();
        # 进入无限循环
        while (1) {
            # 打印当前棋盘状态
            show_board();
            # 递增变量 i1
            i1++;
            # 计算当前玩家的编号
            p1 = i1 - Math.floor(i1 / p) * p;
            if (p1 == 0)
                p1 = p;
            # 进入无限循环
            while (1) {
                # 打印提示信息
                print("PLAYER " + p1 + "\n");
                print("COORDINATES OF CHOMP (ROW,COLUMN)");
                # 获取玩家输入的坐标
                str = await input();
                r1 = parseInt(str);
                c1 = parseInt(str.substr(str.indexOf(",") + 1));
                # 如果输入的坐标合法，则跳出循环
                if (r1 >= 1 && r1 <= r && c1 >= 1 && c1 <= c && a[r1][c1] != 0)
                    break;
                # 打印提示信息
                print("NO FAIR. YOU'RE TRYING TO CHOMP ON EMPTY SPACE!\n");
            }
            # 如果玩家选择退出游戏，则跳出循环
            if (a[r1][c1] == -1)
                break;
            # 从输入的坐标开始，将对应区域的数值置为 0
            for (i = r1; i <= r; i++)
                for (j = c1; j <= c; j++)
                    a[i][j] = 0;
        }
        # 棋局结束，打印提示信息
        print("YOU LOSE, PLAYER " + p1 + "\n");
        # 打印换行符
        print("\n");
        # 打印提示信息
        print("AGAIN (1=YES, 0=NO!)");
        # 将输入转换为整数并赋值给变量 r
        r = parseInt(await input());
        # 如果输入不为 1，则跳出循环
        if (r != 1)
            break;
    }
# 结束 main 函数的定义
}

# 调用 main 函数
main();
```
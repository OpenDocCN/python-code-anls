# `26_Chomp\javascript\chomp.js`

```
# 定义函数print，用于在页面上输出字符串
def print(str):
    document.getElementById("output").appendChild(document.createTextNode(str))

# 定义函数input，用于获取用户输入
def input():
    # 声明变量
    var input_element
    var input_str

    # 返回一个Promise对象，用于异步处理用户输入
    return new Promise(function (resolve) {
        # 创建一个input元素
        input_element = document.createElement("INPUT')
        # 在页面上输出提示符
        print("? ")
        # 设置input元素的类型为文本
        input_element.setAttribute("type", "text")
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
    # 当 space 大于 0 时执行循环
    while (space-- > 0)
```
        str += " ";  # 将空格字符添加到字符串变量str的末尾
    return str;  # 返回处理后的字符串

var a = [];  # 声明一个空数组变量a
var r;  # 声明一个变量r
var c;  # 声明一个变量c

function init_board()  # 定义一个名为init_board的函数
{
    for (i = 1; i <= r; i++)  # 循环变量i从1到r
        for (j = 1; j <= c; j++)  # 循环变量j从1到c
            a[i][j] = 1;  # 将数组a的第i行第j列的元素赋值为1
    a[1][1] = -1;  # 将数组a的第1行第1列的元素赋值为-1
}

function show_board()  # 定义一个名为show_board的函数
{
    print("\n");  # 打印换行符
    print(tab(7) + "1 2 3 4 5 6 7 8 9\n");  # 打印制表符和数字1到9
    for (i = 1; i <= r; i++) {  // 循环遍历行
        str = i + tab(6);  // 将当前行数和6个空格拼接成字符串
        for (j = 1; j <= c; j++) {  // 循环遍历列
            if (a[i][j] == -1)  // 如果当前位置的值为-1
                str += "P ";  // 在字符串后面添加"P "
            else if (a[i][j] == 0)  // 如果当前位置的值为0
                break;  // 跳出内层循环
            else
                str += "* ";  // 在字符串后面添加"* "
        }
        print(str + "\n");  // 打印当前行的字符串并换行
    }
    print("\n");  // 打印空行
}

// Main program
async function main()
{
    print(tab(33) + "CHOMP\n");  // 打印"CHOMP"，并在前面添加33个空格
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  // 打印"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"，并在前面添加15个空格
}
    print("\n");  # 打印空行
    print("\n");  # 打印空行
    print("\n");  # 打印空行
    for (i = 1; i <= 10; i++)  # 循环10次，初始化数组a的每个元素为一个空数组
        a[i] = [];
    // *** THE GAME OF CHOMP *** COPYRIGHT PCC 1973 ***  # 打印游戏标题和版权信息
    print("\n");  # 打印空行
    print("THIS IS THE GAME OF CHOMP (SCIENTIFIC AMERICAN, JAN 1973)\n");  # 打印游戏介绍
    print("DO YOU WANT THE RULES (1=YES, 0=NO!)");  # 打印提示信息，询问是否需要游戏规则
    r = parseInt(await input());  # 获取用户输入并转换为整数赋值给变量r
    if (r != 0) {  # 如果r不等于0
        f = 1;  # 设置变量f为1
        r = 5;  # 设置变量r为5
        c = 7;  # 设置变量c为7
        print("CHOMP IS FOR 1 OR MORE PLAYERS (HUMANS ONLY).\n");  # 打印游戏玩家信息
        print("\n");  # 打印空行
        print("HERE'S HOW A BOARD LOOKS (THIS ONE IS 5 BY 7):\n");  # 打印游戏棋盘大小信息
        init_board();  # 调用初始化棋盘的函数
        show_board();  # 调用展示棋盘的函数
        print("\n");  # 打印空行
        # 打印游戏规则说明
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
    # 进入游戏循环
    while (1) {
        # 打印游戏开始提示
        print("HERE WE GO...\n");
        # 初始化变量f为0
        f = 0;
        # 初始化二维数组a，大小为10x10
        for (i = 1; i <= 10; i++) {
            a[i] = [];
            for (j = 1; j <= 10; j++) {
                a[i][j] = 0;
            }
        }
        print("\n");  # 打印空行
        print("HOW MANY PLAYERS");  # 打印提示信息，询问玩家数量
        p = parseInt(await input());  # 获取玩家数量输入并转换为整数赋值给变量p
        i1 = 0;  # 初始化变量i1为0
        while (1):  # 进入无限循环
            print("HOW MANY ROWS");  # 打印提示信息，询问行数
            r = parseInt(await input());  # 获取行数输入并转换为整数赋值给变量r
            if (r <= 9):  # 如果行数小于等于9
                break;  # 退出循环
            print("TOO MANY ROWS (9 IS MAXIMUM). NOW ");  # 如果行数大于9，打印提示信息
        while (1):  # 进入无限循环
            print("HOW MANY COLUMNS");  # 打印提示信息，询问列数
            c = parseInt(await input());  # 获取列数输入并转换为整数赋值给变量c
            if (c <= 9):  # 如果列数小于等于9
                break;  # 退出循环
            print("TOO MANY COLUMNS (9 IS MAXIMUM). NOW ");  # 如果列数大于9，打印提示信息
        print("\n");  # 打印空行
        init_board();  # 调用初始化棋盘的函数
        while (1) {
            // 进入游戏循环，直到游戏结束
            // 打印游戏棋盘
            show_board();
            // 依次获取每位玩家的选择
            i1++;
            p1 = i1 - Math.floor(i1 / p) * p;
            if (p1 == 0)
                p1 = p;
            // 循环直到玩家选择有效的坐标
            while (1) {
                print("PLAYER " + p1 + "\n");
                print("COORDINATES OF CHOMP (ROW,COLUMN)");
                str = await input();
                r1 = parseInt(str);
                c1 = parseInt(str.substr(str.indexOf(",") + 1));
                if (r1 >= 1 && r1 <= r && c1 >= 1 && c1 <= c && a[r1][c1] != 0)
                    break;
                print("NO FAIR. YOU'RE TRYING TO CHOMP ON EMPTY SPACE!\n");
            }
            // 如果玩家选择的坐标对应的方块为-1，则游戏结束
            if (a[r1][c1] == -1)
                break;
            for (i = r1; i <= r; i++)
                for (j = c1; j <= c; j++)
                    a[i][j] = 0;
        }
        // 游戏结束检测
        print("YOU LOSE, PLAYER " + p1 + "\n");
        print("\n");
        print("AGAIN (1=YES, 0=NO!)");
        r = parseInt(await input());
        if (r != 1)
            break;
    }
}

main();
```

在这段代码中：

- `for (i = r1; i <= r; i++)`：使用循环遍历行数r1到r之间的值，r1和r是变量。
- `for (j = c1; j <= c; j++)`：在上述行数的基础上，再使用循环遍历列数c1到c之间的值，c1和c是变量。
- `a[i][j] = 0;`：将数组a中第i行第j列的值设为0。
- `print("YOU LOSE, PLAYER " + p1 + "\n");`：打印出玩家p1输掉游戏的消息。
- `print("\n");`：打印一个空行。
- `print("AGAIN (1=YES, 0=NO!)");`：打印提示消息，询问玩家是否再玩一次。
- `r = parseInt(await input());`：将输入的值转换为整数并赋给变量r。
- `if (r != 1)`：如果r不等于1，则执行下一步。
- `break;`：跳出当前循环。
- `main();`：调用主函数main()。
```
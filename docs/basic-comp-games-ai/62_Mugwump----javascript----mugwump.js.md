# `basic-computer-games\62_Mugwump\javascript\mugwump.js`

```py
// 定义一个打印函数，将字符串输出到指定元素
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义一个输入函数，返回一个 Promise 对象
function input()
{
    var input_element;
    var input_str;

    return new Promise(function (resolve) {
                       // 创建一个输入框元素
                       input_element = document.createElement("INPUT");

                       // 输出提示符
                       print("? ");
                       // 设置输入框类型和长度
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       // 将输入框添加到指定元素中
                       document.getElementById("output").appendChild(input_element);
                       // 让输入框获得焦点
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘按下事件
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      // 获取输入的字符串
                                                      input_str = input_element.value;
                                                      // 移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 输出输入的字符串
                                                      print(input_str);
                                                      print("\n");
                                                      // 解析输入的字符串并返回
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

// 定义一个生成指定数量空格的函数
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 定义一个空数组
var p = [];

// 主程序
async function main()
{
    // 输出标题
    print(tab(33) + "MUGWUMP\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    // 输出游戏说明
    print("THE OBJECT OF THIS GAME IS TO FIND FOUR MUGWUMPS\n");
    print("HIDDEN ON A 10 BY 10 GRID.  HOMEBASE IS POSITION 0,0.\n");
    print("ANY GUESS YOU MAKE MUST BE TWO NUMBERS WITH EACH\n");
}
    # 打印提示信息，要求输入一个介于0和9之间的数字，第一个数字表示距离homebase的右侧距离，第二个数字表示距离homebase的上方距离
    print("NUMBER BETWEEN 0 AND 9, INCLUSIVE.  FIRST NUMBER\n");
    # 打印提示信息，说明每个数字的含义
    print("IS DISTANCE TO RIGHT OF HOMEBASE AND SECOND NUMBER\n");
    print("IS DISTANCE ABOVE HOMEBASE.\n");
    # 打印空行
    print("\n");
    # 打印提示信息，说明玩家有10次尝试的机会
    print("YOU GET 10 TRIES.  AFTER EACH TRY, I WILL TELL\n");
    # 打印提示信息，说明每次尝试后会告诉玩家离每个mugwump的距离
    print("YOU HOW FAR YOU ARE FROM EACH MUGWUMP.\n");
    # 打印空行
    print("\n");
    # 无限循环，直到条件不满足
    while (1) {
        # 循环4次，初始化p数组
        for (i = 1; i <= 4; i++) {
            p[i] = [];
            # 循环2次，为p[i]数组赋随机值
            for (j = 1; j <= 2; j++) {
                p[i][j] = Math.floor(10 * Math.random());
            }
        }
        # 初始化t为0
        t = 0;
        # do-while循环，最多执行10次
        do {
            t++;
            # 打印提示信息
            print("\n");
            print("\n");
            print("TURN NO. " + t + " -- WHAT IS YOUR GUESS");
            # 等待用户输入
            str = await input();
            # 解析用户输入的字符串为整数
            m = parseInt(str);
            n = parseInt(str.substr(str.indexOf(",") + 1));
            # 遍历p数组，判断用户猜测的坐标与Mugwump的位置关系
            for (i = 1; i <= 4; i++) {
                if (p[i][1] == -1)
                    continue;
                if (p[i][1] == m && p[i][2] == n) {
                    p[i][1] = -1;
                    print("YOU HAVE FOUND MUGWUMP " + i + "\n");
                } else {
                    d = Math.sqrt(Math.pow(p[i][1] - m, 2) + Math.pow(p[i][2] - n, 2));
                    print("YOU ARE " + Math.floor(d * 10) / 10 + " UNITS FROM MUGWUMP " + i + "\n");
                }
            }
            # 判断是否已找到所有Mugwump
            for (j = 1; j <= 4; j++) {
                if (p[j][1] != -1)
                    break;
            }
            # 如果已找到所有Mugwump，打印信息并跳出循环
            if (j > 4) {
                print("\n");
                print("YOU GOT THEM ALL IN " + t + " TURNS!\n");
                break;
            }
        } while (t < 10) ;
        # 如果循环达到10次，打印信息并显示Mugwump的位置
        if (t == 10) {
            print("\n");
            print("SORRY, THAT'S 10 TRIES.  HERE IS WHERE THEY'RE HIDING:\n");
            for (i = 1; i <= 4; i++) {
                if (p[i][1] != -1)
                    print("MUGWUMP " + i + " IS AT (" + p[i][1] + "," + p[i][2] + ")\n");
            }
        }
        # 打印提示信息，重新开始游戏
        print("\n");
        print("THAT WAS FUN! LET'S PLAY AGAIN.......\n");
        print("FOUR MORE MUGWUMPS ARE NOW IN HIDING.\n");
    }
# 调用名为main的函数
main();
```
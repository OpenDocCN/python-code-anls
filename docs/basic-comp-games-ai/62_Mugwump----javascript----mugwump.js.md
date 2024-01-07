# `basic-computer-games\62_Mugwump\javascript\mugwump.js`

```

// 定义一个打印函数，将字符串输出到指定的元素中
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义一个输入函数，返回一个 Promise 对象
function input()
{
    // 声明变量
    var input_element;
    var input_str;

    // 返回一个 Promise 对象
    return new Promise(function (resolve) {
                       // 创建一个输入框元素
                       input_element = document.createElement("INPUT");

                       // 输出提示符
                       print("? ");
                       // 设置输入框属性
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       // 将输入框添加到指定元素中
                       document.getElementById("output").appendChild(input_element);
                       // 输入框获取焦点
                       input_element.focus();
                       // 初始化输入字符串
                       input_str = undefined;
                       // 监听键盘按下事件
                       input_element.addEventListener("keydown", function (event) {
                                                      // 如果按下的是回车键
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

// 声明一个数组
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
    // 输出游戏规则
    print("THE OBJECT OF THIS GAME IS TO FIND FOUR MUGWUMPS\n");
    print("HIDDEN ON A 10 BY 10 GRID.  HOMEBASE IS POSITION 0,0.\n");
    print("ANY GUESS YOU MAKE MUST BE TWO NUMBERS WITH EACH\n");
    print("NUMBER BETWEEN 0 AND 9, INCLUSIVE.  FIRST NUMBER\n");
    print("IS DISTANCE TO RIGHT OF HOMEBASE AND SECOND NUMBER\n");
    print("IS DISTANCE ABOVE HOMEBASE.\n");
    print("\n");
    print("YOU GET 10 TRIES.  AFTER EACH TRY, I WILL TELL\n");
    print("YOU HOW FAR YOU ARE FROM EACH MUGWUMP.\n");
    print("\n");
    // 游戏循环
    while (1) {
        // 随机生成四个位置
        for (i = 1; i <= 4; i++) {
            p[i] = [];
            for (j = 1; j <= 2; j++) {
                p[i][j] = Math.floor(10 * Math.random());
            }
        }
        t = 0;
        // 游戏逻辑
        do {
            t++;
            print("\n");
            print("\n");
            print("TURN NO. " + t + " -- WHAT IS YOUR GUESS");
            // 获取用户输入
            str = await input();
            m = parseInt(str);
            n = parseInt(str.substr(str.indexOf(",") + 1));
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
            for (j = 1; j <= 4; j++) {
                if (p[j][1] != -1)
                    break;
            }
            if (j > 4) {
                print("\n");
                print("YOU GOT THEM ALL IN " + t + " TURNS!\n");
                break;
            }
        } while (t < 10) ;
        if (t == 10) {
            print("\n");
            print("SORRY, THAT'S 10 TRIES.  HERE IS WHERE THEY'RE HIDING:\n");
            for (i = 1; i <= 4; i++) {
                if (p[i][1] != -1)
                    print("MUGWUMP " + i + " IS AT (" + p[i][1] + "," + p[i][2] + ")\n");
            }
        }
        print("\n");
        print("THAT WAS FUN! LET'S PLAY AGAIN.......\n");
        print("FOUR MORE MUGWUMPS ARE NOW IN HIDING.\n");
    }
}

// 调用主程序
main();

```
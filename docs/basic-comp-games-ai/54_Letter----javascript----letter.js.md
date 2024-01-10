# `basic-computer-games\54_Letter\javascript\letter.js`

```
// 定义一个打印函数，将字符串添加到输出元素中
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

                       // 打印提示符
                       print("? ");
                       // 设置输入框类型和长度
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       // 将输入框添加到输出元素中
                       document.getElementById("output").appendChild(input_element);
                       // 让输入框获得焦点
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘事件
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      // 获取输入的字符串
                                                      input_str = input_element.value;
                                                      // 从输出元素中移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入的字符串
                                                      print(input_str);
                                                      print("\n");
                                                      // 解析 Promise 对象
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

// 定义一个制表符函数，返回指定数量的空格字符串
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 主程序
async function main()
{
    // 打印标题
    print(tab(33) + "LETTER\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    // 打印游戏介绍
    print("LETTER GUESSING GAME\n");
    print("\n");
    print("I'LL THINK OF A LETTER OF THE ALPHABET, A TO Z.\n");
    print("TRY TO GUESS MY LETTER AND I'LL GIVE YOU CLUES\n");
    print("AS TO HOW CLOSE YOU'RE GETTING TO MY LETTER.\n");
    # 进入无限循环，直到条件不满足
    while (1) {
        # 生成一个随机字母的 ASCII 码，范围在 65 到 90 之间
        l = 65 + Math.floor(26 * Math.random());
        # 初始化猜测次数为 0
        g = 0;
        # 打印提示信息
        print("\n");
        print("O.K., I HAVE A LETTER.  START GUESSING.\n");
        # 进入内层无限循环，直到条件不满足
        while (1) {
            # 打印提示信息
            print("\n");
            print("WHAT IS YOUR GUESS");
            # 猜测次数加一
            g++;
            # 等待用户输入
            str = await input();
            # 获取用户输入字符的 ASCII 码
            a = str.charCodeAt(0);
            # 打印提示信息
            print("\n");
            # 判断用户输入的字符与随机生成的字符是否相等，相等则跳出内层循环
            if (a == l)
                break;
            # 如果用户输入的字符小于随机生成的字符，打印提示信息
            if (a < l) {
                print("TOO LOW.  TRY A HIGHER LETTER.\n");
            } else {
                # 如果用户输入的字符大于随机生成的字符，打印提示信息
                print("TOO HIGH.  TRY A LOWER LETTER.\n");
            }
        }
        # 打印猜测次数
        print("\n");
        print("YOU GOT IT IN " + g + " GUESSES!!\n");
        # 如果猜测次数大于 5，打印提示信息
        if (g > 5) {
            print("BUT IT SHOULDN'T TAKE MORE THAN 5 GUESSES!\n");
        } else {
            # 如果猜测次数小于等于 5，打印提示信息
            print("GOOD JOB !!!!!\n");
        }
        # 打印提示信息
        print("\n");
        print("LET'S PLAY AGAIN.....");
    }
# 结束当前的函数或代码块
}

# 调用名为main的函数
main();
```
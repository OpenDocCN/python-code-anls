# `basic-computer-games\54_Letter\javascript\letter.js`

```

// 定义一个打印函数，将字符串输出到指定的元素中
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
                       // 输入框获取焦点
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘事件
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      // 获取输入的字符串
                                                      input_str = input_element.value;
                                                      // 移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 输出输入的字符串
                                                      print(input_str);
                                                      print("\n");
                                                      // 解析输入的字符串
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
    // 输出标题
    print(tab(33) + "LETTER\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("LETTER GUESSING GAME\n");
    print("\n");
    print("I'LL THINK OF A LETTER OF THE ALPHABET, A TO Z.\n");
    print("TRY TO GUESS MY LETTER AND I'LL GIVE YOU CLUES\n");
    print("AS TO HOW CLOSE YOU'RE GETTING TO MY LETTER.\n");
    // 游戏循环
    while (1) {
        // 随机生成一个字母的 ASCII 码
        l = 65 + Math.floor(26 * Math.random());
        g = 0;
        print("\n");
        print("O.K., I HAVE A LETTER.  START GUESSING.\n");
        // 猜字母循环
        while (1) {

            print("\n");
            print("WHAT IS YOUR GUESS");
            g++;
            // 等待用户输入
            str = await input();
            a = str.charCodeAt(0);
            print("\n");
            // 判断用户猜测的字母与随机生成的字母的大小关系
            if (a == l)
                break;
            if (a < l) {
                print("TOO LOW.  TRY A HIGHER LETTER.\n");
            } else {
                print("TOO HIGH.  TRY A LOWER LETTER.\n");
            }
        }
        print("\n");
        print("YOU GOT IT IN " + g + " GUESSES!!\n");
        // 根据猜测次数输出不同的提示
        if (g > 5) {
            print("BUT IT SHOULDN'T TAKE MORE THAN 5 GUESSES!\n");
        } else {
            print("GOOD JOB !!!!!\n");
        }
        print("\n");
        print("LET'S PLAY AGAIN.....");
    }
}

// 调用主程序
main();

```
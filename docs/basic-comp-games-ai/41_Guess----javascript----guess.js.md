# `basic-computer-games\41_Guess\javascript\guess.js`

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
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       document.getElementById("output").appendChild(input_element);
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘事件，当按下回车键时，将输入的值添加到输出元素中，并解析为字符串
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

// 打印空行
function make_space()
{
    for (h = 1; h <= 5; h++)
        print("\n");
}

// 主控制部分，使用 async 函数定义
async function main()
{
    while (1) {
        // 打印游戏标题和介绍
        print(tab(33) + "GUESS\n");
        print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
        print("\n");
        print("\n");
        print("\n");
        print("THIS IS A NUMBER GUESSING GAME. I'LL THINK\n");
        print("OF A NUMBER BETWEEN 1 AND ANY LIMIT YOU WANT.\n");
        print("THEN YOU HAVE TO GUESS WHAT IT IS.\n");
        print("\n");

        // 获取用户输入的限制值
        print("WHAT LIMIT DO YOU WANT");
        l = parseInt(await input());
        print("\n");
        // 计算猜测次数的上限
        l1 = Math.floor(Math.log(l) / Math.log(2)) + 1;
        while (1) {
            // 随机生成一个要猜测的数字
            print("I'M THINKING OF A NUMBER BETWEEN 1 AND " + l + "\n");
            g = 1;
            print("NOW YOU TRY TO GUESS WHAT IT IS.\n");
            m = Math.floor(l * Math.random() + 1);
            while (1) {
                // 获取用户输入的猜测值
                n = parseInt(await input());
                if (n <= 0) {
                    make_space();
                    break;
                }
                if (n == m) {
                    // 判断猜测结果，并打印相应的消息
                    print("THAT'S IT! YOU GOT IT IN " + g + " TRIES.\n");
                    if (g == l1) {
                        print("GOOD.\n");
                    } else if (g < l1) {
                        print("VERY GOOD.\n");
                    } else {
                        print("YOU SHOULD HAVE BEEN TO GET IT IN ONLY " + l1 + "\n");
                    }
                    make_space();
                    break;
                }
                g++;
                if (n > m)
                    print("TOO HIGH. TRY A SMALLER ANSWER.\n");
                else
                    print("TOO LOW. TRY A BIGGER ANSWER.\n");
            }
            if (n <= 0)
                break;
        }
    }
}

// 调用主函数
main();

```
# `basic-computer-games\82_Stars\javascript\stars.js`

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
                       // 监听键盘事件，当按下回车键时，获取输入值并解析 Promise
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      input_str = input_element.value;
                                                      // 移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入值
                                                      print(input_str);
                                                      print("\n");
                                                      // 解析 Promise
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

// 初始化变量
var guesses = 7;
var limit = 100;

// 主程序
async function main()
{
    // 打印标题
    print(tab(33) + "STARS\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n\n\n");

    // 打印是否需要说明
    print("DO YOU WANT INSTRUCTIONS? (Y/N)");
    // 等待输入，并将输入值赋给 instructions 变量
    var instructions = await input();
    // 如果用户输入的指令以小写字母y开头
    if(instructions.toLowerCase()[0] == "y") {
        // 打印游戏提示信息
        print(`I AM THINKING OF A WHOLE NUMBER FROM 1 TO ${limit}\n`);
        print("TRY TO GUESS MY NUMBER.  AFTER YOU GUESS, I\n");
        print("WILL TYPE ONE OR MORE STARS (*).  THE MORE\n");
        print("STARS I TYPE, THE CLOSER YOU ARE TO MY NUMBER.\n");
        print("ONE STAR (*) MEANS FAR AWAY, SEVEN STARS (*******)\n");
        print(`MEANS REALLY CLOSE!  YOU GET ${guesses} GUESSES.\n\n\n`);
    }

    // 游戏循环
    while (true) {
        // 生成一个1到limit之间的随机整数
        var randomNum = Math.floor(Math.random() * limit) + 1;
        // 设置loss变量为true
        var loss = true;

        // 打印游戏提示信息
        print("\nOK, I AM THINKING OF A NUMBER, START GUESSING.\n\n");

        // 循环进行猜数字游戏
        for(var guessNum=1; guessNum <= guesses; guessNum++) {
            // 输入猜测的数字
            print("YOUR GUESS");
            var guess = parseInt(await input());

            // 检查猜测是否正确
            if(guess == randomNum) {
                // 设置loss变量为false
                loss = false;
                // 打印猜对的提示信息
                print("\n\n" + "*".repeat(50) + "!!!\n");
                print(`YOU GOT IT IN ${guessNum} GUESSES!!! LET'S PLAY AGAIN...\n`);
                // 退出当前循环
                break;
            }

            // 输出猜测与目标数字的距离
            var dist = Math.abs(guess - randomNum);
            if(isNaN(dist)) print("*");
            else if(dist >= 64) print("*");
            else if(dist >= 32) print("**");
            else if(dist >= 16) print("***");
            else if(dist >= 8) print("****");
            else if(dist >= 4) print("*****");
            else if(dist >= 2) print("******");
            else print("*******")
            print("\n\n")
        }

        // 如果loss为true
        if(loss) {
            // 打印猜错的提示信息
            print(`SORRY, THAT'S ${guesses} GUESSES. THE NUMBER WAS ${randomNum}\n`);
        }
    }
# 结束 main 函数的定义
}

# 调用 main 函数
main();
```
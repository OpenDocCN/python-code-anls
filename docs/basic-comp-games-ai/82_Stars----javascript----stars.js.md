# `basic-computer-games\82_Stars\javascript\stars.js`

```

// 定义一个打印函数，将字符串输出到指定的元素中
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义一个输入函数，返回一个 Promise 对象，当用户输入完成后 resolve
function input()
{
    var input_element;
    var input_str;

    return new Promise(function (resolve) {
                       input_element = document.createElement("INPUT");

                       // 输出提示符
                       print("? ");
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       document.getElementById("output").appendChild(input_element);
                       input_element.focus();
                       input_str = undefined;
                       // 监听用户输入，当按下回车键时，将输入的值 resolve 出去
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

// 定义一个生成指定数量空格的函数
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
    // 输出标题和说明
    print(tab(33) + "STARS\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n\n\n");

    // 是否需要显示游戏说明
    print("DO YOU WANT INSTRUCTIONS? (Y/N)");
    var instructions = await input();
    if(instructions.toLowerCase()[0] == "y") {
        print(`I AM THINKING OF A WHOLE NUMBER FROM 1 TO ${limit}\n`);
        print("TRY TO GUESS MY NUMBER.  AFTER YOU GUESS, I\n");
        print("WILL TYPE ONE OR MORE STARS (*).  THE MORE\n");
        print("STARS I TYPE, THE CLOSER YOU ARE TO MY NUMBER.\n");
        print("ONE STAR (*) MEANS FAR AWAY, SEVEN STARS (*******)\n");
        print(`MEANS REALLY CLOSE!  YOU GET ${guesses} GUESSES.\n\n\n`);
    }

    // 游戏循环
    while (true) {

        // 生成随机数
        var randomNum = Math.floor(Math.random() * limit) + 1;
        var loss = true;

        print("\nOK, I AM THINKING OF A NUMBER, START GUESSING.\n\n");

        for(var guessNum=1; guessNum <= guesses; guessNum++) {

            // 输入猜测
            print("YOUR GUESS");
            var guess = parseInt(await input());

            // 检查猜测是否正确
            if(guess == randomNum) {
                loss = false;
                print("\n\n" + "*".repeat(50) + "!!!\n");
                print(`YOU GOT IT IN ${guessNum} GUESSES!!! LET'S PLAY AGAIN...\n`);
                break;
            }

            // 输出星号距离
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

        if(loss) {
            print(`SORRY, THAT'S ${guesses} GUESSES. THE NUMBER WAS ${randomNum}\n`);
        }
    }
}

// 调用主程序
main();

```
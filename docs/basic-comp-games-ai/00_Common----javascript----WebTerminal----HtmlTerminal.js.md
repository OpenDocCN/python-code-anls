# `basic-computer-games\00_Common\javascript\WebTerminal\HtmlTerminal.js`

```
/**
 * @class HtmlTerminal
 * 
 * This class is a very basic implementation of a "terminal" in the browser.
 * It provides simple functions like "write" and an "input" Callback.
 * 
 * @license AGPL-2.0
 * @author Alexaner Wunschik <https://github.com/mojoaxel>
 */
class HtmlTerminal {

  /**
   * Input callback.
   * If the prompt is activated by calling the input function
   * a callback is defined. If this member is not set this means
   * the prompt is not active.
   * 
   * @private
   * @type {function}
   */
  #inputCallback = undefined;

  /**
   * A html element to show a "prompt".
   * 
   * @private
   * @type {HTMLElement}
   */
  #$prompt;
  
  /**
   * Constructor
   * Creates a basic terminal simulation on the provided HTMLElement.
   * 
   * @param {HTMLElement} $output - a dom element
   */
  constructor($output) {
    // Store the output DOM element in a local variable.
    this.$output = $output;

    // Clear terminal.
    this.clear();

    // Add the call "terminal" to the $output element.
    this.$output.classList.add('terminal');

    // Create a prompt element.
    // This element gets added if input is needed.
    this.#$prompt = document.createElement("input");
    this.#$prompt.setAttribute("id", "prompt");
    this.#$prompt.setAttribute("type", "text");
    this.#$prompt.setAttribute("length", "50");
    this.#$prompt.addEventListener("keydown", this.#handleKey.bind(this));

    // Force focus on the promt on each click.
    // This is needed for mobile support.
    document.body.addEventListener('click', () => {
      this.#$prompt.focus();
    });
  }

  /**
   * Creates a new HTMLElement with the given text content.
   * This element than gets added to the $output as a new "line".
   * 
   * @private
   * @memberof MinimalTerminal
   * @param {String} text - text that should be displayed in the new "line".
   * @returns {HTMLElement} return a new DOM Element <pre class="line"></pre>
   */
  #newLine(text) {
    // 创建一个 <pre> 元素节点
    const $lineNode = document.createElement("pre");
    // 添加 "line" 类到节点
    $lineNode.classList.add("line");
    // 设置节点的文本内容为传入的文本
    $lineNode.innerText = text;
    // 返回创建的节点
    return $lineNode;
  }

  /**
   * TODO
   * 
   * @private
   * @param {*} e 
   */
  #handleKey(e) {
    // 如果没有定义输入回调函数，则直接返回
    if (!this.#inputCallback) {
      return;
    }

    // 如果按下的键是回车键
    if (e.keyCode == 13) {
      // 获取输入框的文本内容
      const text = this.#$prompt.value;
      // 清空输入框的内容
      this.#$prompt.value = '';
      // 移除输入框
      this.#$prompt.remove();
      // 调用输入回调函数，传入文本内容并加上换行符
      this.#inputCallback(text + '\n');
    }
  }

  /**
   * Clear the terminal.
   * Remove all lines.
   * 
   * @public
   */
  clear() {
    // 清空输出区域的文本内容
    this.$output.innerText = "";
  }

  /**
   * Create a new div and add html content.
   * 
   * @public
   * @param {*} htmlContent 
   */
  inserHtml(htmlContent) {
    // 创建一个新的 <div> 元素节点
    const $htmlNode = document.createElement("div");
    // 设置节点的 HTML 内容为传入的 HTML 内容
    $htmlNode.innerHTML = htmlContent;
    // 将节点添加到输出区域
    this.$output.appendChild($htmlNode);
    // 滚动页面到底部
    document.body.scrollTo(0, document.body.scrollHeight);
  }

  /**
   * Write a text to the terminal.
   * By default there is no linebreak at the end of a new line
   * except the line ensd with a "\n".
   * If the given text has multible linebreaks, multibe lines are inserted.
   * 
   * @public
   * @param {string} text 
   */
  write(text) {
    // 如果文本为空或长度为0
    if (!text || text.length <= 0) {
      // 添加一个空行
      this.$output.appendChild(document.createElement("br"));
    } else if (text.endsWith("\n")) {
      // 如果文本以换行符结尾
      // 创建一个新的行节点并添加到输出区域
      const $lineNode = this.#newLine(text);
      this.$output.appendChild(this.#newLine(text));
      // 添加一个换行符
      this.$output.appendChild(document.createElement("br"));
    } else if (text.includes("\n")) {
      // 如果文本包含多个换行符
      // 按照换行符分割文本，逐行写入
      const lines = text.split("\n");
      lines.forEach((line) => {
        this.write(line);
      });
    } else {
      // 如果文本是单行
      // 创建一个新的行节点并添加到输出区域
      this.$output.appendChild(this.#newLine(text));
    }

    // 滚动页面到底部
    // 将页面滚动到底部
    document.body.scrollTo(0, document.body.scrollHeight);
  }

  /**
   * 类似于 "write"，但在末尾添加一个换行符。
   * 
   * @public
   * @param {*} text 
   */
  writeln(text) {
    // 调用 write 方法并在末尾添加换行符
    this.write(text + "\n");
  }

  /**
   * 从用户输入中查询信息。
   * 通过在终端末尾添加一个输入元素来实现，显示提示和闪烁的光标。
   * 如果按下键盘，则将输入添加到提示元素中。
   * 输入以换行符结束。
   * 
   * @public
   * @param {*} callback 
   */
  input(callback) {
    // 显示带有闪烁光标的提示
    this.#inputCallback = callback;
    this.$output.appendChild(this.#$prompt);
    this.#$prompt.focus();
  }
# 闭合前面的函数定义
```
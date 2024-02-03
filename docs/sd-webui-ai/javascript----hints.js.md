# `stable-diffusion-webui\javascript\hints.js`

```
// 鼠标悬停提示，用于各种 UI 元素

// 定义各个 UI 元素的标题和对应的提示信息
var titles = {
    "Sampling steps": "How many times to improve the generated image iteratively; higher values take longer; very low values can produce bad results",
    // 采样步数：迭代改进生成图像的次数；较高的值需要更长时间；非常低的值可能会产生糟糕的结果
    "Sampling method": "Which algorithm to use to produce the image",
    // 采样方法：用于生成图像的算法
    "GFPGAN": "Restore low quality faces using GFPGAN neural network",
    // GFPGAN：使用 GFPGAN 神经网络恢复低质量的人脸
    "Euler a": "Euler Ancestral - very creative, each can get a completely different picture depending on step count, setting steps higher than 30-40 does not help",
    // Euler a：Euler Ancestral - 非常有创意，每个人可以根据步数得到完全不同的图片，将步数设置得超过 30-40 不会有帮助
    "DDIM": "Denoising Diffusion Implicit Models - best at inpainting",
    // DDIM：去噪扩散隐式模型 - 最擅长修复图像缺失部分
    "UniPC": "Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models",
    // UniPC：统一的预测-校正框架，用于快速采样扩散模型
    "DPM adaptive": "Ignores step count - uses a number of steps determined by the CFG and resolution",
    // DPM 自适应：忽略步数 - 使用由 CFG 和分辨率确定的步数

    "\u{1F4D0}": "Auto detect size from img2img",
    // 自动从 img2img 检测尺寸
    "Batch count": "How many batches of images to create (has no impact on generation performance or VRAM usage)",
    // 批次计数：要创建多少批次的图像（不影响生成性能或 VRAM 使用）
    "Batch size": "How many image to create in a single batch (increases generation performance at cost of higher VRAM usage)",
    // 批次大小：在单个批次中创建多少图像（提高生成性能但会增加 VRAM 使用）
    "CFG Scale": "Classifier Free Guidance Scale - how strongly the image should conform to prompt - lower values produce more creative results",
    // CFG 比例：无分类器指导比例 - 图像应该如何符合提示 - 较低的值会产生更有创意的结果
    "Seed": "A value that determines the output of random number generator - if you create an image with same parameters and seed as another image, you'll get the same result",
    // 种子：确定随机数生成器输出的值 - 如果使用相同的参数和种子创建图像，则会得到相同的结果
    "\u{1f3b2}\ufe0f": "Set seed to -1, which will cause a new random number to be used every time",
    // 将种子设置为 -1，这将导致每次使用新的随机数
    "\u267b\ufe0f": "Reuse seed from last generation, mostly useful if it was randomized",
    // 重用上一代的种子，如果它是随机的，则大多数情况下很有用
    "\u2199\ufe0f": "Read generation parameters from prompt or last generation if prompt is empty into user interface.",
    // 从提示或上一代（如果提示为空）读取生成参数到用户界面
    "\u{1f4c2}": "Open images output directory",
    // 打开图像输出目录
    "\u{1f4be}": "Save style",
    // 保存样式
    "\u{1f5d1}\ufe0f": "Clear prompt",
    // 清除提示
    "\u{1f4cb}": "Apply selected styles to current prompt",
    // 将选定的样式应用于当前提示
    "\u{1f4d2}": "Paste available values into the field",
    // 将可用值粘贴到字段中
    "\u{1f3b4}": "Show/hide extra networks",
    // 显示/隐藏额外网络
    "\u{1f300}": "Restore progress", 
    # 使用 Unicode 表情符号表示“恢复进度”，可能用于显示进度或状态信息

    "Inpaint a part of image": "Draw a mask over an image, and the script will regenerate the masked area with content according to prompt", 
    # 对图像的一部分进行修复：在图像上绘制一个蒙版，脚本将根据提示重新生成蒙版区域的内容

    "SD upscale": "Upscale image normally, split result into tiles, improve each tile using img2img, merge whole image back", 
    # SD 放大：正常放大图像，将结果分割成瓷砖，使用 img2img 改进每个瓷砖，然后合并整个图像

    "Just resize": "Resize image to target resolution. Unless height and width match, you will get incorrect aspect ratio.", 
    # 仅调整大小：将图像调整到目标分辨率。除非高度和宽度匹配，否则会得到不正确的宽高比。

    "Crop and resize": "Resize the image so that entirety of target resolution is filled with the image. Crop parts that stick out.", 
    # 裁剪并调整大小：调整图像大小，使整个目标分辨率都填满图像。裁剪超出部分。

    "Resize and fill": "Resize the image so that entirety of image is inside target resolution. Fill empty space with image's colors.", 
    # 调整大小并填充：调整图像大小，使整个图像都在目标分辨率内。用图像的颜色填充空白空间。

    "Mask blur": "How much to blur the mask before processing, in pixels.", 
    # 蒙版模糊：在处理之前对蒙版进行多少像素的模糊处理。

    "Masked content": "What to put inside the masked area before processing it with Stable Diffusion.", 
    # 蒙版内容：在使用 Stable Diffusion 处理蒙版区域之前放入其中的内容。

    "fill": "fill it with colors of the image", 
    # 填充：用图像的颜色填充

    "original": "keep whatever was there originally", 
    # 原始：保留原来的内容

    "latent noise": "fill it with latent space noise", 
    # 潜在噪音：用潜在空间的噪音填充

    "latent nothing": "fill it with latent space zeroes", 
    # 潜在空间零值：用潜在空间的零值填充

    "Inpaint at full resolution": "Upscale masked region to target resolution, do inpainting, downscale back and paste into original image", 
    # 在完整分辨率下修复：将蒙版区域放大到目标分辨率，进行修复，然后缩小并粘贴回原始图像

    "Denoising strength": "Determines how little respect the algorithm should have for image's content. At 0, nothing will change, and at 1 you'll get an unrelated image. With values below 1.0, processing will take less steps than the Sampling Steps slider specifies.", 
    # 降噪强度：确定算法对图像内容的尊重程度。在 0 时，不会发生变化；在 1 时，会得到一个无关的图像。值小于 1.0 时，处理步骤会比采样步骤滑块指定的步骤少。

    "Skip": "Stop processing current image and continue processing.", 
    # 跳过：停止处理当前图像并继续处理。

    "Interrupt": "Stop processing images and return any results accumulated so far.", 
    # 中断：停止处理图像并返回到目前为止累积的任何结果。

    "Save": "Write image to a directory (default - log/images) and generation parameters into csv file.", 
    # 保存：将图像写入目录（默认为 log/images），并将生成参数写入 csv 文件。

    "X values": "Separate values for X axis using commas.", 
    # X 值：使用逗号分隔 X 轴的值。

    "Y values": "Separate values for Y axis using commas.", 
    # Y 值：使用逗号分隔 Y 轴的值。

    "None": "Do not do anything special", 
    # 无：不执行任何特殊操作
    # 将提示矩阵分隔成部分，使用竖线字符（|），脚本将为每种组合创建一张图片（除了第一部分，它将出现在所有组合中）
    "Prompt matrix": "Separate prompts into parts using vertical pipe character (|) and the script will create a picture for every combination of them (except for the first part, which will be present in all combinations)",

    # 创建具有不同参数的图像的网格。使用下面的输入指定哪些参数将由列和行共享
    "X/Y/Z plot": "Create grid(s) where images will have different parameters. Use inputs below to specify which parameters will be shared by columns and rows",

    # 运行 Python 代码。仅限高级用户。必须以 --allow-code 运行程序才能正常工作
    "Custom code": "Run Python code. Advanced user only. Must run program with --allow-code for this to work",

    # 使用逗号分隔的单词列表，第一个单词将用作关键字：脚本将在提示中搜索此单词，并用其他单词替换它
    "Prompt S/R": "Separate a list of words with commas, and the first word will be used as a keyword: script will search for this word in the prompt, and replace it with others",

    # 使用逗号分隔的单词列表，脚本将使用这些单词的每种可能的顺序制作提示的变体
    "Prompt order": "Separate a list of words with commas, and the script will make a variation of prompt with those words for their every possible order",

    # 生成可平铺的图像
    "Tiling": "Produce an image that can be tiled.",

    # 对于 SD 放大，瓦片之间应该有多少像素的重叠。瓦片重叠，以便当它们合并回一个图片时，没有明显可见的接缝
    "Tile overlap": "For SD upscale, how much overlap in pixels should there be between tiles. Tiles overlap so that when they are merged back into one picture, there is no clearly visible seam.",

    # 要混合到生成中的不同图片的种子
    "Variation seed": "Seed of a different picture to be mixed into the generation.",

    # 产生多强的变化。在 0 时，不会有影响。在 1 时，您将获得具有变化种子的完整图片（除了祖先采样器，那里您将只得到一些东西）
    "Variation strength": "How strong of a variation to produce. At 0, there will be no effect. At 1, you will get the complete picture with variation seed (except for ancestral samplers, where you will just get something).",

    # 尝试生成与指定分辨率相同种子在相同分辨率下产生的图片类似的图片
    "Resize seed from height": "Make an attempt to produce a picture similar to what would have been produced with same seed at specified resolution",

    # 尝试生成与指定分辨率相同种子在相同分辨率下产生的图片类似的图片
    "Resize seed from width": "Make an attempt to produce a picture similar to what would have been produced with same seed at specified resolution",

    # 从现有图像重建提示并将其放入提示字段中
    "Interrogate": "Reconstruct prompt from existing image and put it into the prompt field.",

    # 使用类似 [seed] 和 [date] 的标签定义图像文件名的选择方式。留空使用默认值
    "Images filename pattern": "Use tags like [seed] and [date] to define how filenames for images are chosen. Leave empty for default.",
    # 目录名称模式：使用标签如 [seed] 和 [date] 来定义如何选择图像和网格的子目录。留空使用默认设置。
    "Directory name pattern": "Use tags like [seed] and [date] to define how subdirectories for images and grids are chosen. Leave empty for default.",
    
    # 最大提示词数：设置在 [prompt_words] 选项中使用的最大单词数；注意：如果单词太长，可能会超过系统处理文件路径的最大长度
    "Max prompt words": "Set the maximum number of words to be used in the [prompt_words] option; ATTENTION: If the words are too long, they may exceed the maximum length of the file path that the system can handle",

    # 回环：执行多次 img2img 处理。输出图像被用作下一个循环的输入。
    "Loopback": "Performs img2img processing multiple times. Output images are used as input for the next loop.",
    
    # 循环次数：处理图像的次数。每个输出被用作下一个循环的输入。如果设置为 1，行为将与未使用此脚本时相同。
    "Loops": "How many times to process an image. Each output is used as the input of the next loop. If set to 1, behavior will be as if this script were not used.",
    
    # 最终去噪强度：每个批次中每个图像的最终循环的去噪强度。
    "Final denoising strength": "The denoising strength for the final loop of each image in the batch.",
    
    # 去噪强度曲线：去噪曲线控制每个循环中去噪强度的变化速率。激进：大部分变化将发生在循环开始时。线性：变化将在所有循环中保持恒定。懒惰：大部分变化将发生在循环结束时。
    "Denoising strength curve": "The denoising curve controls the rate of denoising strength change each loop. Aggressive: Most of the change will happen towards the start of the loops. Linear: Change will be constant through all loops. Lazy: Most of the change will happen towards the end of the loops.",
    
    # 样式 1：要应用的样式；样式具有正面和负面提示的组件，并且适用于两者
    "Style 1": "Style to apply; styles have components for both positive and negative prompts and apply to both",
    
    # 样式 2：要应用的样式；样式具有正面和负面提示的组件，并且适用于两者
    "Style 2": "Style to apply; styles have components for both positive and negative prompts and apply to both",
    
    # 应用样式：将选定的样式插入提示字段
    "Apply style": "Insert selected styles into prompt fields",
    
    # 创建样式：将当前提示保存为样式。如果在文本中添加令牌 {prompt}，则在将来使用样式时，样式将使用该令牌作为您的提示的占位符。
    "Create style": "Save current prompts as a style. If you add the token {prompt} to the text, the style uses that as a placeholder for your prompt when you use the style in the future.",
    
    # 检查点名称：在生成图像之前从检查点加载权重。您可以使用哈希或文件名的一部分（如在设置中看到的）作为检查点名称。建议与 Y 轴一起使用以减少切换。
    "Checkpoint name": "Loads weights from checkpoint before making images. You can either use hash or a part of filename (as seen in settings) for checkpoint name. Recommended to use with Y axis for less switching.",
    # 修复模型的掩膜强度：仅适用于修复模型。确定对于修复和图像到图像模型如何掩蔽原始图像的强度。1.0表示完全掩蔽，这是默认行为。0.0表示完全未掩蔽的条件。较低的值将有助于保留图像的整体构图，但在处理大的变化时可能会出现困难。

    # Eta 噪声种子增量：如果此值非零，则将添加到种子并用于初始化使用 Eta 的采样器的噪声 RNG。您可以使用此功能生成更多图像变化，或者如果知道自己在做什么，可以使用此功能匹配其他软件的图像。

    # 文件名单词正则表达式：此正则表达式将用于从文件名中提取单词，并使用下面的选项将它们连接成用于训练的标签文本。留空以保留文件名文本不变。

    # 文件名连接字符串：如果上面的选项启用，此字符串将用于将拆分的单词连接成单行。

    # 快速设置列表：设置名称的列表，用逗号分隔，用于将设置放在顶部的快速访问栏中，而不是通常的设置选项卡中。查看 modules/shared.py 获取设置名称。需要重新启动才能应用。

    # 加权和：结果 = A * (1 - M) + B * M

    # 添加差异：结果 = A + (B - C) * M

    # 无插值：结果 = A

    # 初始化文本：如果令牌数量多于向量数量，则可能会跳过一些。将文本框留空以从零开始初始化向量。
    "Learning rate": "How fast should training go. Low values will take longer to train, high values may fail to converge (not generate accurate results) and/or may break the embedding (This has happened if you see Loss: nan in the training info textbox. If this happens, you need to manually restore your embedding from an older not-broken backup).\n\nYou can set a single numeric value, or multiple learning rates using the syntax:\n\n   rate_1:max_steps_1, rate_2:max_steps_2, ...\n\nEG:   0.005:100, 1e-3:1000, 1e-5\n\nWill train with rate of 0.005 for first 100 steps, then 1e-3 until 1000 steps, then 1e-5 for all remaining steps.",

    # 学习率：控制训练速度。较低的值将需要更长时间进行训练，较高的值可能无法收敛（生成准确结果）和/或可能会破坏嵌入（如果在训练信息文本框中看到 Loss: nan，则发生了这种情况。如果发生这种情况，您需要手动从旧的未损坏的备份中恢复您的嵌入）。
    # 您可以设置单个数值，或使用以下语法设置多个学习率：
    # rate_1:max_steps_1, rate_2:max_steps_2, ...
    # 例如：0.005:100, 1e-3:1000, 1e-5
    # 将在前100步使用0.005的速率，然后在1000步之前使用1e-3，然后在所有剩余步骤中使用1e-5。

    "Clip skip": "Early stopping parameter for CLIP model; 1 is stop at last layer as usual, 2 is stop at penultimate layer, etc.",

    # Clip skip：CLIP 模型的早停参数；1 表示像往常一样在最后一层停止，2 表示在倒数第二层停止，依此类推。

    "Approx NN": "Cheap neural network approximation. Very fast compared to VAE, but produces pictures with 4 times smaller horizontal/vertical resolution and lower quality.",
    # Approx NN：廉价的神经网络近似。与 VAE 相比非常快，但生成的图片水平/垂直分辨率小 4 倍且质量较低。
    
    "Approx cheap": "Very cheap approximation. Very fast compared to VAE, but produces pictures with 8 times smaller horizontal/vertical resolution and extremely low quality.",
    # Approx cheap：非常廉价的近似。与 VAE 相比非常快，但生成的图片水平/垂直分辨率小 8 倍且质量极低。

    "Hires. fix": "Use a two step process to partially create an image at smaller resolution, upscale, and then improve details in it without changing composition",
    # Hires. fix：使用两步过程部分创建较小分辨率的图像，放大，然后在不改变构图的情况下改进细节。

    "Hires steps": "Number of sampling steps for upscaled picture. If 0, uses same as for original.",
    # Hires steps：放大图片的采样步数。如果为 0，则使用与原始相同的步数。

    "Upscale by": "Adjusts the size of the image by multiplying the original width and height by the selected value. Ignored if either Resize width to or Resize height to are non-zero.",
    # Upscale by：通过将原始宽度和高度乘以所选值来调整图像的大小。如果 Resize width to 或 Resize height to 任一不为零，则忽略。

    "Resize width to": "Resizes image to this width. If 0, width is inferred from either of two nearby sliders.",
    # Resize width to：将图像调整为此宽度。如果为 0，则宽度从两个附近滑块中推断。

    "Resize height to": "Resizes image to this height. If 0, height is inferred from either of two nearby sliders.",
    # Resize height to：将图像调整为此高度。如果为 0，则高度从两个附近滑块中推断。
    # "Discard weights with matching name": "Regular expression; if weights's name matches it, the weights is not written to the resulting checkpoint. Use ^model_ema to discard EMA weights."
    # 当权重的名称与正则表达式匹配时，该权重不会被写入结果检查点。使用 ^model_ema 来丢弃 EMA 权重。

    # "Extra networks tab order": "Comma-separated list of tab names; tabs listed here will appear in the extra networks UI first and in order listed."
    # 额外网络选项卡的顺序：以逗号分隔的选项卡名称列表；在此列出的选项卡将首先出现在额外网络用户界面中，并按照列出的顺序排列。

    # "Negative Guidance minimum sigma": "Skip negative prompt for steps where image is already mostly denoised; the higher this value, the more skips there will be; provides increased performance in exchange for minor quality reduction."
    # 负向引导最小 sigma：跳过图像已经大部分去噪的步骤的负向提示；该值越高，跳过的次数就越多；以牺牲轻微质量降低为代价提供了更高的性能。
// 更新元素的工具提示信息
function updateTooltip(element) {
    // 如果元素已经有标题，则直接返回
    if (element.title) return;

    // 获取元素的文本内容
    let text = element.textContent;
    // 根据文本内容查找对应的本地化工具提示信息
    let tooltip = localization[titles[text]] || titles[text];

    // 如果未找到对应的工具提示信息，则尝试使用元素的值来查找
    if (!tooltip) {
        let value = element.value;
        if (value) tooltip = localization[titles[value]] || titles[value];
    }

    // 如果仍未找到对应的工具提示信息，则尝试使用元素的 data-value 属性来查找
    if (!tooltip) {
        // Gradio 下拉选项具有 data-value 属性
        let dataValue = element.dataset.value;
        if (dataValue) tooltip = localization[titles[dataValue]] || titles[dataValue];
    }

    // 如果仍未找到对应的工具提示信息，则尝试使用元素的类名来查找
    if (!tooltip) {
        for (const c of element.classList) {
            if (c in titles) {
                tooltip = localization[titles[c]] || titles[c];
                break;
            }
        }
    }

    // 如果找到了工具提示信息，则将其设置为元素的标题
    if (tooltip) {
        element.title = tooltip;
    }
}

// 需要检查添加工具提示的节点集合
const tooltipCheckNodes = new Set();
// 用于延迟处理工具提示检查的定时器
let tooltipCheckTimer = null;

// 处理需要检查添加工具提示的节点
function processTooltipCheckNodes() {
    for (const node of tooltipCheckNodes) {
        updateTooltip(node);
    }
    // 清空节点集合
    tooltipCheckNodes.clear();
}

// 当 UI 更新时执行的回调函数
onUiUpdate(function(mutationRecords) {
    // 遍历所有变动记录
    for (const record of mutationRecords) {
        // 检查是否是子节点列表变动，并且目标节点包含 "options" 类
        if (record.type === "childList" && record.target.classList.contains("options")) {
            // 如果是 Gradio 下拉菜单发生变化，将更新显示当前值的输入元素加入更新队列
            let wrap = record.target.parentNode;
            let input = wrap?.querySelector("input");
            if (input) {
                // 清空输入元素的标题，以便更新
                input.title = "";
                // 将输入元素加入待检查节点集合
                tooltipCheckNodes.add(input);
            }
        }
        // 遍历所有新增节点
        for (const node of record.addedNodes) {
            // 检查节点类型为元素节点且不包含 "hide" 类
            if (node.nodeType === Node.ELEMENT_NODE && !node.classList.contains("hide")) {
                // 如果节点没有标题
                if (!node.title) {
                    // 检查节点标签名，如果符合条件则加入待检查节点集合
                    if (
                        node.tagName === "SPAN" ||
                        node.tagName === "BUTTON" ||
                        node.tagName === "P" ||
                        node.tagName === "INPUT" ||
                        (node.tagName === "LI" && node.classList.contains("item")) // Gradio 下拉菜单项
                    ) {
                        tooltipCheckNodes.add(node);
                    }
                }
                // 查找节点内的所有 span、button、p 元素并加入待检查节点集合
                node.querySelectorAll('span, button, p').forEach(n => tooltipCheckNodes.add(n));
            }
        }
    }
    // 如果待检查节点集合非空
    if (tooltipCheckNodes.size) {
        // 清除之前的定时器，设置新的定时器调用处理待检查节点的函数
        clearTimeout(tooltipCheckTimer);
        tooltipCheckTimer = setTimeout(processTooltipCheckNodes, 1000);
    }
});

// 当页面 UI 加载完成后执行的函数
onUiLoaded(function() {
    // 遍历页面中所有的组件
    for (var comp of window.gradio_config.components) {
        // 如果组件有 webui_tooltip 属性和 elem_id 属性
        if (comp.props.webui_tooltip && comp.props.elem_id) {
            // 获取对应 elem_id 的元素
            var elem = gradioApp().getElementById(comp.props.elem_id);
            // 如果元素存在
            if (elem) {
                // 设置元素的 title 属性为 webui_tooltip 的值
                elem.title = comp.props.webui_tooltip;
            }
        }
    }
});
```
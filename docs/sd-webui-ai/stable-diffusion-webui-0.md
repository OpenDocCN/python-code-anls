# SDWebUI源码解析 0

# Stable Diffusion web UI
A browser interface based on Gradio library for Stable Diffusion.

![](screenshot.png)

## Feature showcase

[Detailed feature showcase with images, art by Greg Rutkowski](https://github.com/AUTOMATIC1111/stable-diffusion-webui-feature-showcase)

- Original txt2img and img2img modes
- One click install and run script (but you still must install python and git)
- Outpainting
- Inpainting
- Prompt matrix
- Stable Diffusion upscale
- Attention
- Loopback
- X/Y plot
- Textual Inversion
- Extras tab with:
  - GFPGAN, neural network that fixes faces
  - RealESRGAN, neural network upscaler
  - ESRGAN, neural network with a lot of third party models
- Resizing aspect ratio options
- Sampling method selection
- Interrupt processing at any time
- 4GB videocard support
- Correct seeds for batches
- Prompt length validation
- Generation parameters added as text to PNG
- Tab to view an existing picture's generation parameters
- Settings page
- Running custom code from UI
- Mouseover hints fo most UI elements
- Possible to change defaults/mix/max/step values for UI elements via text config
- Random artist button
- Tiling support: UI checkbox to create images that can be tiled like textures
- Progress bar and live image generation preview

## Installing and running

You need [python](https://www.python.org/downloads/windows/) and [git](https://git-scm.com/download/win)
installed to run this, and an NVidia videocard.

You need `model.ckpt`, Stable Diffusion model checkpoint, a big file containing the neural network weights. You
can obtain it from the following places:
 - [official download](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original)
 - [file storage](https://drive.yerf.org/wl/?id=EBfTrmcCCUAGaQBXVIj5lJmEhjoP1tgl)
 - magnet:?xt=urn:btih:3a4a612d75ed088ea542acac52f9f45987488d1c&dn=sd-v1-4.ckpt&tr=udp%3a%2f%2ftracker.openbittorrent.com%3a6969%2fannounce&tr=udp%3a%2f%2ftracker.opentrackr.org%3a1337

You optionally can use GPFGAN to improve faces, then you'll need to download the model from [here](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth).

To use ESRGAN models, put them into ESRGAN directory in the same location as webui.py. A file will be loaded
as model if it has .pth extension. Grab models from the [Model Database](https://upscale.wiki/wiki/Model_Database).

### Automatic installation/launch

- install [Python 3.10.6](https://www.python.org/downloads/windows/) and check "Add Python to PATH" during installation. You must install this exact version.
- install [git](https://git-scm.com/download/win)
- place `model.ckpt` into webui directory, next to `webui.bat`.
- _*(optional)*_ place `GFPGANv1.3.pth` into webui directory, next to `webui.bat`.
- run `webui-user.bat` from Windows Explorer. Run it as normal user, ***not*** as administrator.

#### Troubleshooting

- if your version of Python is not in PATH (or if another version is), edit `webui-user.bat`, and modify the
line `set PYTHON=python` to say the full path to your python executable, for example: `set PYTHON=B:\soft\Python310\python.exe`.
You can do this for python, but not for git.
- if you get out of memory errors and your video-card has a low amount of VRAM (4GB), use custom parameter `set COMMANDLINE_ARGS` (see section below)
to enable appropriate optimization according to low VRAM guide below (for example, `set COMMANDLINE_ARGS=--medvram --opt-split-attention`).
- to prevent the creation of virtual environment and use your system python, use custom parameter replacing `set VENV_DIR=-` (see below).
- webui.bat installs requirements from files `requirements_versions.txt`, which lists versions for modules specifically compatible with
Python 3.10.6. If you choose to install for a different version of python,  using custom parameter `set REQS_FILE=requirements.txt`
may help (but I still recommend you to just use the recommended version of python).
- if you feel you broke something and want to reinstall from scratch, delete directories: `venv`, `repositories`.
- if you get a green or black screen instead of generated pictures, you have a card that doesn't support half precision
floating point numbers (Known issue with 16xx cards). You must use `--precision full --no-half` in addition to command line
arguments (set them using `set COMMANDLINE_ARGS`, see below), and the model will take much more space in VRAM (you will likely
have to also use at least `--medvram`).
- installer creates python virtual environment, so none of installed modules will affect your system installation of python if
you had one prior to installing this.
- About _"You must install this exact version"_ from the instructions above: you can use any version of python you like,
and it will likely work, but if you want to seek help about things not working, I will not offer help unless you this
exact version for my sanity.

#### How to run with custom parameters

It's possible to edit `set COMMANDLINE_ARGS=` line in `webui.bat` to run the program with different command line arguments, but that may lead
to inconveniences when the file is updated in the repository.

The recommndended way is to use another .bat file named anything you like, set the parameters you want in it, and run webui.bat from it.
A `webui-user.bat` file included into the repository does exactly this.

Here is an example that runs the prgoram with `--opt-split-attention` argument:

```commandline
@echo off

set COMMANDLINE_ARGS=--opt-split-attention

call webui.bat
```

Another example, this file will run the program with custom python path, a different model named `a.ckpt` and without virtual environment:

```commandline
@echo off

set PYTHON=b:/soft/Python310/Python.exe
set VENV_DIR=-
set COMMANDLINE_ARGS=--ckpt a.ckpt

call webui.bat
```

### What options to use for low VRAM video-cards?
You can, through command line arguments, enable the various optimizations which sacrifice some/a lot of speed in favor of
using less VRAM. Those arguments are added to the `COMMANDLINE_ARGS` parameter, see section above.

Here's a list of optimization arguments:
- If you have 4GB VRAM and want to make 512x512 (or maybe up to 640x640) images, use `--medvram`.
- If you have 4GB VRAM and want to make 512x512 images, but you get an out of memory error with `--medvram`, use `--medvram --opt-split-attention` instead.
- If you have 4GB VRAM and want to make 512x512 images, and you still get an out of memory error, use `--lowvram --always-batch-cond-uncond --opt-split-attention` instead.
- If you have 4GB VRAM and want to make images larger than you can with `--medvram`, use  `--lowvram --opt-split-attention`.
- If you have more VRAM and want to make larger images than you can usually make (for example 1024x1024 instead of 512x512), use `--medvram --opt-split-attention`. You can use `--lowvram`
also but the effect will likely be barely noticeable.
- Otherwise, do not use any of those.

### Running online

Use `--share` option to run online. You will get a xxx.app.gradio link. This is the intended way to use the
program in collabs.

Use `--listen` to make the server listen to network connections. This will allow computers on local newtork
to access the UI, and if you configure port forwarding, also computers on the internet.

Use `--port xxxx` to make the server listen on a specific port, xxxx being the wanted port. Remember that
all ports below 1024 needs root/admin rights, for this reason it is advised to use a port above 1024.
Defaults to port 7860 if available.

### Google collab

If you don't want or can't run locally, here is google collab that allows you to run the webui:

https://colab.research.google.com/drive/1Iy-xW9t1-OQWhb0hNxueGij8phCyluOh

### Textual Inversion
To make use of pretrained embeddings, create `embeddings` directory (in the same palce as `webui.py`)
and put your embeddings into it. They must be .pt files, each with only one trained embedding,
and the filename (without .pt) will be the term you'd use in prompt to get that embedding.

As an example, I trained one for about 5000 steps: https://files.catbox.moe/e2ui6r.pt; it does not produce
very good results, but it does work. Download and rename it to Usada Pekora.pt, and put it into embeddings dir
and use Usada Pekora in prompt.

### How to change UI defaults?

After running once, a `ui-config.json` file appears in webui directory:

```json
{
    "txt2img/Sampling Steps/value": 20,
    "txt2img/Sampling Steps/minimum": 1,
    "txt2img/Sampling Steps/maximum": 150,
    "txt2img/Sampling Steps/step": 1,
    "txt2img/Batch count/value": 1,
    "txt2img/Batch count/minimum": 1,
    "txt2img/Batch count/maximum": 32,
    "txt2img/Batch count/step": 1,
    "txt2img/Batch size/value": 1,
    "txt2img/Batch size/minimum": 1,
```

Edit values to your liking and the next time you launch the program they will be applied.


### Manual instructions
Alternatively, if you don't want to run webui.bat, here are instructions for installing
everything by hand:

```commandline
:: install torch with CUDA support. See https://pytorch.org/get-started/locally/ for more instructions if this fails.
pip install torch --extra-index-url https://download.pytorch.org/whl/cu113

:: check if torch supports GPU; this must output "True". You need CUDA 11. installed for this. You might be able to use
:: a different version, but this is what I tested.
python -c "import torch; print(torch.cuda.is_available())"

:: clone web ui and go into its directory
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
cd stable-diffusion-webui

:: clone repositories for Stable Diffusion and (optionally) CodeFormer
mkdir repositories
git clone https://github.com/CompVis/stable-diffusion.git repositories/stable-diffusion
git clone https://github.com/CompVis/taming-transformers.git repositories/taming-transformers
git clone https://github.com/sczhou/CodeFormer.git repositories/CodeFormer

:: install requirements of Stable Diffusion
pip install transformers==4.19.2 diffusers invisible-watermark --prefer-binary

:: install k-diffusion
pip install git+https://github.com/crowsonkb/k-diffusion.git --prefer-binary

:: (optional) install GFPGAN (face resoration)
pip install git+https://github.com/TencentARC/GFPGAN.git --prefer-binary

:: (optional) install requirements for CodeFormer (face resoration)
pip install -r repositories/CodeFormer/requirements.txt --prefer-binary

:: install requirements of web ui
pip install -r stable-diffusion-webui/requirements.txt  --prefer-binary

:: update numpy to latest version
pip install -U numpy  --prefer-binary

:: (outside of command line) put stable diffusion model into web ui directory
:: the command below must output something like: 1 File(s) 4,265,380,512 bytes
dir model.ckpt

:: (outside of command line) put the GFPGAN model into web ui directory
:: the command below must output something like: 1 File(s) 348,632,874 bytes
dir GFPGANv1.3.pth
```

> Note: the directory structure for manual instruction has been changed on 2022-09-09 to match automatic installation: previosuly
> webui was in a subdirectory of stable diffusion, now it's the reverse. If you followed manual installation before the
> chage, you can still use the program with you existing directory sctructure.

After that the installation is finished.

Run the command to start web ui:

```
python webui.py
```

If you have a 4GB video card, run the command with either `--lowvram` or `--medvram` argument:

```
python webui.py --medvram
```

After a while, you will get a message like this:

```
Running on local URL:  http://127.0.0.1:7860/
```

Open the URL in browser, and you are good to go.


## Credits
- Stable Diffusion - https://github.com/CompVis/stable-diffusion, https://github.com/CompVis/taming-transformers
- k-diffusion - https://github.com/crowsonkb/k-diffusion.git
- GFPGAN - https://github.com/TencentARC/GFPGAN.git
- ESRGAN - https://github.com/xinntao/ESRGAN
- Ideas for optimizations and some code (from users) - https://github.com/basujindal/stable-diffusion
- Idea for SD upscale - https://github.com/jquesnelle/txt2imghd
- Initial Gradio script - posted on 4chan by an Anonymous user. Thank you Anonymous user.
- (You)


# `/opt/to-comment/stable-diffusion-webui/script.js`

It looks like you are describing an image editing tool or script with a set of parameters and options for users to use. The parameters include options for controlling denoising, interrupting the processing of images, and saving the results. The options include some for customizing the image, such as specifying the resolution andupscaling, and the ability to save the results to a file. Is there anything else you'd like to know about this tool or script?



```
titles = {
    "Sampling steps": "How many times to improve the generated image iteratively; higher values take longer; very low values can produce bad results",
    "Sampling method": "Which algorithm to use to produce the image",
	"GFPGAN": "Restore low quality faces using GFPGAN neural network",
	"Euler a": "Euler Ancestral - very creative, each can get a completely different picture depending on step count, setting steps to higher than 30-40 does not help",
	"DDIM": "Denoising Diffusion Implicit Models - best at inpainting",

	"Batch count": "How many batches of images to create",
	"Batch size": "How many image to create in a single batch",
    "CFG Scale": "Classifier Free Guidance Scale - how strongly the image should conform to prompt - lower values produce more creative results",
    "Seed": "A value that determines the output of random number generator - if you create an image with same parameters and seed as another image, you'll get the same result",

    "Inpaint a part of image": "Draw a mask over an image, and the script will regenerate the masked area with content according to prompt",
    "Loopback": "Process an image, use it as an input, repeat. Batch count determins number of iterations.",
    "SD upscale": "Upscale image normally, split result into tiles, improve each tile using img2img, merge whole image back",

    "Just resize": "Resize image to target resolution. Unless height and width match, you will get incorrect aspect ratio.",
    "Crop and resize": "Resize the image so that entirety of target resolution is filled with the image. Crop parts that stick out.",
    "Resize and fill": "Resize the image so that entirety of image is inside target resolution. Fill empty space with image's colors.",

    "Mask blur": "How much to blur the mask before processing, in pixels.",
    "Masked content": "What to put inside the masked area before processing it with Stable Diffusion.",
    "fill": "fill it with colors of the image",
    "original": "keep whatever was there originally",
    "latent noise": "fill it with latent space noise",
    "latent nothing": "fill it with latent space zeroes",
    "Inpaint at full resolution": "Upscale masked region to target resolution, do inpainting, downscale back and paste into original image",

    "Denoising strength": "Determines how little respect the algorithm should have for image's content. At 0, nothing will change, and at 1 you'll get an unrelated image.",
    "Denoising strength change factor": "In loopback mode, on each loop the denoising strength is multiplied by this value. <1 means decreasing variety so your sequence will converge on a fixed picture. >1 means increasing variety so your sequence will become more and more chaotic.",

    "Interrupt": "Stop processing images and return any results accumulated so far.",
    "Save": "Write image to a directory (default - log/images) and generation parameters into csv file.",

    "X values": "Separate values for X axis using commas.",
    "Y values": "Separate values for Y axis using commas.",

    "None": "Do not do anything special",
    "Prompt matrix": "Separate prompts into parts using vertical pipe character (|) and the script will create a picture for every combination of them (except for the first part, which will be present in all combinations)",
    "X/Y plot": "Create a grid where images will have different parameters. Use inputs below to specify which parameters will be shared by columns and rows",
    "Custom code": "Run Python code. Advanced user only. Must run program with --allow-code for this to work",

    "Prompt S/R": "Separate a list of words with commas, and the first word will be used as a keyword: script will search for this word in the prompt, and replace it with others",

    "Tiling": "Produce an image that can be tiled.",
    "Tile overlap": "For SD upscale, how much overlap in pixels should there be between tiles. Tiles overlap so that when they are merged back into one picture, there is no clearly visible seam.",

    "Roll": "Add a random artist to the prompt.",
}

```

This appears to be JavaScript code that generates a preview of a转文本图片效果，并将其宽度、高度以及宽度、高度均设置为100%。这个效果使用了tabs2img和gradio2img两个库，同时也使用了MutationObserver进行元素变化观察，以达到实时响应页面变化的效果。

具体来说，这段代码：

1. 首先，定义了两个变量，一个是进度条的ID，另一个是转文本图片效果的ID，以便在进度条变化时调用setInterval函数。

2. 接下来，定义了一个变量，用于保存当前的宽度、高度、图片对象，以便在调整图片大小时可以同步更新。

3. 然后，使用MutationObserver对象对当前的元素进行观察，以便在元素发生变化时调用相应的函数。

4. 接着，使用gradioApp.get该方法获取到两个图片元素的宽度、高度，并将它们与上面保存的变量进行同步更新。

5. 最后，将生成的效果图元素的宽度、高度、宽度和高度分别设置为100%，以达到实时响应页面变化的效果。

需要注意的是，由于使用了MutationObserver进行元素变化观察，因此只有在元素发生变化时才会调用相应的函数，否则函数不会执行。另外，由于使用了gradioApp.get该方法获取图片元素，因此只有当图片元素存在于页面中时，该方法才会返回对应的元素对象。


```
function gradioApp(){
    return document.getElementsByTagName('gradio-app')[0].shadowRoot;
}

global_progressbar = null

function addTitles(root){
	root.querySelectorAll('span, button, select').forEach(function(span){
		tooltip = titles[span.textContent];

		if(!tooltip){
		    tooltip = titles[span.value];
		}

		if(tooltip){
			span.title = tooltip;
		}
	})

	root.querySelectorAll('select').forEach(function(select){
	    if (select.onchange != null) return;

	    select.onchange = function(){
            select.title = titles[select.value] || "";
	    }
	})

	progressbar = root.getElementById('progressbar')
	if(progressbar!= null && progressbar != global_progressbar){
	    global_progressbar = progressbar

        var mutationObserver = new MutationObserver(function(m){
            txt2img_preview = gradioApp().getElementById('txt2img_preview')
            txt2img_gallery = gradioApp().getElementById('txt2img_gallery')

            img2img_preview = gradioApp().getElementById('img2img_preview')
            img2img_gallery = gradioApp().getElementById('img2img_gallery')

            if(txt2img_preview != null && txt2img_gallery != null){
                txt2img_preview.style.width = txt2img_gallery.clientWidth + "px"
                txt2img_preview.style.height = txt2img_gallery.clientHeight + "px"
            }

            if(img2img_preview != null && img2img_gallery != null){
                img2img_preview.style.width = img2img_gallery.clientWidth + "px"
                img2img_preview.style.height = img2img_gallery.clientHeight + "px"
            }


            window.setTimeout(requestProgress, 500)
        });
        mutationObserver.observe( progressbar, { childList:true, subtree:true })
	}

}

```

这段代码是一个JavaScript脚本，它的作用是观察网页上的按钮，当按钮的文本内容发生变化时，它会将相关信息存储到一个名为 "processedTabs" 的对象中。

具体来说，这段代码执行以下操作：

1. 观察 "gradioApp()" 函数中的子元素，这个函数可能用来获取 Gradio 应用程序的子元素。
2. 遍历 "gradioApp()" 中的子元素，对于每个按钮，首先检查是否已经处理过。如果是，就返回。否则，将 "processedTabs" 对象中相应的键值设置为 1，并将按钮添加到 "mask_buttons" 数组中。
3. 在点击按钮时，使用 "mask_buttons" 数组中的元素，如果数组长度为 2，那么点击第二个元素，否则再点击第一个元素。
4. 将mask_buttons数组中的元素添加到gradioApp的查询元素上。


```
tabNames =  {"txt2img": 1, "img2img": 1, "Extras": 1, "PNG Info": 1, "Settings": 1}
processedTabs = {}

document.addEventListener("DOMContentLoaded", function() {
    var mutationObserver = new MutationObserver(function(m){
        addTitles(gradioApp());

        // fix for gradio breaking when you switch away from tab with mask
        gradioApp().querySelectorAll('button').forEach(function(button){
            title = button.textContent.trim()
            if(processedTabs[title]) return
            if(tabNames[button.textContent.trim()]==null) return;
            processedTabs[title]=1

            button.onclick = function(){
                mask_buttons = gradioApp().querySelectorAll('#img2maskimg button');
                if(mask_buttons.length == 2){
                    mask_buttons[1].click();
                }
            }
        })
    });
    mutationObserver.observe( gradioApp(), { childList:true, subtree:true })


});

```

这两段代码主要作用于网页 gallery 中的图片选择。

`selected_gallery_index()` 函数用于在画廊页面中选择第一个图片并返回其索引。首先，它使用 GradioApp() 函数创建一个画廊对象 `gr`，然后使用 `querySelectorAll()` 函数选择画廊中的所有 `.gallery-item` 元素。接下来，该函数使用 `querySelector()` 函数选择画廊中与第一个 `.gallery-item` 元素具有相同 `class` 属性且包含 `!ring-2` 元素的元素。最后，该函数返回选择的照片的索引。

`extract_image_from_gallery()` 函数接收一个 `gallery` 对象，该对象应该是一个画廊对象，其中包含一个或多个图片。如果 `gallery` 对象只有一个图片，该函数返回该图片的索引。否则，该函数使用 `selected_gallery_index()` 函数获取选择的照片的索引，然后返回该索引对应的图片。


```
function selected_gallery_index(){
    var gr = gradioApp()
    var buttons = gradioApp().querySelectorAll(".gallery-item")
    var button = gr.querySelector(".gallery-item.\\!ring-2")

    var result = -1
    buttons.forEach(function(v, i){ if(v==button) { result = i } })

    return result
}

function extract_image_from_gallery(gallery){
    if(gallery.length == 1){
        return gallery[0]
    }

    index = selected_gallery_index()

    if (index < 0 || index >= gallery.length){
        return []
    }

    return gallery[index];
}


```

这段代码定义了两个函数，一个是`requestProgress()`，另一个是`submit()`。它们的主要目的是在Gradio应用程序中执行一系列用户输入的操作，并将结果返回。以下是这两个函数的作用和使用方法：

1. `requestProgress()`函数的作用是在Gradio应用程序中请求进度条的点击，以获取用户输入的反馈。该函数使用了一个Gradio应用程序的对象（通过调用`gradioApp().getElementById()`获取），并在点击按钮后执行一些操作。如果按钮无效（通过调用`gradioApp().getElementById().click()`获取），该函数将返回，并可能导致应用程序出现错误。

2. `submit()`函数的作用是在Gradio应用程序中执行一系列用户输入的操作，并将结果返回。该函数使用了一个定时器，每当用户停止操作时，它就会调用`requestProgress()`函数来请求进度条的点击。然后，它遍历用户提供的输入值，并将它们添加到结果数组中。最后，它返回结果数组，并可能导致应用程序出现错误。

请注意，这两个函数都使用了一个名为`window.setTimeout()`的JavaScript函数，该函数用于设置一个计时器，在一定时间后执行指定的代码。在这里，第一个函数设置了一个计时器，在500毫秒后执行`requestProgress()`函数。第二个函数在用户停止操作后每隔500毫秒执行一次`requestProgress()`函数，以便在用户操作过程中获取进度条的反馈。


```
function requestProgress(){
    btn = gradioApp().getElementById("check_progress");
    if(btn==null) return;

    btn.click();
}

function submit(){
    window.setTimeout(requestProgress, 500)

    res = []
    for(var i=0;i<arguments.length;i++){
        res.push(arguments[i])
    }
    return res
}

```

这段代码的作用是当用户在访问网站时从剪贴板中复制的图片时，当且仅当图片类型为 "image/png" 或 "image/gif" 时将其存储到剪贴板中的文件数组中。如果剪贴板中仅包含一个文件或文件类型不符合要求，则返回。

具体来说，代码首先检查剪贴板中是否包含一个或多个文件，并检查文件数是否为1。如果是，则进入文件检查阶段。在文件检查阶段，代码遍历剪贴板中的所有文件，并检查文件类型是否为 "image/png" 或 "image/gif"。如果是，则将文件存储到剪贴板中的 "files" 数组中，并使用 DispatchEvent 方法触发一个名为 "change"的事件，该事件将通知所有挂载了该文件的输入框（如果存在）进行更改。最后，代码使用过滤器过滤出所有允许上传的文件，并使用 forEach 方法将文件存储到输入框中。


```
window.addEventListener('paste', e => {
    const files = e.clipboardData.files;
    if (!files || files.length !== 1) {
        return;
    }
    if (!['image/png', 'image/gif', 'image/jpeg'].includes(files[0].type)) {
        return;
    }
    [...gradioApp().querySelectorAll('input[type=file][accept="image/x-png,image/gif,image/jpeg"]')]
        .filter(input => !input.matches('.\\!hidden input[type=file]'))
        .forEach(input => {
            input.files = files;
            input.dispatchEvent(new Event('change'))
        });
});

```

# `/opt/to-comment/stable-diffusion-webui/webui.py`

这段代码的作用是定义了一个名为"example_threaded_executor"的线程池执行器，用于执行一个需要使用多线程并行计算的计算任务。具体来说，它实现了以下几个功能：

1. 导入需要的模块和函数：使用os、threading、torch、numpy、Image、signal、OmegaConf、PIL、state等模块中的函数。

2. 从paths模块中定义了计算任务的脚本路径，即"example_task.py"。

3. 从定义的任务函数中提取出需要用到的数据和计算参数，包括输入数据、输出数据、前文训练好的权重、计算图等。

4. 实例化一个计算任务类，这个类的实现将继承自ThreadingScheduledExecutor类，它将负责调度任务、设置任务执行条件、处理任务执行过程中的异常等。

5. 在ThreadingScheduledExecutor类的构造函数中，实现了接受一个executor_opts对象作为参数，这个参数包含了executor的配置选项，如max_workers、remote_printing等。

6. 在executor的函数体中，实现了异步/同步执行任务，使用了多线程并行计算，从而实现高效的计算任务。

7. 在执行任务前，对输入数据进行处理，包括数据预处理、标准化等。

8. 在任务执行过程中，定时打印输出数据，便于监控任务执行情况。

9. 使用OmegaConf和torch的API，实现任务参数的设置和取消，包括设置任务优先级、设置任务执行定时器、取消任务等。

10. 从LDM.util模块中，使用instantiate_from_config函数，加载了计算任务的配置文件，包括配置文件中定义的选项和参数。


```
import os
import threading

from modules.paths import script_path

import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image

import signal

from ldm.util import instantiate_from_config

from modules.shared import opts, cmd_opts, state
```

这段代码的主要作用是执行一系列图像处理和生成任务。具体来说，它包括以下几个模块：

1. 导入一些外部模块，如`shared`模块中的图像处理和文本转换函数，以及`processing`和`scripts`模块。
2. 从`ui`模块中导入一个名为`plaintext_to_html`的函数，将文本转换为HTML格式。
3. 从`ui`模块中导入一个名为`images`的函数，用于处理和生成图像。
4. 从`shared`模块中导入`codeformer_model`、`esrgan_model`和`face_restoration`三个模型，以及一个名为`realesrgan_model`的模型，这些模型都是用于生成人脸图像的。
5. 在导入的模型之后，分别对文本和图像进行处理和生成，包括：
   - `processing.plaintext_to_html`函数将文本转换为HTML格式；
   - `images.generate_image`函数生成图像；
   - 对文本进行处理，包括：
       - `processing.codeformer_model.plaintext_to_speech`函数将文本转换为语音；
       - `processing.codeformer_model.text_to_speech`函数将文本转换为音频；
       - `processing.codeformer_model.audioplayer`函数播放生成的音频。
   - 对图像进行处理和生成，包括：
       - `images.generate_image`函数生成图像；
       - `images.resize_image`函数调整图像大小；
       - `images.rotate_image`函数旋转图像；
       - `images.flip_image`函数颠倒图像；
       - `images.nearest_neighbor`函数以距离最近的邻居为原点生成图像；
       - `images.generative_algorithm`函数应用生成算法生成图像。


```
import modules.shared as shared
import modules.ui
from modules.ui import plaintext_to_html
import modules.scripts
import modules.processing as processing
import modules.sd_hijack
import modules.codeformer_model
import modules.gfpgan_model
import modules.face_restoration
import modules.realesrgan_model as realesrgan
import modules.esrgan_model as esrgan
import modules.images as images
import modules.lowvram
import modules.txt2img
import modules.img2img


```

这段代码的主要作用是加载两个预训练的模型，并设置一个新的GAN模型。其中，第一个模型是来自EfficientSRGAN的，第二个模型是来自DeepSRGAN的。

代码首先通过`modules.codeformer_model.setup_codeformer()`和`modules.gfpgan_model.setup_gfpgan()`函数加载了这两个模型。接着，通过`shared.face_restorers.append(modules.face_restoration.FaceRestoration())`将一个FaceRestoration模型添加到共享恢复器中，这个模型将在后面的生成过程中使用，用于对输入图像中的人脸进行修复。

接下来，代码通过`esrgan.load_models(cmd_opts.esrgan_models_path)`加载了一个名为`esrgan.model`的模型，并使用`realsrgan.setup_realsrgan()`设置了一个名为`realsrgan`的模型。

最后，代码通过`load_model_from_config()`函数加载了一个指定的模型配置文件，并使用`instantiate_from_config()`函数加载了这个配置文件中的模型。如果配置文件存在错误或未配置，则会输出相应的错误信息。

整个代码的主要作用是加载两个预训练的模型，设置一个新的GAN模型，并将FaceRestoration模型添加到共享恢复器中。


```
modules.codeformer_model.setup_codeformer()
modules.gfpgan_model.setup_gfpgan()
shared.face_restorers.append(modules.face_restoration.FaceRestoration())

esrgan.load_models(cmd_opts.esrgan_models_path)
realesrgan.setup_realesrgan()

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]

    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.eval()
    return model

```

This is a function that uses a code transformer model to modify an image. The function takes an input image and modifies it using a code transformer model that has been trained on a variety of images.

The function first restores an image from a code transformer model, and then converts it to a PIL Image object. If the code transformer model has been configured to have a visibility of 1.0, the image is converted to a grayscale image.

If the `upscaling_resize` parameter is set to 1.0, the image is first upscaled to a smaller size, and then the upscale factor is applied. Finally, the modified image is saved to disk using the `Image.save_image` method.

The function also includes a `next(iter(cached_images.keys()))` line to remove the first image from the `cached_images` dictionary, which is called automatically by the function when it is called. This is done to prevent the function from returning the first image in the `cached_images` dictionary.


```
cached_images = {}


def run_extras(image, gfpgan_visibility, codeformer_visibility, codeformer_weight, upscaling_resize, extras_upscaler_1, extras_upscaler_2, extras_upscaler_2_visibility):
    processing.torch_gc()

    image = image.convert("RGB")

    outpath = opts.outdir_samples or opts.outdir_extras_samples

    if gfpgan_visibility > 0:
        restored_img = modules.gfpgan_model.gfpgan_fix_faces(np.array(image, dtype=np.uint8))
        res = Image.fromarray(restored_img)

        if gfpgan_visibility < 1.0:
            res = Image.blend(image, res, gfpgan_visibility)

        image = res

    if codeformer_visibility > 0:
        restored_img = modules.codeformer_model.codeformer.restore(np.array(image, dtype=np.uint8), w=codeformer_weight)
        res = Image.fromarray(restored_img)

        if codeformer_visibility < 1.0:
            res = Image.blend(image, res, codeformer_visibility)

        image = res

    if upscaling_resize != 1.0:
        def upscale(image, scaler_index, resize):
            small = image.crop((image.width // 2, image.height // 2, image.width // 2 + 10, image.height // 2 + 10))
            pixels = tuple(np.array(small).flatten().tolist())
            key = (resize, scaler_index, image.width, image.height, gfpgan_visibility, codeformer_visibility, codeformer_weight) + pixels

            c = cached_images.get(key)
            if c is None:
                upscaler = shared.sd_upscalers[scaler_index]
                c = upscaler.upscale(image, image.width * resize, image.height * resize)
                cached_images[key] = c

            return c

        res = upscale(image, extras_upscaler_1, upscaling_resize)

        if extras_upscaler_2 != 0 and extras_upscaler_2_visibility>0:
            res2 = upscale(image, extras_upscaler_2, upscaling_resize)
            res = Image.blend(res, res2, extras_upscaler_2_visibility)

        image = res

    while len(cached_images) > 2:
        del cached_images[next(iter(cached_images.keys()))]

    images.save_image(image, outpath, "", None, '', opts.samples_format, short_filename=True, no_prompt=True)

    return image, '', ''


```

这段代码定义了一个名为 `run_pnginfo` 的函数，它接受一个 `Pillow` 图像对象（image）作为参数。

函数的主要作用是获取图像的元数据（如拍摄日期、相机型号等）信息，并将这些信息以字符串形式返回。

具体实现包括以下几个步骤：

1. 从图像的元数据中提取信息。
2. 将信息以字符串格式进行排版。
3. 如果元数据字符串为空，则输出一条消息。
4. 返回元数据信息、字符串和信息。


```
def run_pnginfo(image):
    info = ''
    for key, text in image.info.items():
        info += f"""
<div>
<p><b>{plaintext_to_html(str(key))}</b></p>
<p>{plaintext_to_html(str(text))}</p>
</div>
""".strip()+"\n"

    if len(info) == 0:
        message = "Nothing found in the image."
        info = f"<div><p>{message}<p></div>"

    return '', '', info


```

这段代码的作用是创建一个名为 `queue_lock` 的线程安全锁，用于在多个 GPU 线程之间同步执行任务。

函数 `wrap_gradio_gpu_call` 是一个 Python 函数，它使用 `threading.Lock` 类创建一个锁，然后通过调用另一个函数 `wrap_gradio_call` 压入锁中，从而确保在每次运行时，每个 GPU 线程都会获取到锁。

函数 `wrap_gradio_call` 接收一个函数作为参数，并返回一个新的函数，该函数在每次运行时会获取到锁并执行指定的函数。这个新函数 `f` 通过在函数内部使用 `with queue_lock` 语句来获取锁，并在获取到锁后执行指定的函数。当函数返回时，它将释放锁并返回结果，这样下一个运行时的函数就可以访问到锁并继续执行。


```
queue_lock = threading.Lock()


def wrap_gradio_gpu_call(func):
    def f(*args, **kwargs):
        shared.state.sampling_step = 0
        shared.state.job_count = -1
        shared.state.job_no = 0
        shared.state.current_latent = None
        shared.state.current_image = None
        shared.state.current_image_sampling_step = 0

        with queue_lock:
            res = func(*args, **kwargs)

        shared.state.job = ""
        shared.state.job_count = 0

        return res

    return modules.ui.wrap_gradio_call(f)

```

这段代码是一个Python脚本，它的作用是加载一个名为"scripts"的文件夹中的所有脚本，并将其存储到一个名为"modules.scripts"的文件夹中。

在代码中，首先通过os.path.join(script_path, "scripts")获取了脚本文件的完整路径，并将它作为参数传递给模块的一个名为load_scripts的函数。

接着，代码尝试从transformers库中导入日志功能，以便在初始化模型时忽略一个警告消息。

然后，代码捕获了一个异常，并在异常发生时跳过日志设置。

接下来，代码使用load_model_from_config函数加载了一个名为sd_config的配置文件，并将其存储到了shared.sd_model变量中。

接着，代码使用if语句检查是否设置了cmd_opts.no_half参数，如果是，则只加载模型的 half部分，否则加载模型的完整部分。

最后，代码将shared.sd_model变量复制到sd_config中指定的位置，并返回了sd_config。


```
modules.scripts.load_scripts(os.path.join(script_path, "scripts"))

try:
    # this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.

    from transformers import logging

    logging.set_verbosity_error()
except Exception:
    pass

sd_config = OmegaConf.load(cmd_opts.config)
shared.sd_model = load_model_from_config(sd_config, cmd_opts.ckpt)
shared.sd_model = (shared.sd_model if cmd_opts.no_half else shared.sd_model.half())

```

这段代码是一个Python脚本，主要作用是实现一个GPU模型的Hijack和SD模型的设置。

首先，判断两个选项lowvram和medvram，如果其中一个存在则设置lowvram模块的参数，否则将shared.sd_model的值设置为shared.device。

接着，调用shared.sd_model.model_hijack.hijack函数对shared.sd_model进行Hijack。

最后，创建一个WebUI，接收用户输入并返回一个画布图像，作为输入的demo模型是通过对GPU模型的Hijack得到的。


```
if cmd_opts.lowvram or cmd_opts.medvram:
    modules.lowvram.setup_for_low_vram(shared.sd_model, cmd_opts.medvram)
else:
    shared.sd_model = shared.sd_model.to(shared.device)

modules.sd_hijack.model_hijack.hijack(shared.sd_model)


def webui():
    # make the program just exit at ctrl+c without waiting for anything
    def sigint_handler(sig, frame):
        print(f'Interrupted with signal {sig} in {frame}')
        os._exit(0)

    signal.signal(signal.SIGINT, sigint_handler)

    demo = modules.ui.create_ui(
        txt2img=wrap_gradio_gpu_call(modules.txt2img.txt2img),
        img2img=wrap_gradio_gpu_call(modules.img2img.img2img),
        run_extras=wrap_gradio_gpu_call(run_extras),
        run_pnginfo=run_pnginfo
    )

    demo.launch(share=cmd_opts.share, server_name="0.0.0.0" if cmd_opts.listen else None, server_port=cmd_opts.port)

```

这段代码是一个 if 语句，判断当前程序是否作为主程序运行。如果程序作为主程序运行，则会执行 if 语句块内的代码。在这段代码中，我们使用 __name__ 作为程序名称，因为程序是在 Python 环境下运行，而 __name__ 是在 Python 2.x 中定义为 `__main__` 的。

if __name__ == "__main__":
   webui()
这段代码的作用是：如果程序作为主程序运行，那么程序将执行 if 语句块内的代码。在这里，我们使用 webui() 函数作为程序的入口点。webui() 函数是一个模块，可能是一个 Web 框架（如 Django、Flask 等）中的入口函数，用于启动 Web 应用程序。因此，如果程序作为主程序运行，那么它将启动 Web 应用程序。


```
if __name__ == "__main__":
    webui()

```

# `/opt/to-comment/stable-diffusion-webui/modules/artists.py`

这段代码定义了一个名为 `ArtistsDatabase` 的类，用于保存艺术家及其得分的数据。这个类包含两个方法：`__init__` 和 `categories`。

`__init__` 方法用于初始化数据库文件，如果文件不存在，会抛出异常。这个方法首先创建一个空集合 `self.cats` 和一个空列表 `self.artists`，然后打开文件并逐行读取。对于每一行数据，它将解析出艺术家对象并将其添加到 `self.artists` 列表中，并将该艺术家的 `category` 属性添加到 `self.cats` 集合中。

`categories` 方法用于获取所有艺术家所属的类别，并将其排序。这个方法首先获取 `self.cats` 集合中的所有元素，然后使用 Python 的 `sorted` 函数对它们进行排序。

值得注意的是，这个类没有定义任何函数来读取或写入文件，因为它只是简单地将读取和写入的代码集成在一起。


```
import os.path
import csv
from collections import namedtuple

Artist = namedtuple("Artist", ['name', 'weight', 'category'])


class ArtistsDatabase:
    def __init__(self, filename):
        self.cats = set()
        self.artists = []

        if not os.path.exists(filename):
            return

        with open(filename, "r", newline='', encoding="utf8") as file:
            reader = csv.DictReader(file)

            for row in reader:
                artist = Artist(row["artist"], float(row["score"]), row["category"])
                self.artists.append(artist)
                self.cats.add(artist.category)

    def categories(self):
        return sorted(self.cats)

```

# `/opt/to-comment/stable-diffusion-webui/modules/codeformer_model.py`

这段代码的作用是引入一些常用的Python库和模块，用于在基于PyTorch的图像修复应用程序中进行face_restoration模型的开发和测试。具体来说，它包括以下几个主要部分：

1. 导入必要的库：os、sys、traceback、torch、modules、shared、script_path、modules_shared和face_restoration。

2. 导入模块：modules.import_path和reload。

3. 设置环境：import os，以便在运行时检查当前工作目录。

4. 导入自定义模块：shared。

5. 导入代码：from modules.paths import script_path，以便在运行时获取脚本路径。

6. 导入自定义类：modules.shared。

7. 导入自定义函数：modules.face_restoration。

8. 从PyTorch中导入reload，以便在需要时动态加载和卸载相对应的模块。


```
import os
import sys
import traceback
import torch

from modules import shared
from modules.paths import script_path
import modules.shared
import modules.face_restoration
from importlib import reload

# codeformer people made a choice to include modified basicsr librry to their projectwhich makes
# it utterly impossiblr to use it alongside with other libraries that also use basicsr, like GFPGAN.
# I am making a choice to include some files from codeformer to work around this issue.

```

This is a Python implementation of a face restoration model using the CodeFormer architecture that utilizes the Hosted Faster R-CNN model. The `FaceRestorer` class is responsible for processing the input face images, resizing them, normalizing them, and converting them to the required format for the CodeFormer model.

The model takes as input a cropped face image and outputs a restored face image. The input image is first converted to a tensor, then resized to a smaller size, normalized, and added to a shared variable called `shared.device`. This allows the model to be performed on a multi-device machine.

The model also includes a code-based pre-training step using the pre-trained CodeFormer model, which is likely done to improve the quality of the restoration process.

Finally, the output image is returned, having been processed, normalized, and converted to the required format for the CodeFormer model.


```
pretrain_model_url = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth'

have_codeformer = False
codeformer = None

def setup_codeformer():
    path = modules.paths.paths.get("CodeFormer", None)
    if path is None:
        return


    # both GFPGAN and CodeFormer use bascisr, one has it installed from pip the other uses its own
    #stored_sys_path = sys.path
    #sys.path = [path] + sys.path

    try:
        from torchvision.transforms.functional import normalize
        from modules.codeformer.codeformer_arch import CodeFormer
        from basicsr.utils.download_util import load_file_from_url
        from basicsr.utils import imwrite, img2tensor, tensor2img
        from facelib.utils.face_restoration_helper import FaceRestoreHelper
        from modules.shared import cmd_opts

        net_class = CodeFormer

        class FaceRestorerCodeFormer(modules.face_restoration.FaceRestoration):
            def name(self):
                return "CodeFormer"

            def __init__(self):
                self.net = None
                self.face_helper = None

            def create_models(self):

                if self.net is not None and self.face_helper is not None:
                    return self.net, self.face_helper

                net = net_class(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=['32', '64', '128', '256']).to(shared.device)
                ckpt_path = load_file_from_url(url=pretrain_model_url, model_dir=os.path.join(path, 'weights/CodeFormer'), progress=True)
                checkpoint = torch.load(ckpt_path)['params_ema']
                net.load_state_dict(checkpoint)
                net.eval()

                face_helper = FaceRestoreHelper(1, face_size=512, crop_ratio=(1, 1), det_model='retinaface_resnet50', save_ext='png', use_parse=True, device=shared.device)

                if not cmd_opts.unload_gfpgan:
                    self.net = net
                    self.face_helper = face_helper

                return net, face_helper

            def restore(self, np_image, w=None):
                np_image = np_image[:, :, ::-1]

                net, face_helper = self.create_models()
                face_helper.clean_all()
                face_helper.read_image(np_image)
                face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
                face_helper.align_warp_face()

                for idx, cropped_face in enumerate(face_helper.cropped_faces):
                    cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
                    normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                    cropped_face_t = cropped_face_t.unsqueeze(0).to(shared.device)

                    try:
                        with torch.no_grad():
                            output = net(cropped_face_t, w=w if w is not None else shared.opts.code_former_weight, adain=True)[0]
                            restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                        del output
                        torch.cuda.empty_cache()
                    except Exception as error:
                        print(f'\tFailed inference for CodeFormer: {error}', file=sys.stderr)
                        restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

                    restored_face = restored_face.astype('uint8')
                    face_helper.add_restored_face(restored_face)

                face_helper.get_inverse_affine(None)

                restored_img = face_helper.paste_faces_to_input_image()
                restored_img = restored_img[:, :, ::-1]
                return restored_img

        global have_codeformer
        have_codeformer = True

        global codeformer
        codeformer = FaceRestorerCodeFormer()
        shared.face_restorers.append(codeformer)

    except Exception:
        print("Error setting up CodeFormer:", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)

   # sys.path = stored_sys_path

```

# `/opt/to-comment/stable-diffusion-webui/modules/esrgam_model_arch.py`

这段代码是一个ESRGAN模型的实现，ESRGAN是一种基于自注意力机制的图像生成模型，出自于github.com/xinntao/ESRGAN。主要作用是定义一个ESRGAN模型，通过make_layer函数可以定义模型的层数和每个层的具体组件。具体来说，这段代码定义了一个名为ESRGAN模型的类，其中`make_layer`函数定义了层数和每个层的组件。通过使用`import torch`和`import torch.nn`，可以实现对模型的引入和定义。


```
# this file is taken from https://github.com/xinntao/ESRGAN

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


```

This is a PyTorch implementation of a `ResidualDenseBlock` layer. It takes `5C` parameters: `nf` (number of feature maps), `gc` (number of channels), and `bias` (whether to add a learnable bias).

The block has a Forward pass that applies a `LeakyReLU` activation function to the intermediate outputs of the `Conv2d` layers. The input `x` is first passed through the first `Conv2d` layer with a `3x3` kernel size and a single channel. The output from this layer is then passed through a `LeakyReLU` activation function and fed into the second `Conv2d` layer, which has a similar configuration but with two channels.

The output from the second `Conv2d` layer is then passed through a `LeakyReLU` activation function and fed into the third `Conv2d` layer, which also has a similar configuration but with three channels. The output from this layer is then passed through a `LeakyReLU` activation function and fed into the `Conv2d` layer with 5 channels, which has a similar configuration but with fewer feature maps.

Finally, the output of the `Conv2d` layer is multiplied by 0.2 and added back to the input `x` to produce the final output.


```
class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


```

这段代码定义了一个名为 RRDB 的类，继承自 PyTorch 中的 nn.Module 类。这个类实现了一个在 Residual Dense Block (RDB) 中添加 Residual 的功能。

具体来说，这个类的初始化函数包含三个 Residual Dense Block 层，分别命名为 RDB1、RDB2 和 RDB3。在 forward 函数中，对输入的 x 应用这三个 Residual Dense Block，并将它们的结果与 x 上的零乘以 0.2 之后相加，最终得到输出。

RRDB 的主要作用是将被训练的模型（在代码中没有定义）抵抗大小为 1 的攻击，即在需要保护的关键区域（在代码中没有定义）添加了 Residual，以便在面临深度伪造攻击时能够起到一定的作用。


```
class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


```

This is a Python implementation of a RRDNet model. RRDNet is a deep residual network that uses a relative residual block (RRDB) to improve the speed and accuracy of the model.

It takes as input a feature tensor `x`, and as output a tensor `out` that contains the predictions.

The model has a positive入境到 ResNet 的第一个隐藏层，RRDB 的三个隐藏层和一个 HR 层。

它包含一些自定义的卷积层和激活函数，以便于适应不同的输入大小和分辨率。

RRDNet 在 2021年的 ImageNet 挑战中获得了分类最佳成绩，是目前最先进的深度学习模型之一。


```
class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out

```

# `/opt/to-comment/stable-diffusion-webui/modules/esrgan_model.py`

This is a function that creates a CRTNet model from a pre-trained network and applies some transformations to it. The function takes a list of transformations as an input and returns a CRTNet model.

The function first removes the elements in the list of transformations that have the string 'RRDB\_trunk.', and then apply the transformations to the corresponding elements in the list of transformations.

It then creates a sub-module of the CRTNet model with the name 'model.1' and assigns the weight of the first sub-module to the variable 'crt\_net'. It also assigns the bias of the first sub-module to the variable 'crt\_net'.

It then iterates through the elements of the list of transformations and removes the corresponding elements from the list of transformations.

It finally creates the weights and biases of the sub-module of the CRTNet model and assigns it to the variable 'crt\_net'.

The function also applies some transformations to the convolutional layers of the CRTNet model. These transformations include learning a variable number of convolutional layers with different numbers of filters, and replace the权重 and bias of the convolutional layers with the corresponding values from the pre-trained network.

It finally loads the state of the CRTNet model from the pre-trained network and returns it.


```
import os
import sys
import traceback

import numpy as np
import torch
from PIL import Image

import modules.esrgam_model_arch as arch
from modules import shared
from modules.shared import opts
import modules.images


def load_model(filename):
    # this code is adapted from https://github.com/xinntao/ESRGAN
    pretrained_net = torch.load(filename, map_location='cpu' if torch.has_mps else None)
    crt_model = arch.RRDBNet(3, 3, 64, 23, gc=32)

    if 'conv_first.weight' in pretrained_net:
        crt_model.load_state_dict(pretrained_net)
        return crt_model

    if 'model.0.weight' not in pretrained_net:
        is_realesrgan = "params_ema" in pretrained_net and 'body.0.rdb1.conv1.weight' in pretrained_net["params_ema"]
        if is_realesrgan:
            raise Exception("The file is a RealESRGAN model, it can't be used as a ESRGAN model.")
        else:
            raise Exception("The file is not a ESRGAN model.")

    crt_net = crt_model.state_dict()
    load_net_clean = {}
    for k, v in pretrained_net.items():
        if k.startswith('module.'):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    pretrained_net = load_net_clean

    tbd = []
    for k, v in crt_net.items():
        tbd.append(k)

    # directly copy
    for k, v in crt_net.items():
        if k in pretrained_net and pretrained_net[k].size() == v.size():
            crt_net[k] = pretrained_net[k]
            tbd.remove(k)

    crt_net['conv_first.weight'] = pretrained_net['model.0.weight']
    crt_net['conv_first.bias'] = pretrained_net['model.0.bias']

    for k in tbd.copy():
        if 'RDB' in k:
            ori_k = k.replace('RRDB_trunk.', 'model.1.sub.')
            if '.weight' in k:
                ori_k = ori_k.replace('.weight', '.0.weight')
            elif '.bias' in k:
                ori_k = ori_k.replace('.bias', '.0.bias')
            crt_net[k] = pretrained_net[ori_k]
            tbd.remove(k)

    crt_net['trunk_conv.weight'] = pretrained_net['model.1.sub.23.weight']
    crt_net['trunk_conv.bias'] = pretrained_net['model.1.sub.23.bias']
    crt_net['upconv1.weight'] = pretrained_net['model.3.weight']
    crt_net['upconv1.bias'] = pretrained_net['model.3.bias']
    crt_net['upconv2.weight'] = pretrained_net['model.6.weight']
    crt_net['upconv2.bias'] = pretrained_net['model.6.bias']
    crt_net['HRconv.weight'] = pretrained_net['model.8.weight']
    crt_net['HRconv.bias'] = pretrained_net['model.8.bias']
    crt_net['conv_last.weight'] = pretrained_net['model.10.weight']
    crt_net['conv_last.bias'] = pretrained_net['model.10.bias']

    crt_model.load_state_dict(crt_net)
    crt_model.eval()
    return crt_model

```

这段代码定义了一个名为 `upscale_without_tiling` 的函数，它接受两个参数：`model` 和 `img`。

首先，将 `img` 图像转换为 NumPy 数组，然后将图像的第二维（即第一维）向前移动一位，使得所有元素都变成原来的相反数。然后将图像从 NumPy 数组转换为 PyTorch 数组，并将其存储在变量中。

接着，将 PyTorch 数组 `img` 传递给 `model` 函数中进行前向传递，得到一个张量 `output`。然后将 `output` 数组的下标从 0 移动到 1，使得所有元素都变成原来的相反数，并且只取值在 0 到 1 之间。最后，将 `output` 数组转换为 np.uint8 类型，并将结果存储在变量中。

该函数的作用是实现图像的放大，而不会对图像进行插值或者tiling等操作。


```
def upscale_without_tiling(model, img):
    img = np.array(img)
    img = img[:, :, ::-1]
    img = np.moveaxis(img, 2, 0) / 255
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0).to(shared.device)
    with torch.no_grad():
        output = model(img)
    output = output.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = 255. * np.moveaxis(output, 0, 2)
    output = output.astype(np.uint8)
    output = output[:, :, ::-1]
    return Image.fromarray(output, 'RGB')


```

这段代码定义了一个名为 `esrgan_upscale` 的函数，它接受两个参数：`model` 和 `img`。

函数内部首先判断 `opts.ESRGAN_tile` 是否为0，如果是，则执行 `upscale_without_tiling` 函数，如果不是，则执行 `esrgan_upscale` 函数。

接下来，函数内部创建一个包含网格数据的列表 `grid`，该列表包含一个二维列表，每个元素包含一个图像的行、列、以及包含在网格中的数据。接下来，函数内部遍历 grid 中的每个元素，并将每个元素的信息存储在一个新的列表 `newrow` 中。

最后，函数内部创建一个新的 grid，该 grid 包含 newrow 中所有的元素，并使用 `combine_grid` 函数将它们组合成一个图像，然后返回该图像。


```
def esrgan_upscale(model, img):
    if opts.ESRGAN_tile == 0:
        return upscale_without_tiling(model, img)

    grid = modules.images.split_grid(img, opts.ESRGAN_tile, opts.ESRGAN_tile, opts.ESRGAN_tile_overlap)
    newtiles = []
    scale_factor = 1

    for y, h, row in grid.tiles:
        newrow = []
        for tiledata in row:
            x, w, tile = tiledata

            output = upscale_without_tiling(model, tile)
            scale_factor = output.width // tile.width

            newrow.append([x * scale_factor, w * scale_factor, output])
        newtiles.append([y * scale_factor, h * scale_factor, newrow])

    newgrid = modules.images.Grid(newtiles, grid.tile_w * scale_factor, grid.tile_h * scale_factor, grid.image_w * scale_factor, grid.image_h * scale_factor, grid.overlap * scale_factor)
    output = modules.images.combine_grid(newgrid)
    return output


```

这段代码定义了一个名为 `UpscalerESRGAN` 的类，继承自 `modules.images.Upscaler`。在类的初始化方法 `__init__` 中，传入了两个参数 `filename` 和 `title`，分别表示要加载的图像文件和图像的标题。接着，加载并返回图像模型的 `to` 方法，实现图像的升缩处理。

该代码的目的是实现将指定图像文件升缩到指定设备的 `ESRGAN` 模型。首先读取图像文件的路径和文件名，然后判断文件类型是否为`.pt`或`.pth`，如果不是，则加载并返回图像模型的 `UpscalerESRGAN` 实例。如果加载过程中出现错误，则输出错误信息并抛出异常。


```
class UpscalerESRGAN(modules.images.Upscaler):
    def __init__(self, filename, title):
        self.name = title
        self.model = load_model(filename)

    def do_upscale(self, img):
        model = self.model.to(shared.device)
        img = esrgan_upscale(model, img)
        return img


def load_models(dirname):
    for file in os.listdir(dirname):
        path = os.path.join(dirname, file)
        model_name, extension = os.path.splitext(file)

        if extension != '.pt' and extension != '.pth':
            continue

        try:
            modules.shared.sd_upscalers.append(UpscalerESRGAN(path, model_name))
        except Exception:
            print(f"Error loading ESRGAN model: {path}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)

```

# `/opt/to-comment/stable-diffusion-webui/modules/face_restoration.py`

这段代码定义了一个名为 `FaceRestoration` 的类，该类有两个方法，一个名为 `name`，另一个名为 `restore`。

`name` 方法返回对象的名称，但在此处没有使用它。

`restore` 方法接受一个NumPy图像对象作为参数，并返回原始图像。但是，为了使代码可读性更好，该方法没有实现任何真正的功能。它只是一个类的实例，没有具体的逻辑。

另外，该代码中还有一个名为 `restore_faces` 的函数，它接收一个NumPy图像对象作为参数，并使用共享模块中的 `face_restorers` 列表来查找最合适的面部修复模型。如果找到了合适的模型，它将修复 face，并返回修复后的图像。如果 `face_restorers` 列表为空，它将直接返回原始图像。

修复后的图像可以比原始图像更清晰，但仍然可能存在一些损坏或不准确的问题。


```
from modules import shared


class FaceRestoration:
    def name(self):
        return "None"

    def restore(self, np_image):
        return np_image


def restore_faces(np_image):
    face_restorers = [x for x in shared.face_restorers if x.name() == shared.opts.face_restoration_model or shared.opts.face_restoration_model is None]
    if len(face_restorers) == 0:
        return np_image

    face_restorer = face_restorers[0]

    return face_restorer.restore(np_image)

```

# `/opt/to-comment/stable-diffusion-webui/modules/gfpgan_model.py`

这段代码的作用是调用了`GFPGAN`模型的预训练模型，并返回了该模型的文件路径。

具体来说，代码首先导入了`os`、`sys`和`traceback`模块，然后从`modules`目录中导入`shared`模块，从`paths`函数中导入了`cmd_opts`模块，并从`script_path`函数中导入了`face_restoration`模块。接下来，定义了一个`gfpgan_model_path`函数，该函数调用了`cmd_opts.gfpgan_dir`目录下的预训练模型，并返回了该模型的文件路径。

在函数内部，首先创建了一个包含三个路径元素的列表`places`，然后遍历`places`列表中的每个路径，使用`os.path.join`方法构建一个包含所有路径和模型的字符串`files`。接着，使用列表推导式遍历所有文件，并使用`if`语句检查是否找到了预训练模型，如果找到了，则返回该模型的文件路径，否则抛出一个异常。

最后，需要注意，如果`GFPGAN`模型文件不存在，则函数将抛出一个异常，并打印错误信息。


```
import os
import sys
import traceback

from modules import shared
from modules.shared import cmd_opts
from modules.paths import script_path
import modules.face_restoration


def gfpgan_model_path():
    from modules.shared import cmd_opts

    places = [script_path, '.', os.path.join(cmd_opts.gfpgan_dir, 'experiments/pretrained_models')]
    files = [cmd_opts.gfpgan_model] + [os.path.join(dirname, cmd_opts.gfpgan_model) for dirname in places]
    found = [x for x in files if os.path.exists(x)]

    if len(found) == 0:
        raise Exception("GFPGAN model not found in paths: " + ", ".join(files))

    return found[0]


```

这段代码是一个名为 `gfpgan` 的函数，它返回一个经过训练的 GFPGAN 模型的实例，即使训练尚未完成。

具体来说，代码的作用如下：

1. 定义了一个名为 `gfpgan` 的函数，该函数使用 `gfpgan_constructor` 函数来加载预训练的 GFPGAN 模型。

2. 如果已经加载了 GFPGAN 模型，函数将直接返回该模型。

3. 如果 `gfpgan_constructor` 函数为空，函数将返回 `None`，表示没有加载到模型。

4. 训练过程中，如果 `unload_gfpgan` 参数为 `False`，函数将保留已经加载的模型，否则会将其释放。

5. `gfpgan` 函数的具体实现主要依赖于 `gfpgan_constructor` 函数，该函数调用了 GFPGAN 模型的加载和预训练过程，将训练好的模型存储在 `loaded_gfpgan_model` 变量中，并在训练开始时将其返回。


```
loaded_gfpgan_model = None


def gfpgan():
    global loaded_gfpgan_model

    if loaded_gfpgan_model is not None:
        return loaded_gfpgan_model

    if gfpgan_constructor is None:
        return None

    model = gfpgan_constructor(model_path=gfpgan_model_path(), upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None)

    if not cmd_opts.unload_gfpgan:
        loaded_gfpgan_model = model

    return model


```

这段代码定义了一个名为 `gfpgan_fix_faces` 的函数，它对输入的灰度图像进行处理，以修复人脸并返回处理后的图像。

首先，函数接收一个灰度图像作为输入参数。然后，函数将其转换为 BGR 格式，即彩色图像。接下来，函数调用一个名为 `gfpgan()` 的函数，这个函数可能对图像进行增强，以修复人脸。如果 `has_aligned` 参数为 `False`，并且 `only_center_face` 参数为 `False`，那么函数将尝试使用图像的左、上、中心区域作为人脸的位置，而不是使用原始图像的坐标。最后，如果 `paste_back` 参数为 `True`，那么函数将在修复人脸之后将图像的右下角部分复制粘贴到左上角区域，以保持图像的尺寸和形状。

函数返回处理后的图像，它的 BGR 通道被翻转到了原来的 RGB 通道的后面。这意味着，与原始图像相比，函数返回的图像的第四个通道（即 RGB）被截断了，只有第一个、第二个和第三个通道保留。




```
def gfpgan_fix_faces(np_image):
    np_image_bgr = np_image[:, :, ::-1]
    cropped_faces, restored_faces, gfpgan_output_bgr = gfpgan().enhance(np_image_bgr, has_aligned=False, only_center_face=False, paste_back=True)
    np_image = gfpgan_output_bgr[:, :, ::-1]

    return np_image


have_gfpgan = False
gfpgan_constructor = None

def setup_gfpgan():
    try:
        gfpgan_model_path()

        if os.path.exists(cmd_opts.gfpgan_dir):
            sys.path.append(os.path.abspath(cmd_opts.gfpgan_dir))
        from gfpgan import GFPGANer

        global have_gfpgan
        have_gfpgan = True

        global gfpgan_constructor
        gfpgan_constructor = GFPGANer

        class FaceRestorerGFPGAN(modules.face_restoration.FaceRestoration):
            def name(self):
                return "GFPGAN"

            def restore(self, np_image):
                np_image_bgr = np_image[:, :, ::-1]
                cropped_faces, restored_faces, gfpgan_output_bgr = gfpgan().enhance(np_image_bgr, has_aligned=False, only_center_face=False, paste_back=True)
                np_image = gfpgan_output_bgr[:, :, ::-1]

                return np_image

        shared.face_restorers.append(FaceRestorerGFPGAN())
    except Exception:
        print("Error setting up GFPGAN:", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)

```
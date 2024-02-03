# `stable-diffusion-webui\extensions-builtin\canvas-zoom-and-pan\javascript\zoom.js`

```py
// 当页面加载完成时执行异步函数
onUiLoaded(async() => {
    // 定义页面元素的 ID
    const elementIDs = {
        img2imgTabs: "#mode_img2img .tab-nav",
        inpaint: "#img2maskimg",
        inpaintSketch: "#inpaint_sketch",
        rangeGroup: "#img2img_column_size",
        sketch: "#img2img_sketch"
    };
    // 将选项卡名称映射到对应的元素 ID
    const tabNameToElementId = {
        "Inpaint sketch": elementIDs.inpaintSketch,
        "Inpaint": elementIDs.inpaint,
        "Sketch": elementIDs.sketch
    };

    // 辅助函数
    // 获取当前活动的选项卡

    /**
     * 等待 DOM 中出现一个元素。
     */
    const waitForElement = (id) => new Promise(resolve => {
        const checkForElement = () => {
            const element = document.querySelector(id);
            if (element) return resolve(element);
            setTimeout(checkForElement, 100);
        };
        checkForElement();
    });

    // 获取活动选项卡的 ID
    function getActiveTab(elements, all = false) {
        const tabs = elements.img2imgTabs.querySelectorAll("button");

        if (all) return tabs;

        for (let tab of tabs) {
            if (tab.classList.contains("selected")) {
                return tab;
            }
        }
    }

    // 获取选项卡的 ID
    function getTabId(elements) {
        const activeTab = getActiveTab(elements);
        return tabNameToElementId[activeTab.innerText];
    }

    // 等待选项加载完成
    async function waitForOpts() {
        for (; ;) {
            if (window.opts && Object.keys(window.opts).length) {
                return window.opts;
            }
            await new Promise(resolve => setTimeout(resolve, 100));
        }
    }

    // 检测元素是否具有水平滚动条
    function hasHorizontalScrollbar(element) {
        return element.scrollWidth > element.clientWidth;
    }

    // 用于定义 "Ctrl"、"Shift" 和 "Alt" 键的函数
    // 检查按键是否为修饰键
    function isModifierKey(event, key) {
        switch (key) {
            case "Ctrl":
                return event.ctrlKey;
            case "Shift":
                return event.shiftKey;
            case "Alt":
                return event.altKey;
            default:
                return false;
        }
    }

    // 检查热键是否有效
    function isValidHotkey(value) {
        const specialKeys = ["Ctrl", "Alt", "Shift", "Disable"];
        return (
            (typeof value === "string" &&
                value.length === 1 &&
                /[a-z]/i.test(value)) ||
            specialKeys.includes(value)
        );
    }

    // 标准化热键
    function normalizeHotkey(hotkey) {
        return hotkey.length === 1 ? "Key" + hotkey.toUpperCase() : hotkey;
    }

    // 格式化用于显示的热键
    function formatHotkeyForDisplay(hotkey) {
        return hotkey.startsWith("Key") ? hotkey.slice(3) : hotkey;
    }

    // 使用提供的选项创建热键配置
    }

    // 根据提供的函数名列表禁用配置对象中的函数
    function disableFunctions(config, disabledFunctions) {
        // 将 hasOwnProperty 方法绑定到 functionMap 对象，以避免错误
        const hasOwnProperty =
            Object.prototype.hasOwnProperty.bind(functionMap);

        // 遍历 disabledFunctions 数组，并在配置对象中禁用相应的函数
        disabledFunctions.forEach(funcName => {
            if (hasOwnProperty(funcName)) {
                const key = functionMap[funcName];
                config[key] = "disable";
            }
        });

        // 返回更新后的配置对象
        return config;
    }
    /**
     * The restoreImgRedMask function displays a red mask around an image to indicate the aspect ratio.
     * If the image display property is set to 'none', the mask breaks. To fix this, the function
     * temporarily sets the display property to 'block' and then hides the mask again after 300 milliseconds
     * to avoid breaking the canvas. Additionally, the function adjusts the mask to work correctly on
     * very long images.
     */
    function restoreImgRedMask(elements) {
        // 获取主选项卡的 ID
        const mainTabId = getTabId(elements);
    
        // 如果主选项卡 ID 不存在，则返回
        if (!mainTabId) return;
    
        // 获取主选项卡元素和图片元素
        const mainTab = gradioApp().querySelector(mainTabId);
        const img = mainTab.querySelector("img");
        const imageARPreview = gradioApp().querySelector("#imageARPreview");
    
        // 如果图片元素或预览元素不存在，则返回
        if (!img || !imageARPreview) return;
    
        // 重置预览元素的变换
        imageARPreview.style.transform = "";
        
        // 如果主选项卡宽度大于 865 像素
        if (parseFloat(mainTab.style.width) > 865) {
            // 获取主选项卡的变换信息
            const transformString = mainTab.style.transform;
            const scaleMatch = transformString.match(
                /scale\(([-+]?[0-9]*\.?[0-9]+)\)/
            );
            let zoom = 1; // 默认缩放比例
    
            // 如果存在缩放信息，则更新缩放比例
            if (scaleMatch && scaleMatch[1]) {
                zoom = Number(scaleMatch[1]);
            }
    
            // 设置预览元素的变换原点和缩放
            imageARPreview.style.transformOrigin = "0 0";
            imageARPreview.style.transform = `scale(${zoom})`;
        }
    
        // 如果图片元素的显示属性不是 'none'，则返回
        if (img.style.display !== "none") return;
    
        // 将图片元素的显示属性设置为 'block'
        img.style.display = "block";
    
        // 在 400 毫秒后将图片元素的显示属性设置回 'none'
        setTimeout(() => {
            img.style.display = "none";
        }, 400);
    }
    
    // 等待选项配置完成
    const hotkeysConfigOpts = await waitForOpts();
    
    // 默认配置
    // 默认的热键配置对象，包含了各种功能对应的热键
    const defaultHotkeysConfig = {
        canvas_hotkey_zoom: "Alt",
        canvas_hotkey_adjust: "Ctrl",
        canvas_hotkey_reset: "KeyR",
        canvas_hotkey_fullscreen: "KeyS",
        canvas_hotkey_move: "KeyF",
        canvas_hotkey_overlap: "KeyO",
        canvas_disabled_functions: [],
        canvas_show_tooltip: true,
        canvas_auto_expand: true,
        canvas_blur_prompt: false,
    };

    // 不同功能对应的热键映射关系
    const functionMap = {
        "Zoom": "canvas_hotkey_zoom",
        "Adjust brush size": "canvas_hotkey_adjust",
        "Moving canvas": "canvas_hotkey_move",
        "Fullscreen": "canvas_hotkey_fullscreen",
        "Reset Zoom": "canvas_hotkey_reset",
        "Overlap": "canvas_hotkey_overlap"
    };

    // 从 opts 中加载配置
    const preHotkeysConfig = createHotkeyConfig(
        defaultHotkeysConfig,
        hotkeysConfigOpts
    );

    // 禁用用户不需要的功能
    const hotkeysConfig = disableFunctions(
        preHotkeysConfig,
        preHotkeysConfig.canvas_disabled_functions
    );

    // 初始化变量
    let isMoving = false;
    let mouseX, mouseY;
    let activeElement;

    // 获取页面元素对象
    const elements = Object.fromEntries(
        Object.keys(elementIDs).map(id => [
            id,
            gradioApp().querySelector(elementIDs[id])
        ])
    );
    const elemData = {};

    // 为范围输入应用功能，并恢复红色遮罩并修正长图
    const rangeInputs = elements.rangeGroup ?
        Array.from(elements.rangeGroup.querySelectorAll("input")) :
        [
            gradioApp().querySelector("#img2img_width input[type='range']"),
            gradioApp().querySelector("#img2img_height input[type='range']")
        ];

    // 为范围输入添加事件监听器，当输入改变时恢复红色遮罩
    for (const input of rangeInputs) {
        input?.addEventListener("input", () => restoreImgRedMask(elements));
    }

    // 应用缩放和平移功能到指定元素
    applyZoomAndPan(elementIDs.sketch, false);
    applyZoomAndPan(elementIDs.inpaint, false);
    // 应用缩放和平移效果到指定元素
    applyZoomAndPan(elementIDs.inpaintSketch, false);

    // 将函数定义为全局函数，以便其他扩展可以利用这个解决方案
    const applyZoomAndPanIntegration = async(id, elementIDs) => {
        // 获取主元素
        const mainEl = document.querySelector(id);
        // 如果 id 为 "none"，则对 elementIDs 中的每个元素应用缩放和平移效果
        if (id.toLocaleLowerCase() === "none") {
            for (const elementID of elementIDs) {
                const el = await waitForElement(elementID);
                if (!el) break;
                applyZoomAndPan(elementID);
            }
            return;
        }

        // 如果主元素不存在，则返回
        if (!mainEl) return;
        // 给主元素添加点击事件监听器，点击时对 elementIDs 中的每个元素应用缩放和平移效果
        mainEl.addEventListener("click", async() => {
            for (const elementID of elementIDs) {
                const el = await waitForElement(elementID);
                if (!el) break;
                applyZoomAndPan(elementID);
            }
        }, {once: true});
    };

    // 将 applyZoomAndPan 函数设置为全局函数，只接受一个元素作为参数，例如 applyZoomAndPan("#txt2img_controlnet_ControlNet_input_image")
    window.applyZoomAndPan = applyZoomAndPan;

    // 将 applyZoomAndPanIntegration 函数设置为全局函数，供任何扩展使用
    window.applyZoomAndPanIntegration = applyZoomAndPanIntegration;
    /*
        函数 `applyZoomAndPanIntegration` 接受两个参数：

        1. `id`: 用于指定在单击时应用缩放和平移功能的元素的字符串标识符。
        如果 `id` 值为 "none"，则将功能应用于第二个参数中指定的所有元素，而无需单击事件。

        2. `elementIDs`: 一个包含元素字符串标识符的数组。在单击由第一个参数指定的元素时，将为每个元素应用缩放和平移功能。
        如果在第一个参数中指定了 "none"，则将为每个元素应用功能，而无需单击事件。

        示例用法：
        applyZoomAndPanIntegration("#txt2img_controlnet", ["#txt2img_controlnet_ControlNet_input_image"]);
        在此示例中，单击具有标识符 "txt2img_controlnet" 的元素时，将为具有标识符 "txt2img_controlnet_ControlNet_input_image" 的元素应用缩放和平移功能。
    */

    // 更多示例
    // 添加与 ControlNet txt2img One TAB 的集成
    // applyZoomAndPanIntegration("#txt2img_controlnet", ["#txt2img_controlnet_ControlNet_input_image"]);

    // 添加与 ControlNet txt2img Tabs 的集成
    // applyZoomAndPanIntegration("#txt2img_controlnet",Array.from({ length: 10 }, (_, i) => `#txt2img_controlnet_ControlNet-${i}_input_image`));

    // 添加与 Inpaint Anything 的集成
    // applyZoomAndPanIntegration("None", ["#ia_sam_image", "#ia_sel_mask"]);
});
```
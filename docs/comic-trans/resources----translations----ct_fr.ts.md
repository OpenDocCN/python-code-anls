
# `comic-translate\resources\translations\ct_fr.ts` 详细设计文档

这是Comic Translate应用的法语翻译文件（TS格式），用于存储和管理用户界面的多语言字符串资源。该文件包含了从源代码中提取的各类上下文（Context）及其对应的翻译文本，支持应用的国际化（i18n）流程。

## 整体流程

```mermaid
graph TD
    A[源代码变更] --> B[使用lupdate提取字符串]
    B --> C[更新TS文件 (XML)]
    C --> D{翻译处理}
    D -- 人工/机器翻译 --> E[填充translation标签]
    E --> F[使用lrelease编译为QM文件]
    F --> G[应用运行时加载QM文件]
    G --> H[显示法语用户界面]
```

## 类结构

```
TS Root (根节点)
├── Context: (空上下文/默认)
├── Context: AboutPage (关于页面)
├── Context: AccountPage (账户页面)
├── Context: ComicTranslate (主控制器上下文)
├── Context: ComicTranslateUI (主界面UI)
├── Context: CredentialsPage (凭证/API配置页面)
├── Context: ExportPage (导出设置页面)
├── Context: LlmsPage (LLM模型配置页面)
├── Context: Messages (全局消息/提示)
├── Context: PageListView (页面列表视图)
├── Context: PersonalizationPage (个性化/主题页面)
├── Context: SearchReplaceController (搜索替换控制器)
├── Context: SearchReplacePanel (搜索替换面板)
├── Context: SettingsPage (设置页面逻辑)
├── Context: SettingsPageUI (设置页面界面)
├── Context: TextRenderingPage (文本渲染配置)
├── Context: ToolsPage (工具选择/OCR/翻译器配置)
└── Context: self.* (Qt内部或动态加载的上下文)
```

## 全局变量及字段




### `ComicTranslateUI.English`
    
Label for English language option in the UI

类型：`str`
    


### `ComicTranslateUI.Korean`
    
Label for Korean language option in the UI

类型：`str`
    


### `ComicTranslateUI.Japanese`
    
Label for Japanese language option in the UI

类型：`str`
    


### `ComicTranslateUI.French`
    
Label for French language option in the UI

类型：`str`
    


### `ComicTranslateUI.Simplified Chinese`
    
Label for Simplified Chinese language option in the UI

类型：`str`
    


### `ComicTranslateUI.Traditional Chinese`
    
Label for Traditional Chinese language option in the UI

类型：`str`
    


### `ComicTranslateUI.Chinese`
    
Label for Chinese language option in the UI

类型：`str`
    


### `ComicTranslateUI.Russian`
    
Label for Russian language option in the UI

类型：`str`
    


### `ComicTranslateUI.German`
    
Label for German language option in the UI

类型：`str`
    


### `ComicTranslateUI.Dutch`
    
Label for Dutch language option in the UI

类型：`str`
    


### `ComicTranslateUI.Spanish`
    
Label for Spanish language option in the UI

类型：`str`
    


### `ComicTranslateUI.Italian`
    
Label for Italian language option in the UI

类型：`str`
    


### `ComicTranslateUI.Turkish`
    
Label for Turkish language option in the UI

类型：`str`
    


### `ComicTranslateUI.Polish`
    
Label for Polish language option in the UI

类型：`str`
    


### `ComicTranslateUI.Portuguese`
    
Label for Portuguese language option in the UI

类型：`str`
    


### `ComicTranslateUI.Brazilian Portuguese`
    
Label for Brazilian Portuguese language option in the UI

类型：`str`
    


### `ComicTranslateUI.Thai`
    
Label for Thai language option in the UI

类型：`str`
    


### `ComicTranslateUI.Vietnamese`
    
Label for Vietnamese language option in the UI

类型：`str`
    


### `ComicTranslateUI.Indonesian`
    
Label for Indonesian language option in the UI

类型：`str`
    


### `ComicTranslateUI.Hungarian`
    
Label for Hungarian language option in the UI

类型：`str`
    


### `ComicTranslateUI.Finnish`
    
Label for Finnish language option in the UI

类型：`str`
    


### `ComicTranslateUI.Arabic`
    
Label for Arabic language option in the UI

类型：`str`
    


### `ComicTranslateUI.Czech`
    
Label for Czech language option in the UI

类型：`str`
    


### `ComicTranslateUI.Persian`
    
Label for Persian language option in the UI

类型：`str`
    


### `ComicTranslateUI.Romanian`
    
Label for Romanian language option in the UI

类型：`str`
    


### `ComicTranslateUI.Mongolian`
    
Label for Mongolian language option in the UI

类型：`str`
    


### `ComicTranslateUI.New Project`
    
Menu option to create a new translation project

类型：`str`
    


### `ComicTranslateUI.Search / Replace (Ctrl+F)`
    
Menu option to open search and replace functionality

类型：`str`
    


### `ComicTranslateUI.Start New Project`
    
Button label to initiate a new project workflow

类型：`str`
    


### `ComicTranslateUI.Detect Text`
    
Button to detect text regions in the current image

类型：`str`
    


### `ComicTranslateUI.Recognize Text`
    
Button to perform OCR text recognition on detected regions

类型：`str`
    


### `ComicTranslateUI.Toggle Webtoon Mode`
    
Button to switch between normal and webtoon reading modes

类型：`str`
    


### `ComicTranslateUI.Translate All`
    
Button to translate all pages in the project

类型：`str`
    


### `ComicTranslateUI.Batch Report`
    
Button to view the batch processing report

类型：`str`
    


### `ComicTranslateUI.Images`
    
File type filter for image files

类型：`str`
    


### `ComicTranslateUI.Document`
    
File type filter for document files

类型：`str`
    


### `ComicTranslateUI.Archive`
    
File type filter for archive files

类型：`str`
    


### `ComicTranslateUI.Comic Book Archive`
    
File type filter for comic book archive files (cbz, cbr)

类型：`str`
    


### `ComicTranslateUI.Project File`
    
File type filter for project files

类型：`str`
    


### `ComicTranslateUI.Save Project`
    
Menu option to save the current project

类型：`str`
    


### `ComicTranslateUI.Save as`
    
Menu option to save the project with a new name

类型：`str`
    


### `ComicTranslateUI.Export all Images`
    
Menu option to export all processed images

类型：`str`
    


### `ComicTranslateUI.Undo`
    
Button to undo the last action

类型：`str`
    


### `ComicTranslateUI.Redo`
    
Button to redo a previously undone action

类型：`str`
    


### `ComicTranslateUI.Get Translations`
    
Button to fetch translations for detected text

类型：`str`
    


### `ComicTranslateUI.Segment Text`
    
Button to segment detected text into individual blocks

类型：`str`
    


### `ComicTranslateUI.Clean Image`
    
Button to clean the image using brush strokes

类型：`str`
    


### `ComicTranslateUI.Render`
    
Button to render translated text onto the image

类型：`str`
    


### `ComicTranslateUI.Manual`
    
Label for manual mode selection

类型：`str`
    


### `ComicTranslateUI.Automatic`
    
Label for automatic mode selection

类型：`str`
    


### `ComicTranslateUI.Source Language`
    
Label for selecting the source language

类型：`str`
    


### `ComicTranslateUI.Target Language`
    
Label for selecting the target translation language

类型：`str`
    


### `ComicTranslateUI.Font`
    
Label for font selection in text rendering settings

类型：`str`
    


### `ComicTranslateUI.Font Size`
    
Label for font size setting in text rendering

类型：`str`
    


### `ComicTranslateUI.Line Spacing`
    
Label for line spacing setting in text rendering

类型：`str`
    


### `ComicTranslateUI.Font Color`
    
Label for font color setting in text rendering

类型：`str`
    


### `ComicTranslateUI.Bold`
    
Checkbox label for bold text formatting

类型：`str`
    


### `ComicTranslateUI.Italic`
    
Checkbox label for italic text formatting

类型：`str`
    


### `ComicTranslateUI.Underline`
    
Checkbox label for underline text formatting

类型：`str`
    


### `ComicTranslateUI.Outline`
    
Checkbox label for text outline rendering

类型：`str`
    


### `ComicTranslateUI.Outline Color`
    
Label for outline color setting in text rendering

类型：`str`
    


### `ComicTranslateUI.Outline Width`
    
Label for outline width setting in text rendering

类型：`str`
    


### `ComicTranslateUI.Pan Image`
    
Tool option to pan around the image canvas

类型：`str`
    


### `ComicTranslateUI.Draw or Select Text Boxes`
    
Tool option for drawing or selecting text bounding boxes

类型：`str`
    


### `ComicTranslateUI.Delete Selected Box`
    
Button to delete the currently selected text box

类型：`str`
    


### `ComicTranslateUI.Remove all the Boxes on the Image`
    
Button to remove all text boxes from the current image

类型：`str`
    


### `ComicTranslateUI.Brush/Eraser Size Slider`
    
Slider control for brush or eraser size in image cleaning mode

类型：`str`
    


### `ComicTranslateUI.Box Drawing`
    
Tool mode for drawing text bounding boxes

类型：`str`
    


### `ComicTranslateUI.Inpainting`
    
Tool mode for inpainting image areas

类型：`str`
    


### `SettingsPage.OK`
    
Standard OK button label for dialog confirmations

类型：`str`
    


### `SettingsPage.Yes`
    
Standard Yes button label for confirmations

类型：`str`
    


### `SettingsPage.No`
    
Standard No button label for confirmations

类型：`str`
    


### `SettingsPage.Restart Required`
    
Title for restart required notification dialog

类型：`str`
    


### `SettingsPage.Failed to initiate sign-in process.`
    
Error message when sign-in fails to start

类型：`str`
    


### `SettingsPage.Sign In`
    
Button label to initiate user authentication

类型：`str`
    


### `SettingsPage.Sign In Required`
    
Title for sign-in requirement dialog

类型：`str`
    


### `SettingsPage.Please sign in to purchase or manage credits.`
    
Message prompting user to sign in for credit management

类型：`str`
    


### `SettingsPage.Unable to Open Browser`
    
Error message when external browser cannot be launched

类型：`str`
    


### `SettingsPage.Cancel`
    
Standard Cancel button label for dialogs

类型：`str`
    


### `SettingsPage.Sign In Error`
    
Title for sign-in error notification

类型：`str`
    


### `SettingsPage.Authentication failed: {error}`
    
Error message template for authentication failures

类型：`str`
    


### `SettingsPage.Confirm Sign Out`
    
Title for sign-out confirmation dialog

类型：`str`
    


### `SettingsPage.Are you sure you want to sign out?`
    
Confirmation message for user sign-out

类型：`str`
    


### `SettingsPage.Signing Out...`
    
Status message during sign-out process

类型：`str`
    


### `SettingsPage.Sign Out`
    
Button label to sign out the current user

类型：`str`
    


### `SettingsPage.Session Expired`
    
Title for session expiration notification

类型：`str`
    


### `SettingsPage.Your session has expired. Please sign in again.`
    
Message prompting user to re-authenticate after session timeout

类型：`str`
    


### `SettingsPage.N/A`
    
Label for not available or not applicable status

类型：`str`
    


### `SettingsPage.Free`
    
Label for free tier or free items

类型：`str`
    


### `SettingsPage.Subscription`
    
Label for subscription-based pricing tier

类型：`str`
    


### `SettingsPage.One-time`
    
Label for one-time payment option

类型：`str`
    


### `SettingsPage.Total`
    
Label for total amount or count

类型：`str`
    


### `SettingsPage.Checking...`
    
Status message during update or verification check

类型：`str`
    


### `SettingsPage.Check for Updates`
    
Button label to check for application updates

类型：`str`
    


### `SettingsPage.Update Available`
    
Notification title when a new version is available

类型：`str`
    


### `SettingsPage.A new version {version} is available.`
    
Message template announcing available update version

类型：`str`
    


### `SettingsPage.Release Notes`
    
Button or link label to view release notes

类型：`str`
    


### `SettingsPage.Skip This Version`
    
Button label to skip the current update

类型：`str`
    


### `SettingsPage.Up to Date`
    
Message indicating the application is current

类型：`str`
    


### `SettingsPage.You are using the latest version.`
    
Message confirming the application is up to date

类型：`str`
    


### `SettingsPage.Update Error`
    
Title for update error notification

类型：`str`
    


### `SettingsPage.Downloading update...`
    
Status message during update download

类型：`str`
    


### `SettingsPage.Download Complete`
    
Notification when update download finishes

类型：`str`
    


### `SettingsPage.Installer downloaded to {path}. Run it now?`
    
Message prompting user to run downloaded installer

类型：`str`
    


### `Messages.No Text Blocks Detected.`
    
Warning message when no text regions are found in an image

类型：`str`
    


### `Messages.Could not recognize detected text.`
    
Error message when OCR fails on detected text regions

类型：`str`
    


### `Messages.Could not get translations.`
    
Error message when translation service fails

类型：`str`
    


### `Messages.Comic has been Translated!`
    
Success notification when comic translation completes

类型：`str`
    


### `Messages.No Font selected.`
    
Error message when no font is configured for rendering

类型：`str`
    


### `Messages.Please sign in or sign up via Settings > Account to continue.`
    
Prompt for user to authenticate before continuing

类型：`str`
    


### `Messages.The translator does not support the selected target language.`
    
Error when chosen translator cannot handle target language

类型：`str`
    


### `Messages.No {0} selected. Please select a {1} in Settings > Tools.`
    
Template message for missing tool selection

类型：`str`
    


### `Messages.Insufficient credits to perform this action.`
    
Error when user lacks sufficient credits for operation

类型：`str`
    


### `Messages.Custom requires advanced API configuration.`
    
Warning that custom translation needs API setup

类型：`str`
    


### `Messages.Copy`
    
Button label to copy text to clipboard

类型：`str`
    


### `Messages.Close`
    
Button label to close a window or dialog

类型：`str`
    


### `Messages.We encountered an unexpected server error.`
    
Generic server error message for unexpected issues

类型：`str`
    


### `Messages.The external service provider is having trouble.`
    
Error message when third-party service experiences problems

类型：`str`
    


### `Messages.The server is currently busy or under maintenance.`
    
Error message when server is unavailable

类型：`str`
    


### `Messages.The server took too long to respond.`
    
Error message for server timeout

类型：`str`
    


### `Messages.The selected text recognition tool is not supported.`
    
Error when chosen OCR tool is unavailable

类型：`str`
    


### `Messages.The selected translator is not supported.`
    
Error when chosen translator is unavailable

类型：`str`
    


### `Messages.The selected tool is not supported.`
    
Error when chosen tool is unavailable

类型：`str`
    


### `Messages.{0} image(s) were skipped in this batch.`
    
Template message showing number of skipped images in batch processing

类型：`str`
    


### `Messages.Text Recognition blocked: The AI provider flagged this content.`
    
Error when content safety filter blocks OCR

类型：`str`
    


### `Messages.Translation blocked: The AI provider flagged this content.`
    
Error when content safety filter blocks translation

类型：`str`
    


### `Messages.Insufficient Credits`
    
Title for insufficient credits error

类型：`str`
    


### `ToolsPage.Translator`
    
Label for translator selection section

类型：`str`
    


### `ToolsPage.Text Recognition`
    
Label for OCR/text recognition tool selection

类型：`str`
    


### `ToolsPage.Text Detector`
    
Label for text detection tool selection

类型：`str`
    


### `ToolsPage.Image Cleaning`
    
Label for image cleaning tool selection

类型：`str`
    


### `ToolsPage.Inpainter`
    
Label for inpainting tool selection

类型：`str`
    


### `ToolsPage.AOT`
    
Label for AOT (Any Object Transformer) inpainting model

类型：`str`
    


### `ToolsPage.HD Strategy`
    
Label for high-definition processing strategy setting

类型：`str`
    


### `ToolsPage.Resize`
    
Option for resize HD strategy

类型：`str`
    


### `ToolsPage.Resize Limit:`
    
Label for resize dimension limit setting

类型：`str`
    


### `ToolsPage.Crop`
    
Option for crop HD strategy

类型：`str`
    


### `ToolsPage.Crop Margin:`
    
Label for crop margin setting

类型：`str`
    


### `ToolsPage.Crop Trigger Size:`
    
Label for crop trigger dimension setting

类型：`str`
    


### `ToolsPage.Use GPU`
    
Checkbox to enable GPU acceleration for processing

类型：`str`
    
    

## 全局函数及方法



## 关键组件





### OCR（光学字符识别）工具集

支持多种文本识别服务，包括Microsoft OCR、Google Cloud Vision、Gemini系列以及GPT系列模型，用于从漫画图像中提取文本内容。

### 翻译引擎集成

支持多种翻译服务提供商，包括DeepL、Google翻译、Microsoft Translator、Yandex、Google Gemini、Anthropic Claude、OpenAI GPT等，将识别的文本翻译成目标语言。

### 账户与积分系统

提供用户认证、积分购买与管理的功能，支持订阅层级和积分余额查询，用于支付翻译和OCR服务的费用。

### 批处理管道

支持批量处理漫画页面，包括批量翻译、批处理报告生成、Webtoon模式批处理等功能，提高处理效率。

### 图像渲染与文本绘制

提供文本渲染配置，包括字体选择、大小、颜色、轮廓效果，以及行间距、对齐方式等排版参数的控制。

### 项目文件管理

支持打开和保存漫画项目，处理多种格式包括图片（PNG、JPG）、PDF、Epub以及漫画档案格式（CBR、CBZ）。

### 搜索与替换功能

提供在原文和译文中搜索文本的能力，支持正则表达式匹配、大小写敏感搜索和整词匹配，以及批量替换功能。

### 设置与偏好管理

支持多语言UI（英语、法语、中文、日语、韩语等）、主题切换（明暗模式）、以及各种工具的API凭据配置。

### Webtoon模式

专为长条连载漫画设计的阅读和处理模式，支持垂直长条图像的检测、识别和翻译流程。

### 错误处理与报告

为各种失败场景（如API错误、网络问题、积分不足、内容被拦截等）提供用户友好的错误消息和批处理报告。



## 问题及建议





### 已知问题

-   **大量过时翻译未清理**：许多translation带有`type="vanished"`或`type="obsolete"`标记（如Microsoft OCR、Google Cloud Vision等），这些原文已不存在但翻译仍保留在文件中，导致文件膨胀和维护困难
-   **空context name**：存在`<context><name></name></context>`的空上下文，其中包含过时的翻译条目
-   **上下文命名不一致**：存在`self.main`、`self.settings_page`、`self.settings_page.ui`、`settings_page.ui`等以`self.`开头的特殊上下文名称，与其他正常命名的上下文不一致
-   **格式缩进不一致**：部分`<message>`标签内的内容缩进不规范，如`<message><source>Could not recognize webtoon chunk.`等有多余空格
-   **未翻译的占位符**：某些message缺少translation属性，如`<message><source>DeepL does not translate to Traditional Chinese</source></message>`等
-   **重复翻译条目**：同一个source在不同位置出现多次翻译，如"Replace"在多个context中有不同翻译

### 优化建议

-   **定期清理过时条目**：使用Qt Linguist的"Remove Obsolete"功能批量删除所有带有`type="vanished"`和`type="obsolete"`的条目，保持文件精简
-   **统一context命名规范**：清理空的context name，并将`self.xxx`格式的上下文统一为正常的功能模块命名
-   **建立翻译审核流程**：添加翻译完整性检查，确保所有source都有对应的translation
-   **修复格式问题**：统一XML缩进格式，删除不必要的空格
-   **使用翻译记忆**：对于重复的source，建立统一的翻译标准，避免同一术语在不同位置有不同译法



## 其它




### 设计目标与约束

该漫画翻译应用的核心设计目标是提供一个用户友好的桌面工具，能够自动识别漫画中的文字并进行多语言翻译。主要约束包括：1）支持多种输入格式（图片、PDF、Epub、CBZ、CBR等）；2）集成多种OCR和翻译服务提供商；3）提供批量处理能力以提高效率；4）支持积分制商业模式；5）需要网络连接以调用外部AI服务。

### 错误处理与异常设计

应用采用分级错误处理机制：1）用户输入验证错误通过弹窗提示用户；2）API调用错误（如认证失败、服务不可用、积分不足）通过消息对话框反馈；3）网络连接错误进行重试逻辑；4）批量处理中的单个页面失败不影响整体流程，生成跳过报告。异常类设计包括APIError、NetworkError、AuthenticationError、InsufficientCreditsError等，继承自基础异常类，便于统一捕获和处理。

### 数据流与状态机

应用的数据流遵循以下路径：导入文件 -> 文本检测 -> OCR识别 -> 翻译 -> 图像修复 -> 文本渲染 -> 导出。状态机包括：1）项目状态（新建、加载、保存、修改）；2）处理状态（就绪、处理中、已完成、已取消）；3）批量处理状态（排队、处理中、已完成、部分失败）。页面列表视图维护每个图像的处理状态和跳过原因。

### 外部依赖与接口契约

外部依赖包括：1）OCR服务接口（Microsoft Azure Vision、Google Cloud Vision、OpenAI GPT等）；2）翻译服务接口（DeepL、Google Translate、Microsoft Translator、OpenAI GPT、Claude等）；3）图像修复模型（AOT等）；4）用户认证和积分系统后端API。接口契约定义了标准请求/响应格式，包括图像输入格式、文本块数据结构、翻译结果返回格式、错误码定义等。

### 用户交互流程

主窗口提供两种模式：手动模式和自动模式。手动模式下用户逐步执行：文本检测 -> 文本识别 -> 翻译 -> 图像修复 -> 文本渲染。自动模式一键完成全部流程。工具栏提供撤销/重做、画笔/橡皮擦、文本框绘制等编辑功能。设置页面支持多语言界面切换、主题切换（明/暗）、API凭证配置、文本渲染参数调整等。

### 配置文件与持久化

应用使用JSON格式存储项目文件（.ctproj），包含：页面列表及路径、文本块坐标和内容、翻译结果、渲染参数等。用户偏好设置存储在本地配置文件中，包括：选择的OCR/翻译服务、API密钥（加密存储）、界面语言和主题、字体和渲染参数、导出设置等。批量处理报告保存处理历史和错误信息。

### 安全性考虑

API密钥采用加密存储，不以明文形式保存。认证流程使用OAuth2或类似的令牌机制，令牌有有效期并支持刷新。用户敏感信息（邮箱、积分余额）通过安全通道传输。内容安全过滤器检测并阻止AI服务提供商标记的不当内容。外部API调用使用HTTPS加密传输。

### 性能优化策略

大图像采用分块处理策略（Webtoon模式垂直分条）。高分辨率图像使用HD策略（Resize或Crop）降低处理开销。批量处理支持并行化和进度显示，支持取消操作。图像缓存机制减少重复加载。API调用实现速率限制和自动重试机制。

### 可扩展性架构

模块化设计允许添加新的OCR/翻译服务提供商，只需实现标准接口。插件化的图像修复模块支持替换不同模型。配置系统支持未来添加新的设置项。事件系统允许其他模块订阅处理状态变化。多语言支持通过Qt Linguist翻译文件实现，易于添加新语言。

### 版本兼容性

.ts翻译文件版本为2.1，记录了每个字符串的位置信息（文件名和行号）。已废弃（vanished）和过时（obsolete）的翻译条目保留用于历史参考，新代码使用当前有效的翻译条目。应用支持导入旧版本项目文件并尝试迁移配置。

    
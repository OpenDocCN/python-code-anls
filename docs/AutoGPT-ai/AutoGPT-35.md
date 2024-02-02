# AutoGPT源码解析 35

!!! warning
    The Pinecone, Milvus, Redis, and Weaviate memory backends were rendered incompatible
    by work on the memory system, and have been removed.
    Whether support will be added back in the future is subject to discussion,
    feel free to pitch in: https://github.com/Significant-Gravitas/AutoGPT/discussions/4280

## Setting Your Cache Type

By default, AutoGPT set up with Docker Compose will use Redis as its memory backend.
Otherwise, the default is LocalCache (which stores memory in a JSON file).

To switch to a different backend, change the `MEMORY_BACKEND` in `.env`
to the value that you want:

* `json_file` uses a local JSON cache file
* `pinecone` uses the Pinecone.io account you configured in your ENV settings
* `redis` will use the redis cache that you configured
* `milvus` will use the milvus cache that you configured
* `weaviate` will use the weaviate cache that you configured

!!! warning
    The Pinecone, Milvus, Redis, and Weaviate memory backends were rendered incompatible
    by work on the memory system, and have been removed.
    Whether support will be added back in the future is subject to discussion,
    feel free to pitch in: https://github.com/Significant-Gravitas/AutoGPT/discussions/4280

## Memory Backend Setup

Links to memory backends

- [Pinecone](https://www.pinecone.io/)
- [Milvus](https://milvus.io/) &ndash; [self-hosted](https://milvus.io/docs), or managed with [Zilliz Cloud](https://zilliz.com/)
- [Redis](https://redis.io)
- [Weaviate](https://weaviate.io)

!!! warning
    The Pinecone, Milvus, Redis, and Weaviate memory backends were rendered incompatible
    by work on the memory system, and have been removed.
    Whether support will be added back in the future is subject to discussion,
    feel free to pitch in: https://github.com/Significant-Gravitas/AutoGPT/discussions/4280

### Redis Setup

!!! important
    If you have set up AutoGPT using Docker Compose, then Redis is included, no further
    setup needed.

!!! caution
    This setup is not intended to be publicly accessible and lacks security measures.
    Avoid exposing Redis to the internet without a password or at all!

1. Launch Redis container

    ```py
    docker run -d --name redis-stack-server -p 6379:6379 redis/redis-stack-server:latest
    ```

3. Set the following settings in `.env`

    ```py
    MEMORY_BACKEND=redis
    REDIS_HOST=localhost
    REDIS_PORT=6379
    REDIS_PASSWORD=<PASSWORD>
    ```
    
    Replace `<PASSWORD>` by your password, omitting the angled brackets (<>).

    Optional configuration:

    - `WIPE_REDIS_ON_START=False` to persist memory stored in Redis between runs.
    - `MEMORY_INDEX=<WHATEVER>` to specify a name for the memory index in Redis.
        The default is `auto-gpt`.

!!! info
    See [redis-stack-server](https://hub.docker.com/r/redis/redis-stack-server) for
    setting a password and additional configuration.

!!! warning
    The Pinecone, Milvus, Redis, and Weaviate memory backends were rendered incompatible
    by work on the memory system, and have been removed.
    Whether support will be added back in the future is subject to discussion,
    feel free to pitch in: https://github.com/Significant-Gravitas/AutoGPT/discussions/4280

### 🌲 Pinecone API Key Setup

Pinecone lets you store vast amounts of vector-based memory, allowing the agent to load only relevant memories at any given time.

1. Go to [pinecone](https://app.pinecone.io/) and make an account if you don't already have one.
2. Choose the `Starter` plan to avoid being charged.
3. Find your API key and region under the default project in the left sidebar.

In the `.env` file set:

- `PINECONE_API_KEY`
- `PINECONE_ENV` (example: `us-east4-gcp`)
- `MEMORY_BACKEND=pinecone`

!!! warning
    The Pinecone, Milvus, Redis, and Weaviate memory backends were rendered incompatible
    by work on the memory system, and have been removed.
    Whether support will be added back in the future is subject to discussion,
    feel free to pitch in: https://github.com/Significant-Gravitas/AutoGPT/discussions/4280

### Milvus Setup

[Milvus](https://milvus.io/) is an open-source, highly scalable vector database to store
huge amounts of vector-based memory and provide fast relevant search. It can be quickly
deployed with docker, or as a cloud service provided by [Zilliz Cloud](https://zilliz.com/).

1. Deploy your Milvus service, either locally using docker or with a managed Zilliz Cloud database:
    - [Install and deploy Milvus locally](https://milvus.io/docs/install_standalone-operator.md)

    - Set up a managed Zilliz Cloud database
        1. Go to [Zilliz Cloud](https://zilliz.com/) and sign up if you don't already have account.
        2. In the *Databases* tab, create a new database.
            - Remember your username and password
            - Wait until the database status is changed to RUNNING.
        3. In the *Database detail* tab of the database you have created, the public cloud endpoint, such as:
        `https://xxx-xxxx.xxxx.xxxx.zillizcloud.com:443`.

2. Run `pip3 install pymilvus` to install the required client library.
    Make sure your PyMilvus version and Milvus version are [compatible](https://github.com/milvus-io/pymilvus#compatibility)
    to avoid issues.
    See also the [PyMilvus installation instructions](https://github.com/milvus-io/pymilvus#installation).

3. Update `.env`:
    - `MEMORY_BACKEND=milvus`
    - One of:
        - `MILVUS_ADDR=host:ip` (for local instance)
        - `MILVUS_ADDR=https://xxx-xxxx.xxxx.xxxx.zillizcloud.com:443` (for Zilliz Cloud)

    The following settings are **optional**:

    - `MILVUS_USERNAME='username-of-your-milvus-instance'`
    - `MILVUS_PASSWORD='password-of-your-milvus-instance'`
    - `MILVUS_SECURE=True` to use a secure connection.
        Only use if your Milvus instance has TLS enabled.
        *Note: setting `MILVUS_ADDR` to a `https://` URL will override this setting.*
    - `MILVUS_COLLECTION` to change the collection name to use in Milvus.
        Defaults to `autogpt`.

!!! warning
    The Pinecone, Milvus, Redis, and Weaviate memory backends were rendered incompatible
    by work on the memory system, and have been removed.
    Whether support will be added back in the future is subject to discussion,
    feel free to pitch in: https://github.com/Significant-Gravitas/AutoGPT/discussions/4280

### Weaviate Setup
[Weaviate](https://weaviate.io/) is an open-source vector database. It allows to store
data objects and vector embeddings from ML-models and scales seamlessly to billion of
data objects. To set up a Weaviate database, check out their [Quickstart Tutorial](https://weaviate.io/developers/weaviate/quickstart).

Although still experimental, [Embedded Weaviate](https://weaviate.io/developers/weaviate/installation/embedded)
is supported which allows the AutoGPT process itself to start a Weaviate instance.
To enable it, set `USE_WEAVIATE_EMBEDDED` to `True` and make sure you `poetry add weaviate-client@^3.15.4`.

#### Install the Weaviate client

Install the Weaviate client before usage.

```py
$ poetry add weaviate-client
```

#### Setting up environment variables

In your `.env` file set the following:

```py
MEMORY_BACKEND=weaviate
WEAVIATE_HOST="127.0.0.1" # the IP or domain of the running Weaviate instance
WEAVIATE_PORT="8080" 
WEAVIATE_PROTOCOL="http"
WEAVIATE_USERNAME="your username"
WEAVIATE_PASSWORD="your password"
WEAVIATE_API_KEY="your weaviate API key if you have one"
WEAVIATE_EMBEDDED_PATH="/home/me/.local/share/weaviate" # this is optional and indicates where the data should be persisted when running an embedded instance
USE_WEAVIATE_EMBEDDED=False # set to True to run Embedded Weaviate
MEMORY_INDEX="Autogpt" # name of the index to create for the application
```

## View Memory Usage

View memory usage by using the `--debug` flag :)


## 🧠 Memory pre-seeding

!!! warning
    Data ingestion is broken in v0.4.7 and possibly earlier versions. This is a known issue that will be addressed in future releases. Follow these issues for updates.
    [Issue 4435](https://github.com/Significant-Gravitas/AutoGPT/issues/4435)
    [Issue 4024](https://github.com/Significant-Gravitas/AutoGPT/issues/4024)
    [Issue 2076](https://github.com/Significant-Gravitas/AutoGPT/issues/2076)



Memory pre-seeding allows you to ingest files into memory and pre-seed it before running AutoGPT.

```py
$ python data_ingestion.py -h 
usage: data_ingestion.py [-h] (--file FILE | --dir DIR) [--init] [--overlap OVERLAP] [--max_length MAX_LENGTH]

Ingest a file or a directory with multiple files into memory. Make sure to set your .env before running this script.

options:
  -h, --help               show this help message and exit
  --file FILE              The file to ingest.
  --dir DIR                The directory containing the files to ingest.
  --init                   Init the memory and wipe its content (default: False)
  --overlap OVERLAP        The overlap size between chunks when ingesting files (default: 200)
  --max_length MAX_LENGTH  The max_length of each chunk when ingesting files (default: 4000)

# python data_ingestion.py --dir DataFolder --init --overlap 100 --max_length 2000
```

In the example above, the script initializes the memory, ingests all files within the `AutoGPT/auto_gpt_workspace/DataFolder` directory into memory with an overlap between chunks of 100 and a maximum length of each chunk of 2000.

Note that you can also use the `--file` argument to ingest a single file into memory and that data_ingestion.py will only ingest files within the `/auto_gpt_workspace` directory.

The DIR path is relative to the auto_gpt_workspace directory, so `python data_ingestion.py --dir . --init` will ingest everything in `auto_gpt_workspace` directory.

You can adjust the `max_length` and `overlap` parameters to fine-tune the way the
    documents are presented to the AI when it "recall" that memory:

- Adjusting the overlap value allows the AI to access more contextual information
    from each chunk when recalling information, but will result in more chunks being
    created and therefore increase memory backend usage and OpenAI API requests.
- Reducing the `max_length` value will create more chunks, which can save prompt
    tokens by allowing for more message history in the context, but will also
    increase the number of chunks.
- Increasing the `max_length` value will provide the AI with more contextual
    information from each chunk, reducing the number of chunks created and saving on
    OpenAI API requests. However, this may also use more prompt tokens and decrease
    the overall context available to the AI.

Memory pre-seeding is a technique for improving AI accuracy by ingesting relevant data
into its memory. Chunks of data are split and added to memory, allowing the AI to access
them quickly and generate more accurate responses. It's useful for large datasets or when
specific information needs to be accessed quickly. Examples include ingesting API or
GitHub documentation before running AutoGPT.

!!! attention
    If you use Redis for memory, make sure to run AutoGPT with `WIPE_REDIS_ON_START=False`

    For other memory backends, we currently forcefully wipe the memory when starting
    AutoGPT. To ingest data with those memory backends, you can call the
    `data_ingestion.py` script anytime during an AutoGPT run.

Memories will be available to the AI immediately as they are ingested, even if ingested
while AutoGPT is running.


# Configuration

Configuration is controlled through the `Config` object. You can set configuration variables via the `.env` file. If you don't have a `.env` file, create a copy of `.env.template` in your `AutoGPT` folder and name it `.env`.

## Environment Variables

- `AI_SETTINGS_FILE`: Location of the AI Settings file relative to the AutoGPT root directory. Default: ai_settings.yaml
- `AUDIO_TO_TEXT_PROVIDER`: Audio To Text Provider. Only option currently is `huggingface`. Default: huggingface
- `AUTHORISE_COMMAND_KEY`: Key response accepted when authorising commands. Default: y
- `AZURE_CONFIG_FILE`: Location of the Azure Config file relative to the AutoGPT root directory. Default: azure.yaml
- `BROWSE_CHUNK_MAX_LENGTH`: When browsing website, define the length of chunks to summarize. Default: 3000
- `BROWSE_SPACY_LANGUAGE_MODEL`: [spaCy language model](https://spacy.io/usage/models) to use when creating chunks. Default: en_core_web_sm
- `CHAT_MESSAGES_ENABLED`: Enable chat messages. Optional
- `DISABLED_COMMAND_CATEGORIES`: Command categories to disable. Command categories are Python module names, e.g. autogpt.commands.execute_code. See the directory `autogpt/commands` in the source for all command modules. Default: None
- `ELEVENLABS_API_KEY`: ElevenLabs API Key. Optional.
- `ELEVENLABS_VOICE_ID`: ElevenLabs Voice ID. Optional.
- `EMBEDDING_MODEL`: LLM Model to use for embedding tasks. Default: text-embedding-ada-002
- `EXECUTE_LOCAL_COMMANDS`: If shell commands should be executed locally. Default: False
- `EXIT_KEY`: Exit key accepted to exit. Default: n
- `FAST_LLM`: LLM Model to use for most tasks. Default: gpt-3.5-turbo
- `GITHUB_API_KEY`: [Github API Key](https://github.com/settings/tokens). Optional.
- `GITHUB_USERNAME`: GitHub Username. Optional.
- `GOOGLE_API_KEY`: Google API key. Optional.
- `GOOGLE_CUSTOM_SEARCH_ENGINE_ID`: [Google custom search engine ID](https://programmablesearchengine.google.com/controlpanel/all). Optional.
- `HEADLESS_BROWSER`: Use a headless browser while AutoGPT uses a web browser. Setting to `False` will allow you to see AutoGPT operate the browser. Default: True
- `HUGGINGFACE_API_TOKEN`: HuggingFace API, to be used for both image generation and audio to text. Optional.
- `HUGGINGFACE_AUDIO_TO_TEXT_MODEL`: HuggingFace audio to text model. Default: CompVis/stable-diffusion-v1-4
- `HUGGINGFACE_IMAGE_MODEL`: HuggingFace model to use for image generation. Default: CompVis/stable-diffusion-v1-4
- `IMAGE_PROVIDER`: Image provider. Options are `dalle`, `huggingface`, and `sdwebui`. Default: dalle
- `IMAGE_SIZE`: Default size of image to generate. Default: 256
- `MEMORY_BACKEND`: Memory back-end to use. Currently `json_file` is the only supported and enabled backend. Default: json_file
- `MEMORY_INDEX`: Value used in the Memory backend for scoping, naming, or indexing. Default: auto-gpt
- `OPENAI_API_KEY`: *REQUIRED*- Your [OpenAI API Key](https://platform.openai.com/account/api-keys).
- `OPENAI_ORGANIZATION`: Organization ID in OpenAI. Optional.
- `PLAIN_OUTPUT`: Plain output, which disables the spinner. Default: False
- `PLUGINS_CONFIG_FILE`: Path of the Plugins Config file relative to the AutoGPT root directory. Default: plugins_config.yaml
- `PROMPT_SETTINGS_FILE`: Location of the Prompt Settings file relative to the AutoGPT root directory. Default: prompt_settings.yaml
- `REDIS_HOST`: Redis Host. Default: localhost
- `REDIS_PASSWORD`: Redis Password. Optional. Default:
- `REDIS_PORT`: Redis Port. Default: 6379
- `RESTRICT_TO_WORKSPACE`: The restrict file reading and writing to the workspace directory. Default: True
- `SD_WEBUI_AUTH`: Stable Diffusion Web UI username:password pair. Optional.
- `SD_WEBUI_URL`: Stable Diffusion Web UI URL. Default: http://localhost:7860
- `SHELL_ALLOWLIST`: List of shell commands that ARE allowed to be executed by AutoGPT. Only applies if `SHELL_COMMAND_CONTROL` is set to `allowlist`. Default: None
- `SHELL_COMMAND_CONTROL`: Whether to use `allowlist` or `denylist` to determine what shell commands can be executed (Default: denylist)
- `SHELL_DENYLIST`: List of shell commands that ARE NOT allowed to be executed by AutoGPT. Only applies if `SHELL_COMMAND_CONTROL` is set to `denylist`. Default: sudo,su
- `SMART_LLM`: LLM Model to use for "smart" tasks. Default: gpt-4
- `STREAMELEMENTS_VOICE`: StreamElements voice to use. Default: Brian
- `TEMPERATURE`: Value of temperature given to OpenAI. Value from 0 to 2. Lower is more deterministic, higher is more random. See https://platform.openai.com/docs/api-reference/completions/create#completions/create-temperature
- `TEXT_TO_SPEECH_PROVIDER`: Text to Speech Provider. Options are `gtts`, `macos`, `elevenlabs`, and `streamelements`. Default: gtts
- `USER_AGENT`: User-Agent given when browsing websites. Default: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36"
- `USE_AZURE`: Use Azure's LLM Default: False
- `USE_WEB_BROWSER`: Which web browser to use. Options are `chrome`, `firefox`, `safari` or `edge` Default: chrome
- `WIPE_REDIS_ON_START`: Wipes data / index on start. Default: True


## 🔍 Google API Keys Configuration

!!! note
    This section is optional. Use the official Google API if search attempts return
    error 429. To use the `google_official_search` command, you need to set up your
    Google API key in your environment variables.

Create your project:

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. If you don't already have an account, create one and log in
3. Create a new project by clicking on the *Select a Project* dropdown at the top of the
    page and clicking *New Project*
4. Give it a name and click *Create*
5. Set up a custom search API and add to your .env file:
    5. Go to the [APIs & Services Dashboard](https://console.cloud.google.com/apis/dashboard)
    6. Click *Enable APIs and Services*
    7. Search for *Custom Search API* and click on it
    8. Click *Enable*
    9. Go to the [Credentials](https://console.cloud.google.com/apis/credentials) page
    10. Click *Create Credentials*
    11. Choose *API Key*
    12. Copy the API key
    13. Set it as the `GOOGLE_API_KEY` in your `.env` file
14. [Enable](https://console.developers.google.com/apis/api/customsearch.googleapis.com)
    the Custom Search API on your project. (Might need to wait few minutes to propagate.)
    Set up a custom search engine and add to your .env file:
    15. Go to the [Custom Search Engine](https://cse.google.com/cse/all) page
    16. Click *Add*
    17. Set up your search engine by following the prompts.
        You can choose to search the entire web or specific sites
    18. Once you've created your search engine, click on *Control Panel*
    19. Click *Basics*
    20. Copy the *Search engine ID*
    21. Set it as the `CUSTOM_SEARCH_ENGINE_ID` in your `.env` file

_Remember that your free daily custom search quota allows only up to 100 searches. To increase this limit, you need to assign a billing account to the project to profit from up to 10K daily searches._


# Text to Speech

Enter this command to use TTS _(Text-to-Speech)_ for AutoGPT

```py
python -m autogpt --speak
```

Eleven Labs provides voice technologies such as voice design, speech synthesis, and
premade voices that AutoGPT can use for speech.

1. Go to [ElevenLabs](https://beta.elevenlabs.io/) and make an account if you don't
    already have one.
2. Choose and setup the *Starter* plan.
3. Click the top right icon and find *Profile* to locate your API Key.

In the `.env` file set:

- `ELEVENLABS_API_KEY`
- `ELEVENLABS_VOICE_1_ID` (example: _"premade/Adam"_)

### List of available voices

!!! note
    You can use either the name or the voice ID to configure a voice

| Name   | Voice ID |
| ------ | -------- |
| Rachel | `21m00Tcm4TlvDq8ikWAM` |
| Domi   | `AZnzlk1XvdvUeBnXmlld` |
| Bella  | `EXAVITQu4vr4xnSDxMaL` |
| Antoni | `ErXwobaYiN019PkySvjV` |
| Elli   | `MF3mGyEYCl7XYWbV9V6O` |
| Josh   | `TxGEqnHWrfWFTfGW9XjX` |
| Arnold | `VR6AewLTigWG4xSOukaG` |
| Adam   | `pNInz6obpgDQGcFmaJgB` |
| Sam    | `yoZ06aMxZJJ28mfd3POQ` |


# `docs/_javascript/mathjax.js`

这段代码是一个JavaScript对象，它是从mathjax.js库中引入的。这个对象包含两个关键的属性：tex和options。

. tex：这是数学jax.js库的 tex 属性的包装对象。它包含两个键值对，分别是 "inlineMath" 和 "displayMath"。其中，"inlineMath" 包含了两个数组，"\\(" 和 "\\)"]。"displayMath" 也是一个数组，其中包含两个键值对，"\\[" 和 "\\]"。tex 属性的作用就是将数学公式渲染为文本字符串。

. options：这是数学jax.js库的 options 属性的包装对象。它包含两个键值对，分别是 "ignoreHtmlClass" 和 "processHtmlClass"。其中，"ignoreHtmlClass" 是一个字符串，用于指定哪些 CSS 类名不予处理。"processHtmlClass" 是一个布尔值，表示在将数学公式转换为 HTML 类名时是否对其进行处理。

通过这个 object，mathjax.js库被引入到网页中，并在文档准备好之后开始监听页面事件。当文档中的数学公式被渲染为文本字符串时，mathjax.js库将确保这些文本字符串可以被正确地显示为数学公式。


```py
window.MathJax = {
    tex: {
        inlineMath: [["\\(", "\\)"]],
        displayMath: [["\\[", "\\]"]],
        processEscapes: true,
        processEnvironments: true
    },
    options: {
        ignoreHtmlClass: ".*|",
        processHtmlClass: "arithmatex"
    }
};

document$.subscribe(() => {
    MathJax.typesetPromise()
})
```

# `docs/_javascript/tablesort.js`

这段代码的作用是使用 jQuery 库中的 `subscribe` 函数来订阅一份名为 "document$" 的 DOM 元素中的事件。当 "document$" 中的事件被触发时，会执行订阅函数中的代码。

具体来说，这段代码会执行以下操作：

1. 从文档中选择所有没有 `class` 属性的 "article table" 元素，并将它们存储在一个变量 `tables` 中。
2. 对于每个选择到的表格元素，使用 `Tablessort` 类将表格中的行按照降序或升序排序。
3. 循环遍历 `tables` 中的所有表格元素，将 `Tablessort` 实例应用于每个选择到的表格元素，以确保该实例对所有表格都有效。


```py
document$.subscribe(function () {
    var tables = document.querySelectorAll("article table:not([class])")
    tables.forEach(function (table) {
        new Tablesort(table)
    })
})
```

# AutoGPT Flutter Client

## Description

This repository contains the Flutter client for the AutoGPT project. The application facilitates users in discussing various tasks with a single agent. The app is built to be cross-platform and runs on Web, Android, iOS, Windows, and Mac.

## Features

- List and manage multiple tasks.
- Engage in chat conversations related to selected tasks.

## Design document

The design document for this project provides a detailed outline of the architecture, components, and other important aspects of this application. Please note that this is a living, growing document and it is subject to change as the project evolves.

You can access the design document [here](https://docs.google.com/document/d/1S-o2np1gq5JwFq40wPHDUVLi-mylz4WMvCB8psOUjc8/).

## Requirements

- Flutter 3.x
- Dart 3.x

Flutter comes with Dart, to install Flutter, follow the instructions here: https://docs.flutter.dev/get-started/install

## Installation

1. **Clone the repo:**
```py
git clone https://github.com/Significant-Gravitas/AutoGPT.git
```

2. **Navigate to the project directory:**
```py
cd AutoGPT/frontend
```

3. **Get Flutter packages:**
```py
flutter pub get
```

4. **Run the app:**
```py
flutter run -d chrome --web-port 5000
```

## Project Structure

- `lib/`: Contains the main source code for the application.
- `models/`: Data models that define the structure of the objects used in the app.
- `views/`: The UI components of the application.
- `viewmodels/`: The business logic and data handling for the views.
- `services/`: Contains the service classes that handle communication with backend APIs and other external data sources. These services are used to fetch and update data that the app uses, and they are consumed by the ViewModels.
- `test/`: Contains the test files for unit and widget tests.

## Responsive Design

The app features a responsive design that adapts to different screen sizes and orientations. On larger screens (Web, Windows, Mac), views are displayed side by side horizontally. On smaller screens (Android, iOS), views are displayed in a tab bar controller layout.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


# `frontend/build/web/flutter.js`

This is a class called `FlutterLoader` that is responsible for loading the Flutter web app. It takes an options object as its parameter and returns a `Promise<EngineInitializer>` that will either resolve with an `EngineInitializer` or reject with any error.

The `loadEntrypoint` method is used to load the entrypoint of the app, and it can be override with a custom entrypoint. The `didCreateEngineInitializer` method is called by Flutter to notify that the app's engine is ready to be initialized.


```py
// Copyright 2014 The Flutter Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

if (!_flutter) {
  var _flutter = {};
}
_flutter.loader = null;

(function () {
  "use strict";

  const baseUri = ensureTrailingSlash(getBaseURI());

  function getBaseURI() {
    const base = document.querySelector("base");
    return (base && base.getAttribute("href")) || "";
  }

  function ensureTrailingSlash(uri) {
    if (uri == "") {
      return uri;
    }
    return uri.endsWith("/") ? uri : `${uri}/`;
  }

  /**
   * Wraps `promise` in a timeout of the given `duration` in ms.
   *
   * Resolves/rejects with whatever the original `promises` does, or rejects
   * if `promise` takes longer to complete than `duration`. In that case,
   * `debugName` is used to compose a legible error message.
   *
   * If `duration` is < 0, the original `promise` is returned unchanged.
   * @param {Promise} promise
   * @param {number} duration
   * @param {string} debugName
   * @returns {Promise} a wrapped promise.
   */
  async function timeout(promise, duration, debugName) {
    if (duration < 0) {
      return promise;
    }
    let timeoutId;
    const _clock = new Promise((_, reject) => {
      timeoutId = setTimeout(() => {
        reject(
          new Error(
            `${debugName} took more than ${duration}ms to resolve. Moving on.`,
            {
              cause: timeout,
            }
          )
        );
      }, duration);
    });

    return Promise.race([promise, _clock]).finally(() => {
      clearTimeout(timeoutId);
    });
  }

  /**
   * Handles the creation of a TrustedTypes `policy` that validates URLs based
   * on an (optional) incoming array of RegExes.
   */
  class FlutterTrustedTypesPolicy {
    /**
     * Constructs the policy.
     * @param {[RegExp]} validPatterns the patterns to test URLs
     * @param {String} policyName the policy name (optional)
     */
    constructor(validPatterns, policyName = "flutter-js") {
      const patterns = validPatterns || [
        /\.js$/,
      ];
      if (window.trustedTypes) {
        this.policy = trustedTypes.createPolicy(policyName, {
          createScriptURL: function(url) {
            const parsed = new URL(url, window.location);
            const file = parsed.pathname.split("/").pop();
            const matches = patterns.some((pattern) => pattern.test(file));
            if (matches) {
              return parsed.toString();
            }
            console.error(
              "URL rejected by TrustedTypes policy",
              policyName, ":", url, "(download prevented)");
          }
        });
      }
    }
  }

  /**
   * Handles loading/reloading Flutter's service worker, if configured.
   *
   * @see: https://developers.google.com/web/fundamentals/primers/service-workers
   */
  class FlutterServiceWorkerLoader {
    /**
     * Injects a TrustedTypesPolicy (or undefined if the feature is not supported).
     * @param {TrustedTypesPolicy | undefined} policy
     */
    setTrustedTypesPolicy(policy) {
      this._ttPolicy = policy;
    }

    /**
     * Returns a Promise that resolves when the latest Flutter service worker,
     * configured by `settings` has been loaded and activated.
     *
     * Otherwise, the promise is rejected with an error message.
     * @param {*} settings Service worker settings
     * @returns {Promise} that resolves when the latest serviceWorker is ready.
     */
    loadServiceWorker(settings) {
      if (settings == null) {
        // In the future, settings = null -> uninstall service worker?
        console.debug("Null serviceWorker configuration. Skipping.");
        return Promise.resolve();
      }
      if (!("serviceWorker" in navigator)) {
        let errorMessage = "Service Worker API unavailable.";
        if (!window.isSecureContext) {
          errorMessage += "\nThe current context is NOT secure."
          errorMessage += "\nRead more: https://developer.mozilla.org/en-US/docs/Web/Security/Secure_Contexts";
        }
        return Promise.reject(
          new Error(errorMessage)
        );
      }
      const {
        serviceWorkerVersion,
        serviceWorkerUrl = `${baseUri}flutter_service_worker.js?v=${serviceWorkerVersion}`,
        timeoutMillis = 4000,
      } = settings;

      // Apply the TrustedTypes policy, if present.
      let url = serviceWorkerUrl;
      if (this._ttPolicy != null) {
        url = this._ttPolicy.createScriptURL(url);
      }

      const serviceWorkerActivation = navigator.serviceWorker
        .register(url)
        .then(this._getNewServiceWorker)
        .then(this._waitForServiceWorkerActivation);

      // Timeout race promise
      return timeout(
        serviceWorkerActivation,
        timeoutMillis,
        "prepareServiceWorker"
      );
    }

    /**
     * Returns the latest service worker for the given `serviceWorkerRegistrationPromise`.
     *
     * This might return the current service worker, if there's no new service worker
     * awaiting to be installed/updated.
     *
     * @param {Promise<ServiceWorkerRegistration>} serviceWorkerRegistrationPromise
     * @returns {Promise<ServiceWorker>}
     */
    async _getNewServiceWorker(serviceWorkerRegistrationPromise) {
      const reg = await serviceWorkerRegistrationPromise;

      if (!reg.active && (reg.installing || reg.waiting)) {
        // No active web worker and we have installed or are installing
        // one for the first time. Simply wait for it to activate.
        console.debug("Installing/Activating first service worker.");
        return reg.installing || reg.waiting;
      } else if (!reg.active.scriptURL.endsWith(serviceWorkerVersion)) {
        // When the app updates the serviceWorkerVersion changes, so we
        // need to ask the service worker to update.
        return reg.update().then((newReg) => {
          console.debug("Updating service worker.");
          return newReg.installing || newReg.waiting || newReg.active;
        });
      } else {
        console.debug("Loading from existing service worker.");
        return reg.active;
      }
    }

    /**
     * Returns a Promise that resolves when the `latestServiceWorker` changes its
     * state to "activated".
     *
     * @param {Promise<ServiceWorker>} latestServiceWorkerPromise
     * @returns {Promise<void>}
     */
    async _waitForServiceWorkerActivation(latestServiceWorkerPromise) {
      const serviceWorker = await latestServiceWorkerPromise;

      if (!serviceWorker || serviceWorker.state == "activated") {
        if (!serviceWorker) {
          return Promise.reject(
            new Error("Cannot activate a null service worker!")
          );
        } else {
          console.debug("Service worker already active.");
          return Promise.resolve();
        }
      }
      return new Promise((resolve, _) => {
        serviceWorker.addEventListener("statechange", () => {
          if (serviceWorker.state == "activated") {
            console.debug("Activated new service worker.");
            resolve();
          }
        });
      });
    }
  }

  /**
   * Handles injecting the main Flutter web entrypoint (main.dart.js), and notifying
   * the user when Flutter is ready, through `didCreateEngineInitializer`.
   *
   * @see https://docs.flutter.dev/development/platform-integration/web/initialization
   */
  class FlutterEntrypointLoader {
    /**
     * Creates a FlutterEntrypointLoader.
     */
    constructor() {
      // Watchdog to prevent injecting the main entrypoint multiple times.
      this._scriptLoaded = false;
    }

    /**
     * Injects a TrustedTypesPolicy (or undefined if the feature is not supported).
     * @param {TrustedTypesPolicy | undefined} policy
     */
    setTrustedTypesPolicy(policy) {
      this._ttPolicy = policy;
    }

    /**
     * Loads flutter main entrypoint, specified by `entrypointUrl`, and calls a
     * user-specified `onEntrypointLoaded` callback with an EngineInitializer
     * object when it's done.
     *
     * @param {*} options
     * @returns {Promise | undefined} that will eventually resolve with an
     * EngineInitializer, or will be rejected with the error caused by the loader.
     * Returns undefined when an `onEntrypointLoaded` callback is supplied in `options`.
     */
    async loadEntrypoint(options) {
      const { entrypointUrl = `${baseUri}main.dart.js`, onEntrypointLoaded } =
        options || {};

      return this._loadEntrypoint(entrypointUrl, onEntrypointLoaded);
    }

    /**
     * Resolves the promise created by loadEntrypoint, and calls the `onEntrypointLoaded`
     * function supplied by the user (if needed).
     *
     * Called by Flutter through `_flutter.loader.didCreateEngineInitializer` method,
     * which is bound to the correct instance of the FlutterEntrypointLoader by
     * the FlutterLoader object.
     *
     * @param {Function} engineInitializer @see https://github.com/flutter/engine/blob/main/lib/web_ui/lib/src/engine/js_interop/js_loader.dart#L42
     */
    didCreateEngineInitializer(engineInitializer) {
      if (typeof this._didCreateEngineInitializerResolve === "function") {
        this._didCreateEngineInitializerResolve(engineInitializer);
        // Remove the resolver after the first time, so Flutter Web can hot restart.
        this._didCreateEngineInitializerResolve = null;
        // Make the engine revert to "auto" initialization on hot restart.
        delete _flutter.loader.didCreateEngineInitializer;
      }
      if (typeof this._onEntrypointLoaded === "function") {
        this._onEntrypointLoaded(engineInitializer);
      }
    }

    /**
     * Injects a script tag into the DOM, and configures this loader to be able to
     * handle the "entrypoint loaded" notifications received from Flutter web.
     *
     * @param {string} entrypointUrl the URL of the script that will initialize
     *                 Flutter.
     * @param {Function} onEntrypointLoaded a callback that will be called when
     *                   Flutter web notifies this object that the entrypoint is
     *                   loaded.
     * @returns {Promise | undefined} a Promise that resolves when the entrypoint
     *                                is loaded, or undefined if `onEntrypointLoaded`
     *                                is a function.
     */
    _loadEntrypoint(entrypointUrl, onEntrypointLoaded) {
      const useCallback = typeof onEntrypointLoaded === "function";

      if (!this._scriptLoaded) {
        this._scriptLoaded = true;
        const scriptTag = this._createScriptTag(entrypointUrl);
        if (useCallback) {
          // Just inject the script tag, and return nothing; Flutter will call
          // `didCreateEngineInitializer` when it's done.
          console.debug("Injecting <script> tag. Using callback.");
          this._onEntrypointLoaded = onEntrypointLoaded;
          document.body.append(scriptTag);
        } else {
          // Inject the script tag and return a promise that will get resolved
          // with the EngineInitializer object from Flutter when it calls
          // `didCreateEngineInitializer` later.
          return new Promise((resolve, reject) => {
            console.debug(
              "Injecting <script> tag. Using Promises. Use the callback approach instead!"
            );
            this._didCreateEngineInitializerResolve = resolve;
            scriptTag.addEventListener("error", reject);
            document.body.append(scriptTag);
          });
        }
      }
    }

    /**
     * Creates a script tag for the given URL.
     * @param {string} url
     * @returns {HTMLScriptElement}
     */
    _createScriptTag(url) {
      const scriptTag = document.createElement("script");
      scriptTag.type = "application/javascript";
      // Apply TrustedTypes validation, if available.
      let trustedUrl = url;
      if (this._ttPolicy != null) {
        trustedUrl = this._ttPolicy.createScriptURL(url);
      }
      scriptTag.src = trustedUrl;
      return scriptTag;
    }
  }

  /**
   * The public interface of _flutter.loader. Exposes two methods:
   * * loadEntrypoint (which coordinates the default Flutter web loading procedure)
   * * didCreateEngineInitializer (which is called by Flutter to notify that its
   *                              Engine is ready to be initialized)
   */
  class FlutterLoader {
    /**
     * Initializes the Flutter web app.
     * @param {*} options
     * @returns {Promise?} a (Deprecated) Promise that will eventually resolve
     *                     with an EngineInitializer, or will be rejected with
     *                     any error caused by the loader. Or Null, if the user
     *                     supplies an `onEntrypointLoaded` Function as an option.
     */
    async loadEntrypoint(options) {
      const { serviceWorker, ...entrypoint } = options || {};

      // A Trusted Types policy that is going to be used by the loader.
      const flutterTT = new FlutterTrustedTypesPolicy();

      // The FlutterServiceWorkerLoader instance could be injected as a dependency
      // (and dynamically imported from a module if not present).
      const serviceWorkerLoader = new FlutterServiceWorkerLoader();
      serviceWorkerLoader.setTrustedTypesPolicy(flutterTT.policy);
      await serviceWorkerLoader.loadServiceWorker(serviceWorker).catch(e => {
        // Regardless of what happens with the injection of the SW, the show must go on
        console.warn("Exception while loading service worker:", e);
      });

      // The FlutterEntrypointLoader instance could be injected as a dependency
      // (and dynamically imported from a module if not present).
      const entrypointLoader = new FlutterEntrypointLoader();
      entrypointLoader.setTrustedTypesPolicy(flutterTT.policy);
      // Install the `didCreateEngineInitializer` listener where Flutter web expects it to be.
      this.didCreateEngineInitializer =
        entrypointLoader.didCreateEngineInitializer.bind(entrypointLoader);
      return entrypointLoader.loadEntrypoint(entrypoint);
    }
  }

  _flutter.loader = new FlutterLoader();
})();

```

# `frontend/build/web/flutter_service_worker.js`

该代码是一个 JavaScript 脚本，用于在 Flutter 应用程序中设置硬编码的资源文件。这些文件将在应用程序发布时打包到一起，然后下载到用户的设备上。

具体来说，该脚本定义了三个变量：MANIFEST、TEMP 和 CACHE_NAME，用于存储应用程序的清单文件、临时缓存文件和缓存目录的名称。

接着，该脚本定义了四个变量：RESOURCES，该变量存储了应用程序中所有的资源文件名和路径。这四个资源文件是：version.json、index.html、/ 和 main.dart.js。

最后，该脚本通过调用 flutter.js 文件中的抽吸式加载器函数，从主函数中抽取出 main.dart.js 和 favicon.png 两个文件，并定义了 icons/Icon-192.png 和 icons/Icon-maskable-192.png 两个资源文件。


```py
'use strict';
const MANIFEST = 'flutter-app-manifest';
const TEMP = 'flutter-temp-cache';
const CACHE_NAME = 'flutter-app-cache';

const RESOURCES = {"version.json": "46a52461e018faa623d9196334aa3f50",
"index.html": "cc1a3ce1e56133270358b49a5df3f0bf",
"/": "cc1a3ce1e56133270358b49a5df3f0bf",
"main.dart.js": "e2161c7a27249ead50512890f62bd1cf",
"flutter.js": "6fef97aeca90b426343ba6c5c9dc5d4a",
"favicon.png": "5dcef449791fa27946b3d35ad8803796",
"icons/Icon-192.png": "ac9a721a12bbc803b44f645561ecb1e1",
"icons/Icon-maskable-192.png": "c457ef57daa1d16f64b27b786ec2ea3c",
"icons/Icon-maskable-512.png": "301a7604d45b3e739efc881eb04896ea",
"icons/Icon-512.png": "96e752610906ba2a93c65f8abe1645f1",
```

It looks like you're trying to load an interactive rich text template from a location like GitHub using the "InsertTypo" extension. The template includes an header, a body, and someplaceholders for text and other elements.

This is a common pattern for implementing interactive templates on websites, and the format of the file extension is likely intended to be a hint at its intended purpose. The "./" at the beginning of the file name indicates that this file is a local file, and the "./" at the end indicates that it's a directory-level file (i.e., it's not meant to be imported by another script).

The contents of the file are organized into several sections, including a header with some placeholders for text, a body with more text, and a section for other elements like images and links. The text in the header includes some markup, such as bold and italic font choices, and some math expressions.

Overall, it's difficult to say more without actually seeing the template in action, but this file appears to be a simple example of an interactive rich text template that can be customized by the user.


```py
"manifest.json": "0fa552613b8ec0fda5cda565914e3b16",
"assets/AssetManifest.json": "1b1e4a4276722b65eb1ef765e2991840",
"assets/NOTICES": "28ba0c63fc6e4d1ef829af7441e27f78",
"assets/FontManifest.json": "dc3d03800ccca4601324923c0b1d6d57",
"assets/packages/cupertino_icons/assets/CupertinoIcons.ttf": "055d9e87e4a40dbf72b2af1a20865d57",
"assets/packages/fluttertoast/assets/toastify.js": "56e2c9cedd97f10e7e5f1cebd85d53e3",
"assets/packages/fluttertoast/assets/toastify.css": "a85675050054f179444bc5ad70ffc635",
"assets/shaders/ink_sparkle.frag": "f8b80e740d33eb157090be4e995febdf",
"assets/AssetManifest.bin": "791447d17744ac2ade3999c1672fdbe8",
"assets/fonts/MaterialIcons-Regular.otf": "245e0462249d95ad589a087f1c9f58e1",
"assets/assets/tree_structure.json": "cda9b1a239f956c547411efad9f7c794",
"assets/assets/google_logo.svg.png": "0e29f8e1acfb8996437dbb2b0f591f19",
"assets/assets/images/discord_logo.png": "0e4a4162c5de8665a7d63ae9665405ae",
"assets/assets/images/google_logo.svg.png": "0e29f8e1acfb8996437dbb2b0f591f19",
"assets/assets/images/github_logo.svg.png": "ba087b073efdc4996b035d3a12bad0e4",
```

It looks like you provide an array of JavaScript and CSS files that are expected to be included in a service worker. These files are related to the Canvaskit library, which appears to be a tool for synteling data between different storage systems such as Canvas and Google Cloud Storage.

The JavaScript files in this array are likely used to interact with the Canvaskit library and perform tasks such as loading and processing data. The CSS files are likely used to style the various elements of the application.


```py
"assets/assets/images/twitter_logo.png": "af6c11b96a5e732b8dfda86a2351ecab",
"assets/assets/images/autogpt_logo.png": "6a5362a7d1f2f840e43ee259e733476c",
"assets/assets/github_logo.svg.png": "ba087b073efdc4996b035d3a12bad0e4",
"assets/assets/coding_tree_structure.json": "017a857cf3e274346a0a7eab4ce02eed",
"assets/assets/scrape_synthesize_tree_structure.json": "a9665c1b465bb0cb939c7210f2bf0b13",
"assets/assets/data_tree_structure.json": "5f9627548304155821968182f3883ca7",
"assets/assets/general_tree_structure.json": "41dfbcdc2349dcdda2b082e597c6d5ee",
"canvaskit/skwasm.js": "95f16c6690f955a45b2317496983dbe9",
"canvaskit/skwasm.wasm": "d1fde2560be92c0b07ad9cf9acb10d05",
"canvaskit/chromium/canvaskit.js": "ffb2bb6484d5689d91f393b60664d530",
"canvaskit/chromium/canvaskit.wasm": "393ec8fb05d94036734f8104fa550a67",
"canvaskit/canvaskit.js": "5caccb235fad20e9b72ea6da5a0094e6",
"canvaskit/canvaskit.wasm": "d9f69e0f428f695dc3d66b3a83a4aa8e",
"canvaskit/skwasm.worker.js": "51253d3321b11ddb8d73fa8aa87d3b15"};
// The application shell files that are downloaded before a service worker can
```

这段代码是一个dart文件，其中包含一个名为"main.dart.js"的文件。这段代码的作用是在应用程序安装时，使用npm安装依赖时自动打开一个名为"TEMP"的临时缓存目录，将应用程序所需的静态资源(main.dart.js、index.html和assets目录下的AssetManifest.json和FontManifest.json)添加到缓存中。

具体来说，代码中首先定义了一个名为CORE的数组，包含了main.dart.js、index.html和assets目录下的三个文件。然后代码通过event.waitUntil()方法等待npm安装完成，并在安装完成后使用caches.open()方法打开TEMP缓存目录。接着使用缓存.addAll()方法将CORE数组中的每个文件创建一个新的Request对象，并设置请求的缓存属性为'reload'，即当缓存中的文件被更新时，会重新获取最新的文件并替换掉旧的文件。最后，代码通过npm安装成功后，返回一个Promise，等待缓存目录的设置完成，然后跳出等待循环，继续执行应用程序的安装操作。


```py
// start.
const CORE = ["main.dart.js",
"index.html",
"assets/AssetManifest.json",
"assets/FontManifest.json"];

// During install, the TEMP cache is populated with the application shell files.
self.addEventListener("install", (event) => {
  self.skipWaiting();
  return event.waitUntil(
    caches.open(TEMP).then((cache) => {
      return cache.addAll(
        CORE.map((value) => new Request(value, {'cache': 'reload'})));
    })
  );
});
```

This is a JavaScript function that modifies the content and cache caches, and updates the app shell TEMP files. It first checks for updates, and if there are any updates it updates the content and cache caches. Next, it checks for updates in the app shell TEMP files and updates them if there are any updates. Finally, it saves the updated manifest to the cache.

It uses the `tempCache`, `contentCache`, `cache`, and `manifestCache` objects to interact with the cache. It also uses the `caches`, `clients`, and `manifest` global objects.


```py
// During activate, the cache is populated with the temp files downloaded in
// install. If this service worker is upgrading from one with a saved
// MANIFEST, then use this to retain unchanged resource files.
self.addEventListener("activate", function(event) {
  return event.waitUntil(async function() {
    try {
      var contentCache = await caches.open(CACHE_NAME);
      var tempCache = await caches.open(TEMP);
      var manifestCache = await caches.open(MANIFEST);
      var manifest = await manifestCache.match('manifest');
      // When there is no prior manifest, clear the entire cache.
      if (!manifest) {
        await caches.delete(CACHE_NAME);
        contentCache = await caches.open(CACHE_NAME);
        for (var request of await tempCache.keys()) {
          var response = await tempCache.match(request);
          await contentCache.put(request, response);
        }
        await caches.delete(TEMP);
        // Save the manifest to make future upgrades efficient.
        await manifestCache.put('manifest', new Response(JSON.stringify(RESOURCES)));
        // Claim client to enable caching on first launch
        self.clients.claim();
        return;
      }
      var oldManifest = await manifest.json();
      var origin = self.location.origin;
      for (var request of await contentCache.keys()) {
        var key = request.url.substring(origin.length + 1);
        if (key == "") {
          key = "/";
        }
        // If a resource from the old manifest is not in the new cache, or if
        // the MD5 sum has changed, delete it. Otherwise the resource is left
        // in the cache and can be reused by the new service worker.
        if (!RESOURCES[key] || RESOURCES[key] != oldManifest[key]) {
          await contentCache.delete(request);
        }
      }
      // Populate the cache with the app shell TEMP files, potentially overwriting
      // cache files preserved above.
      for (var request of await tempCache.keys()) {
        var response = await tempCache.match(request);
        await contentCache.put(request, response);
      }
      await caches.delete(TEMP);
      // Save the manifest to make future upgrades efficient.
      await manifestCache.put('manifest', new Response(JSON.stringify(RESOURCES)));
      // Claim client to enable caching on first launch
      self.clients.claim();
      return;
    } catch (err) {
      // On an unhandled exception the state of the cache cannot be guaranteed.
      console.error('Failed to upgrade service worker: ' + err);
      await caches.delete(CACHE_NAME);
      await caches.delete(TEMP);
      await caches.delete(MANIFEST);
    }
  }());
});
```

这段代码是一个 JavaScript 代码片段，用于处理 HTTP GET 请求。它包含一个 fetch 事件的处理程序，用于处理 RESOURCE 文件的请求。

具体来说，当 fetch 事件被触发时，它会执行以下操作：

1. 如果请求方法不是 GET，则直接返回，因为不是所有 fetch 请求都使用 GET 方法。
2. 如果请求的 URL 以 "?" 和 "v=" 开始，那么将其去掉，因为它们只是查询参数。
3. 如果请求的 URL 是根目录或包含 "#" 符号，则将其重定向到 index.html。
4. 如果请求的 URL 是 RESOURCE 文件， 则执行 online-first 方法缓存， 在线优先下载资源。
5. 如果请求的 URL 是 index.html, 则执行 online-first 方法缓存， 在线优先下载资源。
6. 如果缓存中已存在该资源，则返回缓存的资源，否则返回一个 promise 并使用上下文中设定的缓存。

该代码使用 fetch 函数来发送 HTTP GET 请求，使用 '在线优先下载' 缓存实现高效下载资源。


```py
// The fetch handler redirects requests for RESOURCE files to the service
// worker cache.
self.addEventListener("fetch", (event) => {
  if (event.request.method !== 'GET') {
    return;
  }
  var origin = self.location.origin;
  var key = event.request.url.substring(origin.length + 1);
  // Redirect URLs to the index.html
  if (key.indexOf('?v=') != -1) {
    key = key.split('?v=')[0];
  }
  if (event.request.url == origin || event.request.url.startsWith(origin + '/#') || key == '') {
    key = '/';
  }
  // If the URL is not the RESOURCE list then return to signal that the
  // browser should take over.
  if (!RESOURCES[key]) {
    return;
  }
  // If the URL is the index.html, perform an online-first request.
  if (key == '/') {
    return onlineFirst(event);
  }
  event.respondWith(caches.open(CACHE_NAME)
    .then((cache) =>  {
      return cache.match(event.request).then((response) => {
        // Either respond with the cached resource, or perform a fetch and
        // lazily populate the cache only if the resource was successfully fetched.
        return response || fetch(event.request).then((response) => {
          if (response && Boolean(response.ok)) {
            cache.put(event.request, response.clone());
          }
          return response;
        });
      })
    })
  );
});
```

这段代码的作用是在 JavaScript 页面中添加了一个事件监听器，当接收到名为 'message' 的消息事件时，会执行其中的一个参数指定的回调函数。

回调函数中的代码首先检查消息事件的数据是否为 'skipWaiting'，如果是，就执行 self.skipWaiting() 方法，并返回，不再继续执行下面的代码。如果不是 'skipWaiting'，再检查消息数据是否为 'downloadOffline'，如果是，就执行 downloadOffline() 方法，并返回，不再继续执行下面的代码。

downloadOffline() 方法从本地仓库（可能是一个静态资源文件夹）中下载一个或多个资源，并将其添加到 resources 数组中。这个方法使用了 'async/await' 语法，可以在等待浏览器完成网络请求之后继续执行下面的代码。

最后，如果 'message' 事件没有发送 'downloadOffline' 消息，或者是 'skipWaiting' 消息，事件监听器将不再执行下面的代码，从而不会触发 downloadOffline() 方法，确保页面不会自动下载资源。


```py
self.addEventListener('message', (event) => {
  // SkipWaiting can be used to immediately activate a waiting service worker.
  // This will also require a page refresh triggered by the main worker.
  if (event.data === 'skipWaiting') {
    self.skipWaiting();
    return;
  }
  if (event.data === 'downloadOffline') {
    downloadOffline();
    return;
  }
});
// Download offline will check the RESOURCES for all files not in the cache
// and populate them.
async function downloadOffline() {
  var resources = [];
  var contentCache = await caches.open(CACHE_NAME);
  var currentContent = {};
  for (var request of await contentCache.keys()) {
    var key = request.url.substring(origin.length + 1);
    if (key == "") {
      key = "/";
    }
    currentContent[key] = true;
  }
  for (var resourceKey of Object.keys(RESOURCES)) {
    if (!currentContent[resourceKey]) {
      resources.push(resourceKey);
    }
  }
  return contentCache.addAll(resources);
}
```

该函数旨在在从离线缓存中下载资源之前尝试从互联网上下载资源。它使用了 fetch 函数来发起一个 HTTP 请求，并在请求返回时将其缓存到名为 CACHE_NAME 的缓存中。如果从缓存中已下载了资源，该函数将直接返回该资源，否则会使用 cache.match() 函数尝试从互联网上下载该资源。如果从互联网上下载成功，它将返回该资源，否则会捕获并处理任何错误。

以下是函数的示例用法：
```py
// 事件监听器，用于处理从窗口注册的 onload 事件
window.addEventListener('onload', onlineFirst);
```
该函数将在页面上加载资源时被调用，因此它将首先从缓存中下载资源，如果缓存中已下载了资源，它将直接返回该资源，否则它将尝试从互联网上下载资源。


```py
// Attempt to download the resource online before falling back to
// the offline cache.
function onlineFirst(event) {
  return event.respondWith(
    fetch(event.request).then((response) => {
      return caches.open(CACHE_NAME).then((cache) => {
        cache.put(event.request, response.clone());
        return response;
      });
    }).catch((error) => {
      return caches.open(CACHE_NAME).then((cache) => {
        return cache.match(event.request).then((response) => {
          if (response != null) {
            return response;
          }
          throw error;
        });
      });
    })
  );
}

```

# Launch Screen Assets

You can customize the launch screen with your own desired assets by replacing the image files in this directory.

You can also do it by opening your Flutter project's Xcode project with `open ios/Runner.xcworkspace`, selecting `Runner/Assets.xcassets` in the Project Navigator and dropping in the desired images.
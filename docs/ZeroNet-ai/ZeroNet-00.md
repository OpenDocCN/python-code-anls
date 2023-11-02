# ZeroNet源码解析 0

### ZeroNet 0.7.2 (2020-09-?) Rev4206?



### ZeroNet 0.7.1 (2019-07-01) Rev4206
### Added
 - Built-in logging console in the web UI to see what's happening in the background. (pull down top-right 0 button to see it)
 - Display database rebuild errors [Thanks to Lola]
 - New plugin system that allows to install and manage builtin/third party extensions to the ZeroNet client using the web interface.
 - Support multiple trackers_file
 - Add OpenSSL 1.1 support to CryptMessage plugin based on Bitmessage modifications [Thanks to radfish]
 - Display visual error message on startup errors
 - Fix max opened files changing on Windows platform
 - Display TLS1.3 compatibility on /Stats page
 - Add fake SNI and ALPN to peer connections to make it more like standard https connections
 - Hide and ignore tracker_proxy setting in Tor: Always mode as it's going to use Tor anyway.
 - Deny websocket connections from unknown origins
 - Restrict open_browser values to avoid RCE on sandbox escape
 - Offer access web interface by IP address in case of unknown host
 - Link to site's sidebar with "#ZeroNet:OpenSidebar" hash

### Changed
 - Allow .. in file names [Thanks to imachug]
 - Change unstable trackers
 - More clean errors on sites.json/users.json load error
 - Various tweaks for tracker rating on unstable connections
 - Use OpenSSL 1.1 dlls from default Python Windows distribution if possible
 - Re-factor domain resolving for easier domain plugins
 - Disable UDP connections if --proxy is used
 - New, decorator-based Websocket API permission system to avoid future typo mistakes

### Fixed
 - Fix parsing config lines that have no value
 - Fix start.py [Thanks to imachug]
 - Allow multiple values of the same key in the config file [Thanks ssdifnskdjfnsdjk for reporting]
 - Fix parsing config file lines that has % in the value [Thanks slrslr for reporting]
 - Fix bootstrapper plugin hash reloads [Thanks geekless for reporting]
 - Fix CryptMessage plugin OpenSSL dll loading on Windows (ZeroMail errors) [Thanks cxgreat2014 for reporting]
 - Fix startup error when using OpenSSL 1.1 [Thanks to imachug]
 - Fix a bug that did not loaded merged site data for 5 sec after the merged site got added
 - Fix typo that allowed to add new plugins in public proxy mode. [Thanks styromaniac for reporting]
 - Fix loading non-big files with "|all" postfix [Thanks to krzotr]
 - Fix OpenSSL cert generation error crash by change Windows console encoding to utf8

#### Wrapper html injection vulnerability [Reported by ivanq]

In ZeroNet before rev4188 the wrapper template variables was rendered incorrectly.

Result: The opened site was able to gain WebSocket connection with unrestricted ADMIN/NOSANDBOX access, change configuration values and possible RCE on client's machine.

Fix: Fixed the template rendering code, disallowed WebSocket connections from unknown locations, restricted open_browser configuration values to avoid possible RCE in case of sandbox escape.

Note: The fix is also back ported to ZeroNet Py 2.x version (Rev3870)


### ZeroNet 0.7.0 (2019-06-12) Rev4106 (First release targeting Python 3.4+)
### Added
 - 5-10x faster signature verification by using libsecp256k1 (Thanks to ZeroMux)
 - Generated SSL certificate randomization to avoid protocol filters (Thanks to ValdikSS)
 - Offline mode
 - P2P source code update using ZeroNet protocol
 - ecdsaSign/Verify commands to CryptMessage plugin (Thanks to imachug)
 - Efficient file rename: change file names instead of re-downloading the file.
 - Make redirect optional on site cloning (Thanks to Lola)
 - EccPrivToPub / EccPubToPriv functions (Thanks to imachug)
 - Detect and change dark/light theme based on OS setting (Thanks to filips123)

### Changed
 - Re-factored code to Python3 runtime (compatible with Python 3.4-3.8)
 - More safe database sync mode
 - Removed bundled third-party libraries where it's possible
 - Use lang=en instead of lang={lang} in urls to avoid url encode problems
 - Remove environment details from error page
 - Don't push content.json updates larger than 10kb to significantly reduce bw usage for site with many files

### Fixed
 - Fix sending files with \0 characters
 - Security fix: Escape error detail to avoid XSS (reported by krzotr)
 - Fix signature verification using libsecp256k1 for compressed addresses (mostly certificates generated in the browser)
 - Fix newsfeed if you have more than 1000 followed topic/post on one site.
 - Fix site download as zip file
 - Fix displaying sites with utf8 title
 - Error message if dbRebuild fails (Thanks to Lola)
 - Fix browser reopen if executing start.py again. (Thanks to imachug)


### ZeroNet 0.6.5 (2019-02-16) Rev3851 (Last release targeting Python 2.7.x)
### Added
 - IPv6 support in peer exchange, bigfiles, optional file finding, tracker sharing, socket listening and connecting (based on tangdou1 modifications)
 - New tracker database format with IPv6 support
 - Display notification if there is an unpublished modification for your site
 - Listen and shut down normally for SIGTERM (Thanks to blurHY)
 - Support tilde `~` in filenames (by d14na)
 - Support map for Namecoin subdomain names (Thanks to lola)
 - Add log level to config page
 - Support `{data}` for data dir variable in trackers_file value
 - Quick check content.db on startup and rebuild if necessary
 - Don't show meek proxy option if the tor client does not supports it

### Changed
 - Refactored port open checking with IPv6 support
 - Consider non-local IPs as external even is the open port check fails (for CJDNS and Yggdrasil support)
 - Add IPv6 tracker and change unstable tracker
 - Don't correct sent local time with the calculated time correction
 - Disable CSP for Edge
 - Only support CREATE commands in dbschema indexes node and SELECT from storage.query

### Fixed
 - Check the length of master seed when executing cryptGetPrivatekey CLI command
 - Only reload source code on file modification / creation
 - Detection and issue warning for latest no-script plugin
 - Fix atomic write of a non-existent file
 - Fix sql queries with lots of variables and sites with lots of content.json
 - Fix multi-line parsing of zeronet.conf
 - Fix site deletion from users.json
 - Fix site cloning before site downloaded (Reported by unsystemizer)
 - Fix queryJson for non-list nodes (Reported by MingchenZhang)


## ZeroNet 0.6.4 (2018-10-20) Rev3660
### Added
 - New plugin: UiConfig. A web interface that allows changing ZeroNet settings.
 - New plugin: AnnounceShare. Share trackers between users, automatically announce client's ip as tracker if Bootstrapper plugin is enabled.
 - Global tracker stats on ZeroHello: Include statistics from all served sites instead of displaying request statistics only for one site.
 - Support custom proxy for trackers. (Configurable with /Config)
 - Adding peers to sites manually using zeronet_peers get parameter
 - Copy site address with peers link on the sidebar.
 - Zip file listing and streaming support for Bigfiles.
 - Tracker statistics on /Stats page
 - Peer reputation save/restore to speed up sync time after startup.
 - Full support fileGet, fileList, dirList calls on tar.gz/zip files.
 - Archived_before support to user content rules to allow deletion of all user files before the specified date
 - Show and manage "Connecting" sites on ZeroHello
 - Add theme support to ZeroNet sites
 - Dark theme for ZeroHello, ZeroBlog, ZeroTalk

### Changed
 - Dynamic big file allocation: More efficient storage usage by don't pre-allocate the whole file at the beginning, but expand the size as the content downloads.
 - Reduce the request frequency to unreliable trackers.
 - Only allow 5 concurrent checkSites to run in parallel to reduce load under Tor/slow connection.
 - Stop site downloading if it reached 95% of site limit to avoid download loop for sites out of limit
 - The pinned optional files won't be removed from download queue after 30 retries and won't be deleted even if the site owner removes it.
 - Don't remove incomplete (downloading) sites on startup
 - Remove --pin_bigfile argument as big files are automatically excluded from optional files limit.

### Fixed
 - Trayicon compatibility with latest gevent
 - Request number counting for zero:// trackers
 - Peer reputation boost for zero:// trackers.
 - Blocklist of peers loaded from peerdb (Thanks tangdou1 for report)
 - Sidebar map loading on foreign languages (Thx tangdou1 for report)
 - FileGet on non-existent files (Thanks mcdev for reporting)
 - Peer connecting bug for sites with low amount of peers

#### "The Vacation" Sandbox escape bug [Reported by GitCenter / Krixano / ZeroLSTN]

In ZeroNet 0.6.3 Rev3615 and earlier as a result of invalid file type detection, a malicious site could escape the iframe sandbox.

Result: Browser iframe sandbox escape

Applied fix: Replaced the previous, file extension based file type identification with a proper one.

Affected versions: All versions before ZeroNet Rev3616


## ZeroNet 0.6.3 (2018-06-26)
### Added
 - New plugin: ContentFilter that allows to have shared site and user block list.
 - Support Tor meek proxies to avoid tracker blocking of GFW
 - Detect network level tracker blocking and easy setting meek proxy for tracker connections.
 - Support downloading 2GB+ sites as .zip (Thx to Radtoo)
 - Support ZeroNet as a transparent proxy (Thx to JeremyRand)
 - Allow fileQuery as CORS command (Thx to imachug)
 - Windows distribution includes Tor and meek client by default
 - Download sites as zip link to sidebar
 - File server port randomization
 - Implicit SSL for all connection
 - fileList API command for zip files
 - Auto download bigfiles size limit on sidebar
 - Local peer number to the sidebar
 - Open site directory button in sidebar

### Changed
 - Switched to Azure Tor meek proxy as Amazon one became unavailable
 - Refactored/rewritten tracker connection manager
 - Improved peer discovery for optional files without opened port
 - Also delete Bigfile's piecemap on deletion

### Fixed
 - Important security issue: Iframe sandbox escape [Reported by Ivanq / gitcenter]
 - Local peer discovery when running multiple clients on the same machine
 - Uploading small files with Bigfile plugin
 - Ctrl-c shutdown when running CLI commands
 - High CPU/IO usage when Multiuser plugin enabled
 - Firefox back button
 - Peer discovery on older Linux kernels
 - Optional file handling when multiple files have the same hash_id (first 4 chars of the hash)
 - Msgpack 0.5.5 and 0.5.6 compatibility

## ZeroNet 0.6.2 (2018-02-18)

### Added
 - New plugin: AnnounceLocal to make ZeroNet work without an internet connection on the local network.
 - Allow dbQuey and userGetSettings using the `as` API command on different sites with Cors permission
 - New config option: `--log_level` to reduce log verbosity and IO load
 - Prefer to connect to recent peers from trackers first
 - Mark peers with port 1 is also unconnectable for future fix for trackers that do not support port 0 announce

### Changed
 - Don't keep connection for sites that have not been modified in the last week
 - Change unreliable trackers to new ones
 - Send maximum 10 findhash request in one find optional files round (15sec)
 - Change "Unique to site" to "No certificate" for default option in cert selection dialog.
 - Dont print warnings if not in debug mode
 - Generalized tracker logging format
 - Only recover sites from sites.json if they had peers
 - Message from local peers does not means internet connection
 - Removed `--debug_gevent` and turned on Gevent block logging by default

### Fixed
 - Limit connections to 512 to avoid reaching 1024 limit on windows
 - Exception when logging foreign operating system socket errors
 - Don't send private (local) IPs on pex
 - Don't connect to private IPs in tor always mode
 - Properly recover data from msgpack unpacker on file stream start
 - Symlinked data directory deletion when deleting site using Windows
 - De-duplicate peers before publishing
 - Bigfile info for non-existing files


## ZeroNet 0.6.1 (2018-01-25)

### Added
 - New plugin: Chart
 - Collect and display charts about your contribution to ZeroNet network
 - Allow list as argument replacement in sql queries. (Thanks to imachug)
 - Newsfeed query time statistics (Click on "From XX sites in X.Xs on ZeroHello)
 - New UiWebsocket API command: As to run commands as other site
 - Ranged ajax queries for big files
 - Filter feed by type and site address
 - FileNeed, Bigfile upload command compatibility with merger sites
 - Send event on port open / tor status change
 - More description on permission request

### Changed
 - Reduce memory usage of sidebar geoip database cache
 - Change unreliable tracker to new one
 - Don't display Cors permission ask if it already granted
 - Avoid UI blocking when rebuilding a merger site
 - Skip listing ignored directories on signing
 - In Multiuser mode show the seed welcome message when adding new certificate instead of first visit
 - Faster async port opening on multiple network interfaces
 - Allow javascript modals
 - Only zoom sidebar globe if mouse button is pressed down

### Fixed
 - Open port checking error reporting (Thanks to imachug)
 - Out-of-range big file requests
 - Don't output errors happened on gevent greenlets twice
 - Newsfeed skip sites with no database
 - Newsfeed queries with multiple params
 - Newsfeed queries with UNION and UNION ALL
 - Fix site clone with sites larger that 10MB
 - Unreliable Websocket connection when requesting files from different sites at the same time


## ZeroNet 0.6.0 (2017-10-17)

### Added
 - New plugin: Big file support
 - Automatic pinning on Big file download
 - Enable TCP_NODELAY for supporting sockets
 - actionOptionalFileList API command arguments to list non-downloaded files or only big files
 - serverShowdirectory API command arguments to allow to display site's directory in OS file browser
 - fileNeed API command to initialize optional file downloading
 - wrapperGetAjaxKey API command to request nonce for AJAX request
 - Json.gz support for database files
 - P2P port checking (Thanks for grez911)
 - `--download_optional auto` argument to enable automatic optional file downloading for newly added site
 - Statistics for big files and protocol command requests on /Stats
 - Allow to set user limitation based on auth_address

### Changed
 - More aggressive and frequent connection timeout checking
 - Use out of msgpack context file streaming for files larger than 512KB
 - Allow optional files workers over the worker limit
 - Automatic redirection to wrapper on nonce_error
 - Send websocket event on optional file deletion
 - Optimize sites.json saving
 - Enable faster C-based msgpack packer by default
 - Major optimization on Bootstrapper plugin SQL queries
 - Don't reset bad file counter on restart, to allow easier give up on unreachable files
 - Incoming connection limit changed from 1000 to 500 to avoid reaching socket limit on Windows
 - Changed tracker boot.zeronet.io domain, because zeronet.io got banned in some countries

#### Fixed
 - Sub-directories in user directories

## ZeroNet 0.5.7 (2017-07-19)
### Added
 - New plugin: CORS to request read permission to other site's content
 - New API command: userSetSettings/userGetSettings to store site's settings in users.json
 - Avoid file download if the file size does not match with the requested one
 - JavaScript and wrapper less file access using /raw/ prefix ([Example](http://127.0.0.1:43110/raw/1AsRLpuRxr3pb9p3TKoMXPSWHzh6i7fMGi/en.tar.gz/index.html))
 - --silent command line option to disable logging to stdout


### Changed
 - Better error reporting on sign/verification errors
 - More test for sign and verification process
 - Update to OpenSSL v1.0.2l
 - Limit compressed files to 6MB to avoid zip/tar.gz bomb
 - Allow space, [], () characters in filenames
 - Disable cross-site resource loading to improve privacy. [Reported by Beardog108]
 - Download directly accessed Pdf/Svg/Swf files instead of displaying them to avoid wrapper escape using in JS in SVG file. [Reported by Beardog108]
 - Disallow potentially unsafe regular expressions to avoid ReDoS [Reported by MuxZeroNet]

### Fixed
 - Detecting data directory when running Windows distribution exe [Reported by Plasmmer]
 - OpenSSL loading under Android 6+
 - Error on exiting when no connection server started


## ZeroNet 0.5.6 (2017-06-15)
### Added
 - Callback for certSelect API command
 - More compact list formatting in json

### Changed
 - Remove obsolete auth_key_sha512 and signature format
 - Improved Spanish translation (Thanks to Pupiloho)

### Fixed
 - Opened port checking (Thanks l5h5t7 & saber28 for reporting)
 - Standalone update.py argument parsing (Thanks Zalex for reporting)
 - uPnP crash on startup (Thanks Vertux for reporting)
 - CoffeeScript 1.12.6 compatibility (Thanks kavamaken & imachug)
 - Multi value argument parsing
 - Database error when running from directory that contains special characters (Thanks Pupiloho for reporting)
 - Site lock violation logging


#### Proxy bypass during source upgrade [Reported by ZeroMux]

In ZeroNet before 0.5.6 during the client's built-in source code upgrade mechanism,
ZeroNet did not respect Tor and/or proxy settings.

Result: ZeroNet downloaded the update without using the Tor network and potentially leaked the connections.

Fix: Removed the problematic code line from the updater that removed the proxy settings from the socket library.

Affected versions: ZeroNet 0.5.5 and earlier, Fixed in: ZeroNet 0.5.6


#### XSS vulnerability using DNS rebinding. [Reported by Beardog108]

In ZeroNet before 0.5.6 the web interface did not validate the request's Host parameter.

Result: An attacker using a specially crafted DNS entry could have bypassed the browser's cross-site-scripting protection
and potentially gained access to user's private data stored on site.

Fix: By default ZeroNet only accept connections from 127.0.0.1 and localhost hosts.
If you bind the ui server to an external interface, then it also adds the first http request's host to the allowed host list
or you can define it manually using --ui_host.

Affected versions: ZeroNet 0.5.5 and earlier, Fixed in: ZeroNet 0.5.6


## ZeroNet 0.5.5 (2017-05-18)
### Added
- Outgoing socket binding by --bind parameter
- Database rebuilding progress bar
- Protect low traffic site's peers from cleanup closing
- Local site blacklisting
- Cloned site source code upgrade from parent
- Input placeholder support for displayPrompt
- Alternative interaction for wrapperConfirm

### Changed
- New file priorities for faster site display on first visit
- Don't add ? to url if push/replaceState url starts with #

### Fixed
- PermissionAdd/Remove admin command requirement
- Multi-line confirmation dialog


## ZeroNet 0.5.4 (2017-04-14)
### Added
- Major speed and CPU usage enhancements in Tor always mode
- Send skipped modifications to outdated clients

### Changed
- Upgrade libs to latest version
- Faster port opening and closing
- Deny site limit modification in MultiUser mode

### Fixed
- Filling database from optional files
- OpenSSL detection on systems with OpenSSL 1.1
- Users.json corruption on systems with slow hdd
- Fix leaking files in data directory by webui


## ZeroNet 0.5.3 (2017-02-27)
### Added
- Tar.gz/zip packed site support
- Utf8 filenames in archive files
- Experimental --db_mode secure database mode to prevent data loss on systems with unreliable power source.
- Admin user support in MultiUser mode
- Optional deny adding new sites in MultiUser mode

### Changed
- Faster update and publish times by new socket sharing algorithm

### Fixed
- Fix missing json_row errors when using Mute plugin


## ZeroNet 0.5.2 (2017-02-09)
### Added
- User muting
- Win/Mac signed exe/.app
- Signed commits

### Changed
- Faster site updates after startup
- New macOS package for 10.10 compatibility

### Fixed
- Fix "New version just released" popup on page first visit
- Fix disappearing optional files bug (Thanks l5h5t7 for reporting)
- Fix skipped updates on unreliable connections (Thanks P2P for reporting)
- Sandbox escape security fix (Thanks Firebox for reporting)
- Fix error reporting on async websocket functions


## ZeroNet 0.5.1 (2016-11-18)
### Added
- Multi language interface
- New plugin: Translation helper for site html and js files
- Per-site favicon

### Fixed
- Parallel optional file downloading


## ZeroNet 0.5.0 (2016-11-08)
### Added
- New Plugin: Allow list/delete/pin/manage files on ZeroHello
- New API commands to follow user's optional files, and query stats for optional files
- Set total size limit on optional files.
- New Plugin: Save peers to database and keep them between restarts to allow more faster optional file search and make it work without trackers
- Rewritten uPnP port opener + close port on exit (Thanks to sirMackk!)
- Lower memory usage by lazy PeerHashfield creation
- Loaded json files statistics and database info at /Stats page

### Changed
- Separate lock file for better Windows compatibility
- When executing start.py open browser even if ZeroNet is already running
- Keep plugin order after reload to allow plugins to extends an another plug-in
- Only save sites.json if fully loaded to avoid data loss
- Change aletorrenty tracker to a more reliable one
- Much lower findhashid CPU usage
- Pooled downloading of large amount of optional files
- Lots of other optional file changes to make it better
- If we have 1000 peers for a site make cleanup more aggressive
- Use warning instead of error on verification errors
- Push updates to newer clients first
- Bad file reset improvements

### Fixed
- Fix site deletion errors on startup
- Delay websocket messages until it's connected
- Fix database import if data file contains extra data
- Fix big site download
- Fix diff sending bug (been chasing it for a long time)
- Fix random publish errors when json file contained [] characters
- Fix site delete and siteCreate bug
- Fix file write confirmation dialog


## ZeroNet 0.4.1 (2016-09-05)
### Added
- Major core changes to allow fast startup and lower memory usage
- Try to reconnect to Tor on lost connection
- Sidebar fade-in
- Try to avoid incomplete data files overwrite
- Faster database open
- Display user file sizes in sidebar
- Concurrent worker number depends on --connection_limit

### Changed
- Close databases after 5 min idle time
- Better site size calculation
- Allow "-" character in domains
- Always try to keep connections for sites
- Remove merger permission from merged sites
- Newsfeed scans only last 3 days to speed up database queries
- Updated ZeroBundle-win to Python 2.7.12

### Fixed
- Fix for important security problem, which is allowed anyone to publish new content without valid certificate from ID provider. Thanks Kaffie for pointing it out!
- Fix sidebar error when no certificate provider selected
- Skip invalid files on database rebuilding
- Fix random websocket connection error popups
- Fix new siteCreate command
- Fix site size calculation
- Fix port open checking after computer wake up
- Fix --size_limit parsing from command line


## ZeroNet 0.4.0 (2016-08-11)
### Added
- Merger site plugin
- Live source code reloading: Faster core development by allowing me to make changes in ZeroNet source code without restarting it.
- New json table format for merger sites
- Database rebuild from sidebar.
- Allow to store custom data directly in json table: Much simpler and faster SQL queries.
- User file archiving: Allows the site owner to archive inactive user's content into single file. (Reducing initial sync time/cpu/memory usage)
- Also trigger onUpdated/update database on file delete.
- Permission request from ZeroFrame API.
- Allow to store extra data in content.json using fileWrite API command.
- Faster optional files downloading
- Use alternative sources (Gogs, Gitlab) to download updates
- Track provided sites/connection and prefer to keep the ones with more sites to reduce connection number

### Changed
- Keep at least 5 connection per site
- Changed target connection for sites to 10 from 15
- ZeroHello search function stability/speed improvements
- Improvements for clients with slower HDD

### Fixed
- Fix IE11 wrapper nonce errors
- Fix sidebar on mobile devices
- Fix site size calculation
- Fix IE10 compatibility
- Windows XP ZeroBundle compatibility (THX to people of China)


## ZeroNet 0.3.7 (2016-05-27)
### Changed
- Patch command to reduce bandwidth usage by transfer only the changed lines
- Other cpu/memory optimizations


## ZeroNet 0.3.6 (2016-05-27)
### Added
- New ZeroHello
- Newsfeed function

### Fixed
- Security fixes


## ZeroNet 0.3.5 (2016-02-02)
### Added
- Full Tor support with .onion hidden services
- Bootstrap using ZeroNet protocol

### Fixed
- Fix Gevent 1.0.2 compatibility


## ZeroNet 0.3.4 (2015-12-28)
### Added
- AES, ECIES API function support
- PushState and ReplaceState url manipulation support in API
- Multiuser localstorage


# ZeroNet [![Build Status](https://travis-ci.org/HelloZeroNet/ZeroNet.svg?branch=master)](https://travis-ci.org/HelloZeroNet/ZeroNet) [![Documentation](https://img.shields.io/badge/docs-faq-brightgreen.svg)](https://zeronet.io/docs/faq/) [![Help](https://img.shields.io/badge/keep_this_project_alive-donate-yellow.svg)](https://zeronet.io/docs/help_zeronet/donate/)

[简体中文](./README-zh-cn.md)
[English](./README.md)

Децентрализованные вебсайты использующие Bitcoin криптографию и BitTorrent сеть - https://zeronet.io


## Зачем?

* Мы верим в открытую, свободную, и не отцензуренную сеть и коммуникацию.
* Нет единой точки отказа: Сайт онлайн пока по крайней мере 1 пир обслуживает его.
* Никаких затрат на хостинг: Сайты обслуживаются посетителями.
* Невозможно отключить: Он нигде, потому что он везде.
* Быстр и работает оффлайн: Вы можете получить доступ к сайту, даже если Интернет недоступен.


## Особенности
 * Обновляемые в реальном времени сайты
 * Поддержка Namecoin .bit доменов
 * Лёгок в установке: распаковал & запустил
 * Клонирование вебсайтов в один клик
 * Password-less [BIP32](https://github.com/bitcoin/bips/blob/master/bip-0032.mediawiki)
   based authorization: Ваша учетная запись защищена той же криптографией, что и ваш Bitcoin-кошелек
 * Встроенный SQL-сервер с синхронизацией данных P2P: Позволяет упростить разработку сайта и ускорить загрузку страницы
 * Анонимность: Полная поддержка сети Tor с помощью скрытых служб .onion вместо адресов IPv4
 * TLS зашифрованные связи
 * Автоматическое открытие uPnP порта
 * Плагин для поддержки многопользовательской (openproxy)
 * Работает с любыми браузерами и операционными системами


## Как это работает?

* После запуска `zeronet.py` вы сможете посетить зайты (zeronet сайты) используя адрес
  `http://127.0.0.1:43110/{zeronet_address}`
(например. `http://127.0.0.1:43110/1HeLLo4uzjaLetFx6NH3PMwFP3qbRbTf3D`).
* Когда вы посещаете новый сайт zeronet, он пытается найти пиров с помощью BitTorrent
  чтобы загрузить файлы сайтов (html, css, js ...) из них.
* Каждый посещенный зайт также обслуживается вами. (Т.е хранится у вас на компьютере)
* Каждый сайт содержит файл `content.json`, который содержит все остальные файлы в хэше sha512
  и подпись, созданную с использованием частного ключа сайта.
* Если владелец сайта (у которого есть закрытый ключ для адреса сайта) изменяет сайт, то он/она
  подписывает новый `content.json` и публикует его для пиров. После этого пиры проверяют целостность `content.json`
  (используя подпись), они загружают измененные файлы и публикуют новый контент для других пиров.

####  [Слайд-шоу о криптографии ZeroNet, обновлениях сайтов, многопользовательских сайтах »](https://docs.google.com/presentation/d/1_2qK1IuOKJ51pgBvllZ9Yu7Au2l551t3XBgyTSvilew/pub?start=false&loop=false&delayms=3000)
####  [Часто задаваемые вопросы »](https://zeronet.io/docs/faq/)

####  [Документация разработчика ZeroNet »](https://zeronet.io/docs/site_development/getting_started/)


## Скриншоты

![Screenshot](https://i.imgur.com/H60OAHY.png)
![ZeroTalk](https://zeronet.io/docs/img/zerotalk.png)

#### [Больше скриншотов в ZeroNet документации »](https://zeronet.io/docs/using_zeronet/sample_sites/)


## Как вступить

* Скачайте ZeroBundle пакет:
  * [Microsoft Windows](https://github.com/HelloZeroNet/ZeroNet-win/archive/dist/ZeroNet-win.zip)
  * [Apple macOS](https://github.com/HelloZeroNet/ZeroNet-mac/archive/dist/ZeroNet-mac.zip)
  * [Linux 64-bit](https://github.com/HelloZeroNet/ZeroBundle/raw/master/dist/ZeroBundle-linux64.tar.gz)
  * [Linux 32-bit](https://github.com/HelloZeroNet/ZeroBundle/raw/master/dist/ZeroBundle-linux32.tar.gz)
* Распакуйте где угодно
* Запустите `ZeroNet.exe` (win), `ZeroNet(.app)` (osx), `ZeroNet.sh` (linux)

### Linux терминал

* `wget https://github.com/HelloZeroNet/ZeroBundle/raw/master/dist/ZeroBundle-linux64.tar.gz`
* `tar xvpfz ZeroBundle-linux64.tar.gz`
* `cd ZeroBundle`
* Запустите с помощью `./ZeroNet.sh`

Он загружает последнюю версию ZeroNet, затем запускает её автоматически.

#### Ручная установка для Debian Linux

* `sudo apt-get update`
* `sudo apt-get install msgpack-python python-gevent`
* `wget https://github.com/HelloZeroNet/ZeroNet/archive/master.tar.gz`
* `tar xvpfz master.tar.gz`
* `cd ZeroNet-master`
* Запустите с помощью `python2 zeronet.py`
* Откройте http://127.0.0.1:43110/ в вашем браузере.

### [Arch Linux](https://www.archlinux.org)

* `git clone https://aur.archlinux.org/zeronet.git`
* `cd zeronet`
* `makepkg -srci`
* `systemctl start zeronet`
* Откройте http://127.0.0.1:43110/ в вашем браузере.

Смотрите [ArchWiki](https://wiki.archlinux.org)'s [ZeroNet
article](https://wiki.archlinux.org/index.php/ZeroNet) для дальнейшей помощи.

### [Gentoo Linux](https://www.gentoo.org)

* [`layman -a raiagent`](https://github.com/leycec/raiagent)
* `echo '>=net-vpn/zeronet-0.5.4' >> /etc/portage/package.accept_keywords`
* *(Опционально)* Включить поддержку Tor: `echo 'net-vpn/zeronet tor' >>
  /etc/portage/package.use`
* `emerge zeronet`
* `rc-service zeronet start`
* Откройте http://127.0.0.1:43110/ в вашем браузере.

Смотрите `/usr/share/doc/zeronet-*/README.gentoo.bz2` для дальнейшей помощи.

### [FreeBSD](https://www.freebsd.org/)

* `pkg install zeronet` or `cd /usr/ports/security/zeronet/ && make install clean`
* `sysrc zeronet_enable="YES"`
* `service zeronet start`
* Откройте http://127.0.0.1:43110/ в вашем браузере.

### [Vagrant](https://www.vagrantup.com/)

* `vagrant up`
* Подключитесь к VM с помощью `vagrant ssh`
* `cd /vagrant`
* Запустите `python2 zeronet.py --ui_ip 0.0.0.0`
* Откройте http://127.0.0.1:43110/ в вашем браузере.

### [Docker](https://www.docker.com/)
* `docker run -d -v <local_data_folder>:/root/data -p 15441:15441 -p 127.0.0.1:43110:43110 nofish/zeronet`
* Это изображение Docker включает в себя прокси-сервер Tor, который по умолчанию отключён.
  Остерегайтесь что некоторые хостинг-провайдеры могут не позволить вам запускать Tor на своих серверах.
  Если вы хотите включить его,установите переменную среды `ENABLE_TOR` в` true` (по умолчанию: `false`) Например:

 `docker run -d -e "ENABLE_TOR=true" -v <local_data_folder>:/root/data -p 15441:15441 -p 127.0.0.1:43110:43110 nofish/zeronet`
* Откройте http://127.0.0.1:43110/ в вашем браузере.

### [Virtualenv](https://virtualenv.readthedocs.org/en/latest/)

* `virtualenv env`
* `source env/bin/activate`
* `pip install msgpack gevent`
* `python2 zeronet.py`
* Откройте http://127.0.0.1:43110/ в вашем браузере.

## Текущие ограничения

* ~~Нет torrent-похожего файла разделения для поддержки больших файлов~~ (поддержка больших файлов добавлена)
* ~~Не анонимнее чем Bittorrent~~ (добавлена встроенная поддержка Tor)
* Файловые транзакции не сжаты ~~ или незашифрованы еще ~~ (добавлено шифрование TLS)
* Нет приватных сайтов


## Как я могу создать сайт в Zeronet?

Завершите работу zeronet, если он запущен

```pybash
$ zeronet.py siteCreate
...
- Site private key (Приватный ключ сайта): 23DKQpzxhbVBrAtvLEc2uvk7DZweh4qL3fn3jpM3LgHDczMK2TtYUq
- Site address (Адрес сайта): 13DNDkMUExRf9Xa9ogwPKqp7zyHFEqbhC2
...
- Site created! (Сайт создан)
$ zeronet.py
...
```

Поздравляем, вы закончили! Теперь каждый может получить доступ к вашему зайту используя
`http://localhost:43110/13DNDkMUExRf9Xa9ogwPKqp7zyHFEqbhC2`

Следующие шаги: [ZeroNet Developer Documentation](https://zeronet.io/docs/site_development/getting_started/)


## Как я могу модифицировать Zeronet сайт?

* Измените файлы расположенные в data/13DNDkMUExRf9Xa9ogwPKqp7zyHFEqbhC2 директории.
  Когда закончите с изменением:

```pybash
$ zeronet.py siteSign 13DNDkMUExRf9Xa9ogwPKqp7zyHFEqbhC2
- Signing site (Подпись сайта): 13DNDkMUExRf9Xa9ogwPKqp7zyHFEqbhC2...
Private key (Приватный ключ) (input hidden):
```

* Введите секретный ключ, который вы получили при создании сайта, потом:

```pybash
$ zeronet.py sitePublish 13DNDkMUExRf9Xa9ogwPKqp7zyHFEqbhC2
...
Site:13DNDk..bhC2 Publishing to 3/10 peers...
Site:13DNDk..bhC2 Successfuly published to 3 peers
- Serving files....
```

* Вот и всё! Вы успешно подписали и опубликовали свои изменения.


## Поддержите проект

- Bitcoin: 1QDhxQ6PraUZa21ET5fYUCPgdrwBomnFgX
- Paypal: https://zeronet.io/docs/help_zeronet/donate/

### Спонсоры

* Улучшенная совместимость с MacOS / Safari стала возможной благодаря [BrowserStack.com](https://www.browserstack.com)

#### Спасибо!

* Больше информации, помощь, журнал изменений, zeronet сайты: https://www.reddit.com/r/zeronet/
* Приходите, пообщайтесь с нами: [#zeronet @ FreeNode](https://kiwiirc.com/client/irc.freenode.net/zeronet) или на [gitter](https://gitter.im/HelloZeroNet/ZeroNet)
* Email: hello@zeronet.io (PGP: CB9613AE)


# ZeroNet [![Build Status](https://travis-ci.org/HelloZeroNet/ZeroNet.svg?branch=py3)](https://travis-ci.org/HelloZeroNet/ZeroNet) [![Documentation](https://img.shields.io/badge/docs-faq-brightgreen.svg)](https://zeronet.io/docs/faq/) [![Help](https://img.shields.io/badge/keep_this_project_alive-donate-yellow.svg)](https://zeronet.io/docs/help_zeronet/donate/)

[English](./README.md)

使用 Bitcoin 加密和 BitTorrent 网络的去中心化网络 - https://zeronet.io


## 为什么？

* 我们相信开放，自由，无审查的网络和通讯
* 不会受单点故障影响：只要有在线的节点，站点就会保持在线
* 无托管费用：站点由访问者托管
* 无法关闭：因为节点无处不在
* 快速并可离线运行：即使没有互联网连接也可以使用


## 功能
 * 实时站点更新
 * 支持 Namecoin 的 .bit 域名
 * 安装方便：只需解压并运行
 * 一键克隆存在的站点
 * 无需密码、基于 [BIP32](https://github.com/bitcoin/bips/blob/master/bip-0032.mediawiki)
   的认证：您的账户被与比特币钱包相同的加密方法保护
 * 内建 SQL 服务器和 P2P 数据同步：让开发更简单并提升加载速度
 * 匿名性：完整的 Tor 网络支持，支持通过 .onion 隐藏服务相互连接而不是通过 IPv4 地址连接
 * TLS 加密连接
 * 自动打开 uPnP 端口
 * 多用户（openproxy）支持的插件
 * 适用于任何浏览器 / 操作系统


## 原理

* 在运行 `zeronet.py` 后，您将可以通过
  `http://127.0.0.1:43110/{zeronet_address}`（例如：
  `http://127.0.0.1:43110/1HeLLo4uzjaLetFx6NH3PMwFP3qbRbTf3D`）访问 zeronet 中的站点
* 在您浏览 zeronet 站点时，客户端会尝试通过 BitTorrent 网络来寻找可用的节点，从而下载需要的文件（html，css，js...）
* 您将会储存每一个浏览过的站点
* 每个站点都包含一个名为 `content.json` 的文件，它储存了其他所有文件的 sha512 散列值以及一个通过站点私钥生成的签名
* 如果站点的所有者（拥有站点地址的私钥）修改了站点，并且他 / 她签名了新的 `content.json` 然后推送至其他节点，
  那么这些节点将会在使用签名验证 `content.json` 的真实性后，下载修改后的文件并将新内容推送至另外的节点

####  [关于 ZeroNet 加密，站点更新，多用户站点的幻灯片 »](https://docs.google.com/presentation/d/1_2qK1IuOKJ51pgBvllZ9Yu7Au2l551t3XBgyTSvilew/pub?start=false&loop=false&delayms=3000)
####  [常见问题 »](https://zeronet.io/docs/faq/)

####  [ZeroNet 开发者文档 »](https://zeronet.io/docs/site_development/getting_started/)


## 屏幕截图

![Screenshot](https://i.imgur.com/H60OAHY.png)
![ZeroTalk](https://zeronet.io/docs/img/zerotalk.png)

#### [ZeroNet 文档中的更多屏幕截图 »](https://zeronet.io/docs/using_zeronet/sample_sites/)


## 如何加入

### Windows

 - 下载 [ZeroNet-py3-win64.zip](https://github.com/HelloZeroNet/ZeroNet-win/archive/dist-win64/ZeroNet-py3-win64.zip) (18MB)
 - 在任意位置解压缩
 - 运行 `ZeroNet.exe`
 
### macOS

 - 下载 [ZeroNet-dist-mac.zip](https://github.com/HelloZeroNet/ZeroNet-dist/archive/mac/ZeroNet-dist-mac.zip) (13.2MB)
 - 在任意位置解压缩
 - 运行 `ZeroNet.app`
 
### Linux (x86-64bit)

 - `wget https://github.com/HelloZeroNet/ZeroNet-linux/archive/dist-linux64/ZeroNet-py3-linux64.tar.gz`
 - `tar xvpfz ZeroNet-py3-linux64.tar.gz`
 - `cd ZeroNet-linux-dist-linux64/`
 - 使用以下命令启动 `./ZeroNet.sh`
 - 在浏览器打开 http://127.0.0.1:43110/ 即可访问 ZeroHello 页面
 
 __提示：__ 若要允许在 Web 界面上的远程连接，使用以下命令启动 `./ZeroNet.sh --ui_ip '*' --ui_restrict your.ip.address`

### 从源代码安装

 - `wget https://github.com/HelloZeroNet/ZeroNet/archive/py3/ZeroNet-py3.tar.gz`
 - `tar xvpfz ZeroNet-py3.tar.gz`
 - `cd ZeroNet-py3`
 - `sudo apt-get update`
 - `sudo apt-get install python3-pip`
 - `sudo python3 -m pip install -r requirements.txt`
 - 使用以下命令启动 `python3 zeronet.py`
 - 在浏览器打开 http://127.0.0.1:43110/ 即可访问 ZeroHello 页面

## 现有限制

* ~~没有类似于 torrent 的文件拆分来支持大文件~~ （已添加大文件支持）
* ~~没有比 BitTorrent 更好的匿名性~~ （已添加内置的完整 Tor 支持）
* 传输文件时没有压缩~~和加密~~ （已添加 TLS 支持）
* 不支持私有站点


## 如何创建一个 ZeroNet 站点？

 * 点击 [ZeroHello](http://127.0.0.1:43110/1HeLLo4uzjaLetFx6NH3PMwFP3qbRbTf3D) 站点的 **⋮** > **「新建空站点」** 菜单项
 * 您将被**重定向**到一个全新的站点，该站点只能由您修改
 * 您可以在 **data/[您的站点地址]** 目录中找到并修改网站的内容
 * 修改后打开您的网站，将右上角的「0」按钮拖到左侧，然后点击底部的**签名**并**发布**按钮

接下来的步骤：[ZeroNet 开发者文档](https://zeronet.io/docs/site_development/getting_started/)

## 帮助这个项目

- Bitcoin: 1QDhxQ6PraUZa21ET5fYUCPgdrwBomnFgX
- Paypal: https://zeronet.io/docs/help_zeronet/donate/

### 赞助商

* [BrowserStack.com](https://www.browserstack.com) 使更好的 macOS/Safari 兼容性成为可能

#### 感谢您！

* 更多信息，帮助，变更记录和 zeronet 站点：https://www.reddit.com/r/zeronet/
* 前往 [#zeronet @ FreeNode](https://kiwiirc.com/client/irc.freenode.net/zeronet) 或 [gitter](https://gitter.im/HelloZeroNet/ZeroNet) 和我们聊天
* [这里](https://gitter.im/ZeroNet-zh/Lobby)是一个 gitter 上的中文聊天室
* Email: hello@zeronet.io (PGP: [960F FF2D 6C14 5AA6 13E8 491B 5B63 BAE6 CB96 13AE](https://zeronet.io/files/tamas@zeronet.io_pub.asc))


# ZeroNet [![Build Status](https://travis-ci.org/HelloZeroNet/ZeroNet.svg?branch=py3)](https://travis-ci.org/HelloZeroNet/ZeroNet) [![Documentation](https://img.shields.io/badge/docs-faq-brightgreen.svg)](https://zeronet.io/docs/faq/) [![Help](https://img.shields.io/badge/keep_this_project_alive-donate-yellow.svg)](https://zeronet.io/docs/help_zeronet/donate/) ![tests](https://github.com/HelloZeroNet/ZeroNet/workflows/tests/badge.svg) [![Docker Pulls](https://img.shields.io/docker/pulls/nofish/zeronet)](https://hub.docker.com/r/nofish/zeronet)

Decentralized websites using Bitcoin crypto and the BitTorrent network - https://zeronet.io / [onion](http://zeronet34m3r5ngdu54uj57dcafpgdjhxsgq5kla5con4qvcmfzpvhad.onion)


## Why?

* We believe in open, free, and uncensored network and communication.
* No single point of failure: Site remains online so long as at least 1 peer is
  serving it.
* No hosting costs: Sites are served by visitors.
* Impossible to shut down: It's nowhere because it's everywhere.
* Fast and works offline: You can access the site even if Internet is
  unavailable.


## Features
 * Real-time updated sites
 * Namecoin .bit domains support
 * Easy to setup: unpack & run
 * Clone websites in one click
 * Password-less [BIP32](https://github.com/bitcoin/bips/blob/master/bip-0032.mediawiki)
   based authorization: Your account is protected by the same cryptography as your Bitcoin wallet
 * Built-in SQL server with P2P data synchronization: Allows easier site development and faster page load times
 * Anonymity: Full Tor network support with .onion hidden services instead of IPv4 addresses
 * TLS encrypted connections
 * Automatic uPnP port opening
 * Plugin for multiuser (openproxy) support
 * Works with any browser/OS


## How does it work?

* After starting `zeronet.py` you will be able to visit zeronet sites using
  `http://127.0.0.1:43110/{zeronet_address}` (eg.
  `http://127.0.0.1:43110/1HeLLo4uzjaLetFx6NH3PMwFP3qbRbTf3D`).
* When you visit a new zeronet site, it tries to find peers using the BitTorrent
  network so it can download the site files (html, css, js...) from them.
* Each visited site is also served by you.
* Every site contains a `content.json` file which holds all other files in a sha512 hash
  and a signature generated using the site's private key.
* If the site owner (who has the private key for the site address) modifies the
  site, then he/she signs the new `content.json` and publishes it to the peers.
  Afterwards, the peers verify the `content.json` integrity (using the
  signature), they download the modified files and publish the new content to
  other peers.

####  [Slideshow about ZeroNet cryptography, site updates, multi-user sites »](https://docs.google.com/presentation/d/1_2qK1IuOKJ51pgBvllZ9Yu7Au2l551t3XBgyTSvilew/pub?start=false&loop=false&delayms=3000)
####  [Frequently asked questions »](https://zeronet.io/docs/faq/)

####  [ZeroNet Developer Documentation »](https://zeronet.io/docs/site_development/getting_started/)


## Screenshots

![Screenshot](https://i.imgur.com/H60OAHY.png)
![ZeroTalk](https://zeronet.io/docs/img/zerotalk.png)

#### [More screenshots in ZeroNet docs »](https://zeronet.io/docs/using_zeronet/sample_sites/)


## How to join

### Windows

 - Download [ZeroNet-py3-win64.zip](https://github.com/HelloZeroNet/ZeroNet-win/archive/dist-win64/ZeroNet-py3-win64.zip) (18MB)
 - Unpack anywhere
 - Run `ZeroNet.exe`
 
### macOS

 - Download [ZeroNet-dist-mac.zip](https://github.com/HelloZeroNet/ZeroNet-dist/archive/mac/ZeroNet-dist-mac.zip) (13.2MB)
 - Unpack anywhere
 - Run `ZeroNet.app`
 
### Linux (x86-64bit)
 - `wget https://github.com/HelloZeroNet/ZeroNet-linux/archive/dist-linux64/ZeroNet-py3-linux64.tar.gz`
 - `tar xvpfz ZeroNet-py3-linux64.tar.gz`
 - `cd ZeroNet-linux-dist-linux64/`
 - Start with: `./ZeroNet.sh`
 - Open the ZeroHello landing page in your browser by navigating to: http://127.0.0.1:43110/
 
 __Tip:__ Start with `./ZeroNet.sh --ui_ip '*' --ui_restrict your.ip.address` to allow remote connections on the web interface.
 
 ### Android (arm, arm64, x86)
 - minimum Android version supported 16 (JellyBean)
 - [<img src="https://play.google.com/intl/en_us/badges/images/generic/en_badge_web_generic.png" 
      alt="Download from Google Play" 
      height="80">](https://play.google.com/store/apps/details?id=in.canews.zeronetmobile)
 - APK download: https://github.com/canewsin/zeronet_mobile/releases
 - XDA Labs: https://labs.xda-developers.com/store/app/in.canews.zeronet
 
#### Docker
There is an official image, built from source at: https://hub.docker.com/r/nofish/zeronet/

### Install from source

 - `wget https://github.com/HelloZeroNet/ZeroNet/archive/py3/ZeroNet-py3.tar.gz`
 - `tar xvpfz ZeroNet-py3.tar.gz`
 - `cd ZeroNet-py3`
 - `sudo apt-get update`
 - `sudo apt-get install python3-pip`
 - `sudo python3 -m pip install -r requirements.txt`
 - Start with: `python3 zeronet.py`
 - Open the ZeroHello landing page in your browser by navigating to: http://127.0.0.1:43110/

## Current limitations

* ~~No torrent-like file splitting for big file support~~ (big file support added)
* ~~No more anonymous than Bittorrent~~ (built-in full Tor support added)
* File transactions are not compressed ~~or encrypted yet~~ (TLS encryption added)
* No private sites


## How can I create a ZeroNet site?

 * Click on **⋮** > **"Create new, empty site"** menu item on the site [ZeroHello](http://127.0.0.1:43110/1HeLLo4uzjaLetFx6NH3PMwFP3qbRbTf3D).
 * You will be **redirected** to a completely new site that is only modifiable by you!
 * You can find and modify your site's content in **data/[yoursiteaddress]** directory
 * After the modifications open your site, drag the topright "0" button to left, then press **sign** and **publish** buttons on the bottom

Next steps: [ZeroNet Developer Documentation](https://zeronet.io/docs/site_development/getting_started/)

## Help keep this project alive

- Bitcoin: 1QDhxQ6PraUZa21ET5fYUCPgdrwBomnFgX
- Paypal: https://zeronet.io/docs/help_zeronet/donate/

### Sponsors

* Better macOS/Safari compatibility made possible by [BrowserStack.com](https://www.browserstack.com)

#### Thank you!

* More info, help, changelog, zeronet sites: https://www.reddit.com/r/zeronet/
* Come, chat with us: [#zeronet @ FreeNode](https://kiwiirc.com/client/irc.freenode.net/zeronet) or on [gitter](https://gitter.im/HelloZeroNet/ZeroNet)
* Email: hello@zeronet.io (PGP: [960F FF2D 6C14 5AA6 13E8 491B 5B63 BAE6 CB96 13AE](https://zeronet.io/files/tamas@zeronet.io_pub.asc))


# `start.py`

这段代码是一个用于启动 ZeroNet 网络模型的 Python 脚本。它主要包含两个部分：

1. 导入一些必要的模块，包括 `sys` 模块、`zeronet` 模块以及其他可能需要的模块。
2. 定义了一个 `main` 函数，它是这个脚本的核心部分。这个函数内部会执行以下操作：
a. 如果用户没有提供 `--open_browser` 参数，那么将 `"default_browser"` 替换到 `sys.argv` 列表中，使得用户需要提供的参数列表为 `["--open_browser", "default_browser"]`。这将在运行脚本时自动打开默认浏览器。
b. 调用 `zeronet.start()` 函数来启动 ZeroNet 网络模型。

总的来说，这段代码的主要作用是启动一个名为 "MyScript" 的 ZeroNet 网络模型，并在运行脚本时自动打开默认浏览器。


```py
#!/usr/bin/env python3


# Included modules
import sys

# ZeroNet Modules
import zeronet


def main():
    if "--open_browser" not in sys.argv:
        sys.argv = [sys.argv[0]] + ["--open_browser", "default_browser"] + sys.argv[1:]
    zeronet.start()

```

这段代码是一个if语句，它的作用是判断当前脚本是否作为主程序运行。如果当前脚本作为主程序运行，那么程序将跳转到__main__函数中执行。

具体来说，这段代码是一个带有两个修辞保留的if语句。第一个if语句检查当前脚本是否被称为__main__函数，如果是，那么执行main函数。第二个if语句是在__main__函数内部执行的，它将打印一个消息表明脚本已成功运行。

因此，这段代码的作用是检查当前脚本是否作为主程序运行，如果是，就执行main函数，否则打印一条消息表明脚本已成功运行。


```py
if __name__ == '__main__':
    main()

```

# `update.py`

This script appears to be a plugin deployment tool for WordPress. It takes a list of plugins that are to be installed and a list of websites or directories to update, and updates the websites or directories with the plugins.

It first checks if any of the websites or directories to update exist, and if not, creates them. Then, it loops through each plugin and checks if the file associated with that plugin is up-to-date. If the file is not up-to-date, it renames the old file and updates the file in the new location.

It also creates a "disabled-" version of the plugin name for each plugin in the plugins\_disabled directory, and renames any files in the disabled-目录.

It appears to be a command-line tool, as it uses the `print` and `os` commands to interact with the filesystem and os. The `print` command is used to display information about the plugin and the files to be updated. The `os` command is used to interact with the file system, such as renaming files.


```py
import os
import sys
import json
import re
import shutil


def update():
    from Config import config
    config.parse(silent=True)

    if getattr(sys, 'source_update_dir', False):
        if not os.path.isdir(sys.source_update_dir):
            os.makedirs(sys.source_update_dir)
        source_path = sys.source_update_dir.rstrip("/")
    else:
        source_path = os.getcwd().rstrip("/")

    if config.dist_type.startswith("bundle_linux"):
        runtime_path = os.path.normpath(os.path.dirname(sys.executable) + "/../..")
    else:
        runtime_path = os.path.dirname(sys.executable)

    updatesite_path = config.data_dir + "/" + config.updatesite

    sites_json = json.load(open(config.data_dir + "/sites.json"))
    updatesite_bad_files = sites_json.get(config.updatesite, {}).get("cache", {}).get("bad_files", {})
    print(
        "Update site path: %s, bad_files: %s, source path: %s, runtime path: %s, dist type: %s" %
        (updatesite_path, len(updatesite_bad_files), source_path, runtime_path, config.dist_type)
    )

    updatesite_content_json = json.load(open(updatesite_path + "/content.json"))
    inner_paths = list(updatesite_content_json.get("files", {}).keys())
    inner_paths += list(updatesite_content_json.get("files_optional", {}).keys())

    # Keep file only in ZeroNet directory
    inner_paths = [inner_path for inner_path in inner_paths if re.match("^(core|bundle)", inner_path)]

    # Checking plugins
    plugins_enabled = []
    plugins_disabled = []
    if os.path.isdir("%s/plugins" % source_path):
        for dir in os.listdir("%s/plugins" % source_path):
            if dir.startswith("disabled-"):
                plugins_disabled.append(dir.replace("disabled-", ""))
            else:
                plugins_enabled.append(dir)
        print("Plugins enabled:", plugins_enabled, "disabled:", plugins_disabled)

    update_paths = {}

    for inner_path in inner_paths:
        if ".." in inner_path:
            continue
        inner_path = inner_path.replace("\\", "/").strip("/")  # Make sure we have unix path
        print(".", end=" ")
        if inner_path.startswith("core"):
            dest_path = source_path + "/" + re.sub("^core/", "", inner_path)
        elif inner_path.startswith(config.dist_type):
            dest_path = runtime_path + "/" + re.sub("^bundle[^/]+/", "", inner_path)
        else:
            continue

        if not dest_path:
            continue

        # Keep plugin disabled/enabled status
        match = re.match(re.escape(source_path) + "/plugins/([^/]+)", dest_path)
        if match:
            plugin_name = match.group(1).replace("disabled-", "")
            if plugin_name in plugins_enabled:  # Plugin was enabled
                dest_path = dest_path.replace("plugins/disabled-" + plugin_name, "plugins/" + plugin_name)
            elif plugin_name in plugins_disabled:  # Plugin was disabled
                dest_path = dest_path.replace("plugins/" + plugin_name, "plugins/disabled-" + plugin_name)
            print("P", end=" ")

        dest_dir = os.path.dirname(dest_path)
        if dest_dir and not os.path.isdir(dest_dir):
            os.makedirs(dest_dir)

        if dest_dir != dest_path.strip("/"):
            update_paths[updatesite_path + "/" + inner_path] = dest_path

    num_ok = 0
    num_rename = 0
    num_error = 0
    for path_from, path_to in update_paths.items():
        print("-", path_from, "->", path_to)
        if not os.path.isfile(path_from):
            print("Missing file")
            continue

        data = open(path_from, "rb").read()

        try:
            open(path_to, 'wb').write(data)
            num_ok += 1
        except Exception as err:
            try:
                print("Error writing: %s. Renaming old file as workaround..." % err)
                path_to_tmp = path_to + "-old"
                if os.path.isfile(path_to_tmp):
                    os.unlink(path_to_tmp)
                os.rename(path_to, path_to_tmp)
                num_rename += 1
                open(path_to, 'wb').write(data)
                shutil.copymode(path_to_tmp, path_to)  # Copy permissions
                print("Write done after rename!")
                num_ok += 1
            except Exception as err:
                print("Write error after rename: %s" % err)
                num_error += 1
    print("* Updated files: %s, renamed: %s, error: %s" % (num_ok, num_rename, num_error))


```

这段代码是一个Python脚本，其作用是在运行脚本时执行一些操作，具体解释如下：

1. `if __name__ == "__main__":` 是一个if语句，判断当前是否是在脚本的主文件中运行，如果是，则执行下面的代码。

2. `sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))` 是一个Python内置函数，`sys.path` 是一个序列到 `__main__` 的引用，`os.path.dirname(__file__)` 获取当前文件所在的目录，`src` 目录是相对于当前文件所在的目录的名称，`os.path.join(os.path.dirname(__file__), "src")` 组合了上述目录和名称，最终返回当前目录和 `src` 目录的路径。`sys.path.insert(0, ...)` 将 `src` 目录添加到 `sys.path` 的开头，这样就可以在任何地方运行这个脚本了。

3. `update()` 是一个函数，具体的实现不在本次解释范围内。


```py
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))  # Imports relative to src

    update()

```

# `zeronet.py`

这段代码是一个用于管理 ZeroNet 服务器的脚本。它主要实现了以下功能：

1. 检查 Python 版本：如果版本小于 3，则输出错误并退出脚本。
2. 设置名为 "--silent" 的参数，如果该参数不在命令行中，则输出 "ZeroNet is starting"。
3. 启动 ZeroNet 服务器：如果尝试启动服务器时出现错误，则输出相应的错误信息。
4. 处理异常：捕获异常并输出错误信息，同时将异常信息记录在 error.log 文件中。
5. 在服务器启动后，将所有异常信息记录到 error.log 文件中：以便于在有新错误时方便地追踪和解决问题。
6. 如果服务器在运行时遇到意外情况（如关闭或重启），则在服务器重新启动后重新启动它。
7. 在服务器更新后，将所有更新信息记录到 error.log 文件中：方便日后分析。
8. 如果服务器需要重新启动，将在重启前输出通知信息。
9. 将 ZeroNet 的文档目录设置为配置文件中指定的目录，以便在需要时自动加载配置文件。


```py
#!/usr/bin/env python3
import os
import sys


def main():
    if sys.version_info.major < 3:
        print("Error: Python 3.x is required")
        sys.exit(0)

    if "--silent" not in sys.argv:
        print("- Starting ZeroNet...")

    main = None
    try:
        import main
        main.start()
    except Exception as err:  # Prevent closing
        import traceback
        try:
            import logging
            logging.exception("Unhandled exception: %s" % err)
        except Exception as log_err:
            print("Failed to log error:", log_err)
            traceback.print_exc()
        from Config import config
        error_log_path = config.log_dir + "/error.log"
        traceback.print_exc(file=open(error_log_path, "w"))
        print("---")
        print("Please report it: https://github.com/HelloZeroNet/ZeroNet/issues/new?assignees=&labels=&template=bug-report.md")
        if sys.platform.startswith("win") and "python.exe" not in sys.executable:
            displayErrorMessage(err, error_log_path)

    if main and (main.update_after_shutdown or main.restart_after_shutdown):  # Updater
        if main.update_after_shutdown:
            print("Shutting down...")
            prepareShutdown()
            import update
            print("Updating...")
            update.update()
            if main.restart_after_shutdown:
                print("Restarting...")
                restart()
        else:
            print("Shutting down...")
            prepareShutdown()
            print("Restarting...")
            restart()


```

这段代码是一个函数，名为`displayErrorMessage`，它接受两个参数，一个是`err`对象，另一个是错误日志文件路径。函数的作用是显示一个错误消息，并告知用户如何进一步报告该错误。

函数内部首先从`ctypes`和`urllib.parse`导入了一些库，然后使用`subprocess`库执行一个命令，该命令会运行一个名为`notepad.exe`的软件，并将错误日志文件路径作为参数传递给它。接着，函数内部使用`ctypes`库将错误对象转换为字符串，并将其与错误标题一起显示在屏幕上。

如果用户点击“是”，那么函数内部的`subprocess.Popen`函数会打开一个包含错误报告的URL，该URL使用`urllib.parse`库将错误消息编码为`问号和空格，并将其作为参数传递给`notepad.exe`命令。如果用户点击“否”，或者关闭了任何窗口，那么函数内部就不会执行任何操作。


```py
def displayErrorMessage(err, error_log_path):
    import ctypes
    import urllib.parse
    import subprocess

    MB_YESNOCANCEL = 0x3
    MB_ICONEXCLAIMATION = 0x30

    ID_YES = 0x6
    ID_NO = 0x7
    ID_CANCEL = 0x2

    err_message = "%s: %s" % (type(err).__name__, err)
    err_title = "Unhandled exception: %s\nReport error?" % err_message

    res = ctypes.windll.user32.MessageBoxW(0, err_title, "ZeroNet error", MB_YESNOCANCEL | MB_ICONEXCLAIMATION)
    if res == ID_YES:
        import webbrowser
        report_url = "https://github.com/HelloZeroNet/ZeroNet/issues/new?assignees=&labels=&template=bug-report.md&title=%s"
        webbrowser.open(report_url % urllib.parse.quote("Unhandled exception: %s" % err_message))
    if res in [ID_YES, ID_NO]:
        subprocess.Popen(['notepad.exe', error_log_path])

```

这段代码定义了一个名为 "prepareShutdown" 的函数，它在函数内部使用了三个步骤来准备系统关闭。

第一步，它导入了 atexit 模块，这是一个 Python 标准库中的模块，用于在程序启动时执行操作。在这个例子中，它调用了 atbase 函数中的 run_exitfuncs() 函数，这个函数会执行系统关闭时定义好的函数。

第二步，它关闭了 "main" 模块中的所有日志文件和对应的日志处理程序。这是因为在这个例子中，"main" 模块中的文件是使用 Python's built-in logging 模块进行处理的，而在这个函数中，我们关闭了这些日志文件和对应的日志处理程序。

第三步，它等待了一段时间之后，再次尝试关闭文件。这是因为我们在函数中使用了 time.sleep() 函数来让程序等待一段时间，以确保所有文件都已经被关闭。


```py
def prepareShutdown():
    import atexit
    atexit._run_exitfuncs()

    # Close log files
    if "main" in sys.modules:
        logger = sys.modules["main"].logging.getLogger()

        for handler in logger.handlers[:]:
            handler.flush()
            handler.close()
            logger.removeHandler(handler)

    import time
    time.sleep(1)  # Wait for files to close

```

这段代码是一个 Python 函数，名为 `restart()`。函数的作用是重新启动 Python 解释器并执行一些自定义操作，以使 Python 程序在重新启动后可以正常运行。

具体来说，这段代码做了以下几件事情：

1. 去掉命令行参数中的 `.pkg` 字段，使得 Python 程序在重新启动后不会自动加载依赖包。
2. 如果 `frozen` 属性为 `True`，则执行一些自定义操作以使程序在重新启动后可以正常运行。
3. 如果 `frozen` 属性为 `False`，则执行以下操作：

a. 去掉了 `--open_browser` 参数的前面 `sys.executable` 参数。

b. 删除了 `--open_browser` 参数。

c. 如果 `sys.platform` 是 `'win32'`，则使用引号将参数列表中的每个参数包裹起来。
4. 如果 `os.execv()` 函数出现错误，函数会捕获该错误并打印错误信息。
5. 输出 "Executing %s %s" 来显示重新启动的 Python 程序的路径和参数。
6. 输出 "Bye." 来表示程序的结束。


```py
def restart():
    args = sys.argv[:]

    sys.executable = sys.executable.replace(".pkg", "")  # Frozen mac fix

    if not getattr(sys, 'frozen', False):
        args.insert(0, sys.executable)

    # Don't open browser after restart
    if "--open_browser" in args:
        del args[args.index("--open_browser") + 1]  # argument value
        del args[args.index("--open_browser")]  # argument key

    if getattr(sys, 'frozen', False):
        pos_first_arg = 1  # Only the executable
    else:
        pos_first_arg = 2  # Interpter, .py file path

    args.insert(pos_first_arg, "--open_browser")
    args.insert(pos_first_arg + 1, "False")

    if sys.platform == 'win32':
        args = ['"%s"' % arg for arg in args]

    try:
        print("Executing %s %s" % (sys.executable, args))
        os.execv(sys.executable, args)
    except Exception as err:
        print("Execv error: %s" % err)
    print("Bye.")


```

这段代码是一个 Python 函数，名为 `start()`。函数的作用是让 Python 程序执行以下操作：

1. 将程序的工作目录（working directory）切换到包含 `zeroet.py` 文件的目录。
2. 将 `src/lib` 目录中的 Python 库添加到 `sys.path` 列表的第一个位置，以便程序可以导入该目录中的库。
3. 如果 `--update` 参数在命令行中，移除该参数，并输出 "Updating..." 消息，然后下载并安装零以致更新库。
4. 如果没有 `--update` 参数，程序将进入主函数。

`start()` 函数可以在程序的其他部分中被调用，以执行上述操作。由于它只是简单地将工作目录切换到正确的位置，因此它并不会对程序的执行产生实际的更改。


```py
def start():
    app_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(app_dir)  # Change working dir to zeronet.py dir
    sys.path.insert(0, os.path.join(app_dir, "src/lib"))  # External liblary directory
    sys.path.insert(0, os.path.join(app_dir, "src"))  # Imports relative to src

    if "--update" in sys.argv:
        sys.argv.remove("--update")
        print("Updating...")
        import update
        update.update()
    else:
        main()


```

这段代码是一个Python程序中的一个if语句，其中if语句的判断条件是(__name__ == '__main__')，如果这个条件为真，那么程序会执行if语句后面的代码块。if语句后面的代码块中包含了一个start()函数，这个函数的作用是启动一个名为“start”的服务器。因此，这段代码的作用是启动一个名为“start”的服务器，这个服务器可能在程序运行时提供一些额外的功能或者服务，具体实现可能因程序而异。


```py
if __name__ == '__main__':
    start()

```

---
name: Bug report
about: Create a report to help us improve ZeroNet
title: ''
labels: ''
assignees: ''

---

### Step 1: Please describe your environment

  * ZeroNet version: _____
  * Operating system: _____
  * Web browser: _____
  * Tor status: not available/always/disabled
  * Opened port: yes/no
  * Special configuration: ____

### Step 2: Describe the problem:

#### Steps to reproduce:

  1. _____
  2. _____
  3. _____

#### Observed Results:

  * What happened? This could be a screenshot, a description, log output (you can send log/debug.log file to hello@zeronet.io if necessary), etc.

#### Expected Results:

  * What did you expect to happen?


---
name: Feature request
about: Suggest an idea for ZeroNet
title: ''
labels: ''
assignees: ''

---

**Is your feature request related to a problem? Please describe.**
A clear and concise description of what the problem is. Ex. I'm always frustrated when [...]

**Describe the solution you'd like**
A clear and concise description of what you want to happen.

**Describe alternatives you've considered**
A clear and concise description of any alternative solutions or features you've considered.

**Additional context**
Add any other context or screenshots about the feature request here.

# client_qt —— LIBS AI Agent 的 Qt 上位机

这是整个项目的「脸」。它**不 import 任何 Python**，只是 Step 10 那个 HTTP 服务的客户端：
两个进程、两种语言，靠 JSON over HTTP 说话。

```
Qt 窗口  ──HTTP/JSON──▶  server/app.py (FastAPI)  ──▶  agent/ (决策)  ──▶  tools/ (计算)
```

**数字全部来自 `tools/` 里的纯 Python 计算**；大模型只负责决定调哪个工具、并把结果讲成人话。
客户端不做任何算术、不判合格与否。

## 编译

需要：CMake ≥ 3.16、C++17 编译器（MSVC）、Qt 5.15 或 Qt 6（Widgets + Network 两个模块）。

```bat
:: 编译 + 打包成能拷走的目录（产物在 dist\），首次建议用这个
_deploy.bat

:: 只是改了代码想快速验证，不打包（快很多）
_build.bat
```

工具链位置属于**机器配置**，不写在脚本里。二选一：

```bat
:: ① 建一个 local_env.bat 放在本目录（这个文件不会被提交 / 导出）
set LIBS_VS_DIR=<你的 Visual Studio 目录>\Conmmunity
set LIBS_QT_DIR=<你的 Qt 安装目录>

:: ② 或者每次在 shell 里设好环境变量
```

两个变量都没设时脚本会直接停下来并告诉你怎么填，不会拿一个猜出来的路径去编译。

## 运行

```bat
:: 开窗口（默认连 http://127.0.0.1:8000）
dist\libs_agent_client.exe

:: 后端在别的机器 / 端口
dist\libs_agent_client.exe --server http://192.168.1.20:8000

:: 无界面自检：只查 /health，不打大模型，不花钱
dist\libs_agent_client.exe --selftest

:: 自检 + 真的提一个问题（会消耗一次大模型调用）
dist\libs_agent_client.exe --selftest --ask "看看 data-304-1 那组数据的采集质量"

:: 开窗口并自动把这句话问出去（做演示/截图用，省得手点）
dist\libs_agent_client.exe --demo "现在都有哪些材料的数据？"

:: 界面状态与问答同时写一份日志
dist\libs_agent_client.exe --log D:\tmp\client.log
```

`--log` 在 GUI 和自检两种模式下都管用。**GUI 出问题时截图看不出它当时是什么状态**，
日志里那几行 `STATUS / ASK / CHAT ok / ANSWER` 才是证据。

### 为什么打包后还留着 `--selftest`？

桌面程序是 GUI 子系统（`WIN32_EXECUTABLE ON`），双击不弹黑框，
代价是**进程没有控制台**，`printf` 全掉进空气里。所以：

- 从命令行启动自检时，程序用 `AttachConsole(ATTACH_PARENT_PROCESS)` 把父进程的控制台借过来；
- 借不到（被脚本拉起 / 直接双击）就靠 `--log 文件` 落盘。

两条路都留，是为了「图形程序也永远能自证」。

## 四个文件的分工

| 文件 | 职责 |
|---|---|
| `src/main.cpp` | 参数解析 → 决定走 GUI 还是自检；自检的超时兜底 |
| `src/BackendClient.*` | 只会说 HTTP：发 `/health`、`/chat`，把结果翻译成信号。**不碰界面** |
| `src/MainWindow.*` | 只会画界面：状态栏、聊天区、输入行。**不碰网络细节** |
| `CMakeLists.txt` | Qt6 优先 / Qt5 兜底的查找逻辑；`/utf-8` 解决 MSVC 中文源码 |

界面层和协议层分开，是因为「协议」和「窗口」是两件会各自变化的事 ——
这样自检那条路能复用同一套请求逻辑，不必为了验证接口先起一个图形界面。

## 会话怎么管

客户端**只存一个 `session_id`**，对话历史（含思考模型的 `reasoning_content`）全在服务端内存里。
服务端一重启，客户端带着旧 id 过去，服务端会**新建一个会话**而不是报错 —— 客户端不必知道服务端重启过。
「新会话」按钮只是把本地那个 id 丢掉，服务端那份靠「最久没用过」自动淘汰（上限 32 个）。

详见 `server/app.py` 开头那六条设计决定。

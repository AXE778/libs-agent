cmake_minimum_required(VERSION 3.16)

project(LibsAgentClient VERSION 0.1.0 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Step 11：LIBS AI Agent 的 Qt 上位机
#
# 它不 import 任何 Python —— 它只是 Step 10 那个 HTTP 服务的客户端。
# 两个进程、两种语言，靠 JSON over HTTP 说话。
# ---------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Qt 的元对象编译器（moc）要处理 Q_OBJECT 宏，uic/rcc 处理 .ui / 资源文件。
# 打开这三个开关，就不用手写 qt_wrap_cpp 那一套了。
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# MSVC 默认按「系统代码页（GBK）」读源码，源码里的中文会变成乱码甚至编译报错。
# /utf-8 让它按 UTF-8 读 —— 一句话解决，别去折腾 #pragma execution_character_set。
if (MSVC)
    add_compile_options(/utf-8)
endif()

# Qt6 优先、Qt5 兜底：本机装的是 Qt 5.15.2，但以后换 6 不用改这个文件。
find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets Network)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets Network)

message(STATUS "使用 Qt ${QT_VERSION} （${QT_VERSION_MAJOR} 系列）")

add_executable(libs_agent_client
    src/main.cpp
    src/MainWindow.h
    src/MainWindow.cpp
    src/BackendClient.h
    src/BackendClient.cpp
)

# Widgets：窗口控件；Network：QNetworkAccessManager（就是 qt 版的 requests）
target_link_libraries(libs_agent_client
    PRIVATE
        Qt${QT_VERSION_MAJOR}::Widgets
        Qt${QT_VERSION_MAJOR}::Network
)

# 打包成桌面程序：不加这个，双击 exe 会先弹一个黑乎乎的控制台窗口。
# 代价是进程不再自带控制台 —— 所以 main.cpp 里自检那条路要用
# AttachConsole 借父进程的控制台，否则 --selftest 会一句话都打不出来。
set_target_properties(libs_agent_client PROPERTIES WIN32_EXECUTABLE ON)

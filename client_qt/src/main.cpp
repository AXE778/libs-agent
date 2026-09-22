#include "BackendClient.h"
#include "MainWindow.h"

#include <QApplication>
#include <QCoreApplication>
#include <QFile>
#include <QJsonArray>
#include <QJsonObject>
#include <QStringList>
#include <QTimer>
#include <QUrl>

#include <cstdio>

#ifdef _WIN32
#  include <windows.h>
#endif

// main.cpp —— 程序的入口，只干三件事：
//   ① 解析命令行参数（不自作主张去猜配置）；
//   ② 有 --selftest 就走无界面自检（给自动化 / 排障用）；
//   ③ 否则开窗口。
//
// 为什么自检必须单独做一条路，而不是「开窗口然后点按钮」？
//   图形界面没法在脚本里验收 —— 编译通过不等于请求发得出去、
//   JSON 字段名拼得对。自检把「协议这一层」变成可以看 stdout 的东西，
//   界面好不好看留着人来判断。

namespace {

const char *kDefaultServer = "http://127.0.0.1:8000";

// 打包之后这个程序是 GUI 子系统（WIN32_EXECUTABLE ON）：双击不弹黑框，
// 代价是**进程没有控制台**，printf 全掉进空气里 —— 自检就没法用了。
//
// 解法：从命令行（cmd / Git Bash）带 --selftest 启动时，把自己挂到
// **父进程的控制台**上，再把 stdout/stderr 指回 "CONOUT$"。
// 这样同一个 exe 既能当纯 GUI 程序双击，又保留了可脚本化的自检。
// （双击时不带 --selftest，根本不会走到这里，所以也不会凭空冒出黑框。）
#ifdef _WIN32
void attachParentConsole()
{
    if (!AttachConsole(ATTACH_PARENT_PROCESS))
        return;                      // 父进程没有控制台（比如双击）—— 那就作罢
    FILE *f = nullptr;
    freopen_s(&f, "CONOUT$", "w", stdout);
    freopen_s(&f, "CONOUT$", "w", stderr);
}
#endif

struct Options
{
    QString server = QString::fromLatin1(kDefaultServer);
    bool    selftest = false;
    QString question;          // 自检时顺带问一句
    QString logPath;           // 自检 / 界面输出同时写一份到这个文件
    QString demo;              // 非空 = 开窗口，并且自动问这一句
    bool    showHelp = false;
    QString badReason;         // 非空 = 参数写错了
};

// 自检输出同时落一份到文件时用（见 --log）。
// 为什么需要它：打包后的程序是 GUI 子系统，没有控制台，
// 如果是被别的程序（脚本、CI、编辑器）拉起来的，连 AttachConsole 都借不到 ——
// 那时候「看不见输出」就等于「没法排障」。写文件是最不依赖环境的兜底。
QFile *g_log = nullptr;

// 直接写 stdout（不走 qDebug）。
// qDebug 到 stderr 且带引号包装，脚本里不好解析；这里要的是「能 grep 的纯文本」。
void say(const QString &text)
{
    const QByteArray bytes = text.toUtf8();
    std::fwrite(bytes.constData(), 1, size_t(bytes.size()), stdout);
    std::fflush(stdout);

    if (g_log) {
        g_log->write(bytes);
        g_log->flush();     // 崩了也要留下最后几行 —— 日志的价值就在最后一屏
    }
}

void printHelp()
{
    say(QStringLiteral(
        "LIBS AI Agent —— Qt 客户端\n"
        "\n"
        "用法：\n"
        "  libs_agent_client.exe                        开窗口（默认连 %1）\n"
        "  libs_agent_client.exe --server http://a:8000 指定后端地址\n"
        "  libs_agent_client.exe --selftest             无界面自检：只查 /health\n"
        "  libs_agent_client.exe --selftest --ask 问题  自检并真的提一个问题\n"
        "  libs_agent_client.exe --selftest --log 文件  自检结果同时写进这个文件\n"
        "  libs_agent_client.exe --log 文件             开窗口，并把界面状态与问答写进日志\n"
        "  libs_agent_client.exe --demo 问题           开窗口，并自动把这个问题问出去\n"
        "  libs_agent_client.exe --help                 看这段说明\n"
        "\n"
        "环境变量 LIBS_API_URL 也可以指定后端地址（命令行优先）。\n"
        "注意：--selftest 不打开窗口，结果打在标准输出，适合脚本和排障。\n"
        "打包后的程序没有控制台，输出看不见时加 --log 文件。\n")
        .arg(QString::fromLatin1(kDefaultServer)));
}

Options parseOptions(const QStringList &args)
{
    Options opt;

    // 环境变量兜底：容器 / CI 里不方便传参数时用得上。
    const QByteArray env = qgetenv("LIBS_API_URL");
    if (!env.isEmpty())
        opt.server = QString::fromLocal8Bit(env);

    for (int i = 0; i < args.size(); ++i) {
        const QString &a = args.at(i);

        const bool needsValue = (a == QLatin1String("--server")
                                 || a == QLatin1String("--ask")
                                 || a == QLatin1String("--log")
                                 || a == QLatin1String("--demo"));
        if (needsValue && i + 1 >= args.size()) {
            opt.badReason = QStringLiteral("%1 后面缺一个值").arg(a);
            return opt;
        }

        if (a == QLatin1String("--help") || a == QLatin1String("-h")) {
            opt.showHelp = true;
        } else if (a == QLatin1String("--selftest")) {
            opt.selftest = true;
        } else if (a == QLatin1String("--server")) {
            opt.server = args.at(++i);
        } else if (a.startsWith(QLatin1String("--server="))) {
            opt.server = a.mid(QStringLiteral("--server=").size());
        } else if (a == QLatin1String("--ask")) {
            opt.question = args.at(++i);
        } else if (a.startsWith(QLatin1String("--ask="))) {
            opt.question = a.mid(QStringLiteral("--ask=").size());
        } else if (a == QLatin1String("--log")) {
            opt.logPath = args.at(++i);
        } else if (a.startsWith(QLatin1String("--log="))) {
            opt.logPath = a.mid(QStringLiteral("--log=").size());
        } else if (a == QLatin1String("--demo")) {
            opt.demo = args.at(++i);
        } else if (a.startsWith(QLatin1String("--demo="))) {
            opt.demo = a.mid(QStringLiteral("--demo=").size());
        } else {
            opt.badReason = QStringLiteral("看不懂的参数：%1").arg(a);
            return opt;
        }
    }

    // 只有 --ask 没有 --selftest 时，默认就是想自检一下，别让人写两遍。
    if (!opt.question.isEmpty())
        opt.selftest = true;

    const QUrl url(opt.server);
    if (!url.isValid() || url.scheme().isEmpty() || url.host().isEmpty()) {
        opt.badReason = QStringLiteral("后端地址不像个地址：%1（例：http://127.0.0.1:8000）")
                            .arg(opt.server);
    }
    return opt;
}

void printSteps(const QJsonArray &steps)
{
    if (steps.isEmpty()) {
        say(QStringLiteral("  （这一轮没有调用任何工具）\n"));
        return;
    }
    for (const QJsonValue &v : steps) {
        const QJsonObject s = v.toObject();
        say(QStringLiteral("  %1. %2  %3  %4 秒\n")
                .arg(QString::number(s.value(QStringLiteral("round")).toInt()),
                     s.value(QStringLiteral("tool")).toString(),
                     s.value(QStringLiteral("ok")).toBool() ? QStringLiteral("ok") : QStringLiteral("失败"))
                .arg(s.value(QStringLiteral("seconds")).toDouble(), 0, 'f', 2));
    }
}

// 自检：这条路上不会打开任何窗口，全部结果都在 stdout。
int runSelfTest(const Options &opt)
{
    say(QStringLiteral("== libs_agent_client 自检 ==\n"));
    say(QStringLiteral("后端 %1\n").arg(opt.server));

    BackendClient client;
    client.setBaseUrl(QUrl(opt.server));

    int code = 0;

    QObject::connect(&client, &BackendClient::healthReady, [&](const QJsonObject &info) {
        say(QStringLiteral("HEALTH OK  provider=%1  model=%2  tools=%3  api_key_configured=%4\n")
                .arg(info.value(QStringLiteral("provider")).toString(),
                     info.value(QStringLiteral("model")).toString(),
                     QString::number(info.value(QStringLiteral("n_tools")).toInt()),
                     info.value(QStringLiteral("api_key_configured")).toBool()
                         ? QStringLiteral("true") : QStringLiteral("false")));
        if (!info.value(QStringLiteral("api_key_configured")).toBool())
            say(QStringLiteral("  提示：服务端没配 Key，/chat 会返回 502。\n"));

        if (opt.question.isEmpty()) {
            say(QStringLiteral("（没给 --ask，自检到此结束）\n"));
            QCoreApplication::exit(code);
            return;
        }

        say(QStringLiteral("正在提问（一次可能要 20 秒以上）…\n"));
        client.sendChat(opt.question);
    });

    QObject::connect(&client, &BackendClient::healthFailed, [&](const QString &error) {
        say(QStringLiteral("HEALTH FAIL  %1\n").arg(error));
        say(QStringLiteral("排障：先起服务 → .venv\\Scripts\\python.exe -m server.app\n"));
        code = 2;
        QCoreApplication::exit(code);
    });

    QObject::connect(&client, &BackendClient::chatReady, [&](const QJsonObject &resp) {
        say(QStringLiteral("CHAT OK  session_id=%1  stop=%2  n_tools=%3  elapsed=%4 秒\n")
                .arg(resp.value(QStringLiteral("session_id")).toString(),
                     resp.value(QStringLiteral("stop")).toString(),
                     QString::number(resp.value(QStringLiteral("n_tools")).toInt()))
                .arg(resp.value(QStringLiteral("elapsed_s")).toDouble(), 0, 'f', 2));
        printSteps(resp.value(QStringLiteral("steps")).toArray());
        say(QStringLiteral("--- 回答 ---\n"));
        say(resp.value(QStringLiteral("answer")).toString() + QStringLiteral("\n"));
        QCoreApplication::exit(code);
    });

    QObject::connect(&client, &BackendClient::chatFailed, [&](const QString &error) {
        say(QStringLiteral("CHAT FAIL  %1\n").arg(error));
        code = 3;
        QCoreApplication::exit(code);
    });

    // 兜底超时。QNetworkAccessManager 默认不设超时（Qt 5.15 起要显式 setTransferTimeout），
    // 若服务端卡住，自检会呆等 —— 脚本里最忌讳这个，所以自己盯一刀。
    QTimer::singleShot(180000, [] {
        say(QStringLiteral("TIMEOUT  180 秒还没回来，放弃。\n"));
        QCoreApplication::exit(4);
    });

    client.checkHealth();
    return QCoreApplication::exec();
}

} // namespace

int main(int argc, char *argv[])
{
    // ★ 参数必须在构造 QApplication 之前解析完：
    //   自检模式要用 QCoreApplication（不需要图形界面，无头环境也能跑）。
    QStringList args;
    args.reserve(argc - 1);
    for (int i = 1; i < argc; ++i)
        args << QString::fromLocal8Bit(argv[i]);

    const Options opt = parseOptions(args);

    if (opt.showHelp) {
        printHelp();
        return 0;
    }
    if (!opt.badReason.isEmpty()) {
        say(QStringLiteral("参数错误：%1\n\n").arg(opt.badReason));
        printHelp();
        return 2;
    }

    if (opt.selftest) {
#ifdef _WIN32
        attachParentConsole();   // 有父控制台就借来用（cmd 里跑时管用）
#endif
        QCoreApplication app(argc, argv);

        // 再挂一份文件输出。两者不冲突：有控制台就两边都看得到，
        // 没有控制台（被脚本拉起 / 打包后双击）时文件就是唯一的证据。
        QFile logFile;
        if (!opt.logPath.isEmpty()) {
            logFile.setFileName(opt.logPath);
            if (logFile.open(QIODevice::WriteOnly | QIODevice::Truncate))
                g_log = &logFile;
        }

        const int rc = runSelfTest(opt);

        g_log = nullptr;         // 别留一个指向已析构对象的悬空指针
        if (logFile.isOpen())
            logFile.close();
        return rc;
    }

    QApplication app(argc, argv);
    // GUI 也支持 --log：界面一出问题，光看截图看不出它当时是什么状态。
    MainWindow window(opt.server, opt.logPath);
    window.setAutoAsk(opt.demo);
    window.show();
    return app.exec();
}

#pragma once

#include <QMainWindow>
#include <QString>

class QTextBrowser;
class QLineEdit;
class QPushButton;
class QLabel;
class QJsonObject;
class QTimer;
class QFile;
class BackendClient;

// MainWindow —— Step 11 的主窗口。
//
// 第 1 小步：只有界面骨架。
// 第 2 小步：接通 /health（状态栏显示后端状态）。
// 第 3 小步（本文件当前状态）：接通 /chat —— 提问、显示回答、显示调了哪些工具。
// 第 4 小步：打包（windeployqt），并补上 --log 与 --demo。
class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    // baseUrl 由 main.cpp 传入（命令行 --server 可覆盖），窗口自己不猜配置。
    // logPath 非空时，界面上的关键动态（状态栏文字、提问、回答）会同时写一份日志 ——
    // 窗口一出问题就只能截屏，而截屏看不出「它当时到底是什么状态」。
    explicit MainWindow(const QString &baseUrl,
                        const QString &logPath = QString(),
                        QWidget *parent = nullptr);
    ~MainWindow() override;

    // 演示/验收用：窗口一开、后端一确认，就自动把这句话问出去。
    // 为什么要它：想验证「界面上真能问出答案」就得点按钮，而自动化去抢前台焦点
    // 既不可靠、又有误操作风险。让程序自己走一遍同一条代码路径最干净。
    void setAutoAsk(const QString &question);

private slots:
    void onSendClicked();
    void onCheckBackend();
    void onNewSession();
    void onHealthReady(const QJsonObject &info);
    void onHealthFailed(const QString &error);
    void onChatReady(const QJsonObject &response);
    void onChatFailed(const QString &error);
    void onChatTimeout();

private:
    void buildUi();
    void appendSay(const QString &who, const QString &text, const QString &colorCss);
    void appendNote(const QString &html);
    void setStatusText(const QString &text, const QString &colorCss);
    void setBusy(bool busy);
    void logLine(const QString &text);

    BackendClient *m_backend = nullptr;

    QTextBrowser *m_chat = nullptr;
    QLineEdit *m_input = nullptr;
    QPushButton *m_send = nullptr;
    QPushButton *m_ping = nullptr;
    QPushButton *m_new = nullptr;
    QLabel *m_status = nullptr;
    QTimer *m_chatGuard = nullptr;

    bool m_healthOk = false;
    // 服务端生成、客户端原样带回。对话历史（含 reasoning_content）全在服务端，
    // 客户端只拿这一个 id —— 详见 server/app.py 开头第 ① 条设计决定。
    QString m_sessionId;
    QFile *m_log = nullptr;

    QString m_autoAsk;          // 非空 = 后端一连上就自动问这句
    bool m_autoAskDone = false;
};

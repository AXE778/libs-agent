#include "MainWindow.h"
#include "BackendClient.h"

#include <QDateTime>
#include <QFile>
#include <QHBoxLayout>
#include <QJsonArray>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QStatusBar>
#include <QStringList>
#include <QTextBrowser>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>

// 一次提问要等多久算「不对劲」。服务端自己会等限流重试（最坏几十秒），
// 所以这个值要明显大于正常耗时；它只是兜底，不是超时策略。
static const int kChatGuardMs = 180000;

MainWindow::MainWindow(const QString &baseUrl, const QString &logPath, QWidget *parent)
    : QMainWindow(parent)
{
    if (!logPath.isEmpty()) {
        // 追加而不是覆盖：实验室里连着开几次，日志应该是连起来的。
        m_log = new QFile(logPath, this);
        if (!m_log->open(QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text)) {
            delete m_log;
            m_log = nullptr;      // 写不进去就算了，绝不能因为日志打不开就开不了窗口
        } else {
            logLine(QStringLiteral("=== 客户端启动　GUI　后端 %1 ===").arg(baseUrl));
        }
    }

    m_backend = new BackendClient(this);
    m_backend->setBaseUrl(QUrl(baseUrl));

    connect(m_backend, &BackendClient::healthReady,  this, &MainWindow::onHealthReady);
    connect(m_backend, &BackendClient::healthFailed, this, &MainWindow::onHealthFailed);
    connect(m_backend, &BackendClient::chatReady,    this, &MainWindow::onChatReady);
    connect(m_backend, &BackendClient::chatFailed,   this, &MainWindow::onChatFailed);

    buildUi();

    // 启动即探一次后端，用户不必先点一次按钮。
    // 用 singleShot 而不是直接调用：让窗口先画出来，否则界面会「卡在空白处等网络」。
    QTimer::singleShot(200, this, &MainWindow::onCheckBackend);
}

MainWindow::~MainWindow()
{
    if (m_log) {
        logLine(QStringLiteral("=== 客户端退出 ==="));
        m_log->close();           // m_log 是 this 的孩子，会被 Qt 自己回收
    }
}

void MainWindow::logLine(const QString &text)
{
    if (!m_log || !m_log->isOpen())
        return;
    const QString stamp = QDateTime::currentDateTime().toString(QStringLiteral("yyyy-MM-dd HH:mm:ss"));
    m_log->write(QStringLiteral("[%1] %2\n").arg(stamp, text).toUtf8());
    m_log->flush();
}

void MainWindow::setAutoAsk(const QString &question)
{
    m_autoAsk = question.trimmed();
}

void MainWindow::buildUi()
{
    setWindowTitle(QStringLiteral("LIBS AI Agent —— 客户端"));
    resize(880, 620);

    auto *central = new QWidget(this);
    auto *root = new QVBoxLayout(central);
    root->setContentsMargins(12, 12, 12, 10);
    root->setSpacing(8);

    m_chat = new QTextBrowser(central);
    m_chat->setOpenExternalLinks(true);
    m_chat->setStyleSheet(QStringLiteral(
        "QTextBrowser{background:#FAFAF8;border:1px solid #D8D6CE;"
        "border-radius:8px;padding:10px;font-size:13px;}"));
    root->addWidget(m_chat, 1);

    // ---- 输入行 ----
    auto *row = new QHBoxLayout();
    row->setSpacing(8);

    m_input = new QLineEdit(central);
    m_input->setPlaceholderText(
        QStringLiteral("例：看看 data-304-1 那组数据的采集质量　（回车发送）"));
    m_input->setStyleSheet(QStringLiteral(
        "QLineEdit{border:1px solid #C9C7BF;border-radius:8px;padding:8px 10px;"
        "font-size:13px;background:#FFFFFF;}"));
    row->addWidget(m_input, 1);

    m_send = new QPushButton(QStringLiteral("发送"), central);
    m_send->setStyleSheet(QStringLiteral(
        "QPushButton{background:#EEEDFE;color:#26215C;border:1px solid #534AB7;"
        "border-radius:8px;padding:8px 18px;font-size:13px;}"
        "QPushButton:disabled{color:#9A9A96;border-color:#D8D6CE;background:#F2F1EC;}"));
    row->addWidget(m_send);

    m_new = new QPushButton(QStringLiteral("新会话"), central);
    m_new->setStyleSheet(QStringLiteral(
        "QPushButton{background:#F2F1EC;color:#3D3D3A;border:1px solid #C9C7BF;"
        "border-radius:8px;padding:8px 14px;font-size:13px;}"));
    row->addWidget(m_new);

    m_ping = new QPushButton(QStringLiteral("检查后端"), central);
    m_ping->setStyleSheet(QStringLiteral(
        "QPushButton{background:#F2F1EC;color:#3D3D3A;border:1px solid #C9C7BF;"
        "border-radius:8px;padding:8px 14px;font-size:13px;}"));
    row->addWidget(m_ping);

    root->addLayout(row);
    setCentralWidget(central);

    // ---- 状态栏 ----
    m_status = new QLabel(central);
    statusBar()->addWidget(m_status);
    statusBar()->setStyleSheet(QStringLiteral("QStatusBar{background:#F5F4EF;}"));
    setStatusText(QStringLiteral("后端：尚未检查"), QStringLiteral("color:#888780;"));

    // 兜底：万一服务端卡住，界面别一直锁在「处理中」。
    m_chatGuard = new QTimer(this);
    m_chatGuard->setSingleShot(true);
    m_chatGuard->setInterval(kChatGuardMs);
    connect(m_chatGuard, &QTimer::timeout, this, &MainWindow::onChatTimeout);

    connect(m_send,  &QPushButton::clicked, this, &MainWindow::onSendClicked);
    connect(m_new,   &QPushButton::clicked, this, &MainWindow::onNewSession);
    connect(m_ping,  &QPushButton::clicked, this, &MainWindow::onCheckBackend);
    connect(m_input, &QLineEdit::returnPressed, this, &MainWindow::onSendClicked);

    appendNote(QStringLiteral(
        "<div style='color:#888780;font-size:12px;line-height:1.7'>"
        "Step 11 客户端。数字全部由服务端 <code>tools/</code> 里的 Python 算出来，"
        "模型只负责决定调哪个工具、并把结果讲成人话。<br>"
        "一次提问通常要 20 秒以上（要调多次模型），这是正常的。"
        "</div>"));
}

void MainWindow::appendSay(const QString &who, const QString &text, const QString &colorCss)
{
    // 用户输入 / 模型回答都可能带 < > &，必须先转义再过 HTML，
    // 否则一句「a < b」就能把后面的排版吃掉。
    const QString safe = text.toHtmlEscaped().replace(QLatin1Char('\n'), QStringLiteral("<br>"));
    m_chat->append(QStringLiteral(
        "<div style='margin:6px 0;line-height:1.7;color:%1'>"
        "<b>%2</b>　%3</div>")
        .arg(colorCss, who.toHtmlEscaped(), safe));
}

void MainWindow::appendNote(const QString &html)
{
    // 这个入口只给「代码里自己写的」HTML 用，不过用户输入。
    m_chat->append(html);
}

void MainWindow::setStatusText(const QString &text, const QString &colorCss)
{
    m_status->setText(text);
    m_status->setStyleSheet(QStringLiteral("font-size:12px;padding:2px 4px;") + colorCss);
    logLine(QStringLiteral("STATUS %1").arg(text));
}

void MainWindow::setBusy(bool busy)
{
    m_send->setEnabled(!busy);
    m_input->setEnabled(!busy);
    m_new->setEnabled(!busy);
    m_send->setText(busy ? QStringLiteral("处理中…") : QStringLiteral("发送"));
}

// ---------------------------------------------------------------------------
// 后端健康
// ---------------------------------------------------------------------------
void MainWindow::onCheckBackend()
{
    setStatusText(QStringLiteral("后端：正在检查…"), QStringLiteral("color:#854F0B;"));
    m_backend->checkHealth();
}

void MainWindow::onHealthReady(const QJsonObject &info)
{
    m_healthOk = true;

    const QString provider = info.value(QStringLiteral("provider")).toString();
    const QString model    = info.value(QStringLiteral("model")).toString();
    const int nTools       = info.value(QStringLiteral("n_tools")).toInt();
    const bool keyOk       = info.value(QStringLiteral("api_key_configured")).toBool();
    const QString base     = m_backend->baseUrl().toString();

    setStatusText(QStringLiteral("后端：已连接　%1 · %2 · %3 个工具")
                      .arg(provider, model).arg(nTools),
                  QStringLiteral("color:#0F6E4F;"));

    appendNote(QStringLiteral(
        "<div style='margin:6px 0;color:#854F0B;font-size:12px;line-height:1.7'>"
        "（后端已连接：%1　服务商 %2　模型 %3　工具 %4 个）</div>")
        .arg(base.toHtmlEscaped(), provider.toHtmlEscaped(),
             model.toHtmlEscaped(), QString::number(nTools)));

    if (!keyOk) {
        // /health 只检查「Key 填了没有」，不验证有效性 —— 验证要花一次真实请求，
        // 而账号每分钟只有 3 次，不该浪费在健康检查上。
        appendNote(QStringLiteral(
            "<div style='margin:6px 0;color:#A32D2D;font-size:12px;line-height:1.7'>"
            "⚠ 服务端没有读到 API Key（<code>.env</code> 里没填或还是占位值）。"
            "服务能起来，但一问就会返回 502。</div>"));
        logLine(QStringLiteral("WARN 服务端未配置 API Key"));
    }

    // 演示模式：后端一确认就连着走一遍「填进输入框 → 点发送」的同一路径，
    // 不是绕过界面直接调 BackendClient —— 否则验的就不是界面了。
    if (!m_autoAsk.isEmpty() && !m_autoAskDone) {
        m_autoAskDone = true;     // 「检查后端」可以点很多次，只自动问一次
        const QString q = m_autoAsk;
        QTimer::singleShot(400, this, [this, q] {
            m_input->setText(q);
            onSendClicked();
        });
    }
}

void MainWindow::onHealthFailed(const QString &error)
{
    m_healthOk = false;

    setStatusText(QStringLiteral("后端：连不上"), QStringLiteral("color:#A32D2D;"));

    appendNote(QStringLiteral(
        "<div style='margin:6px 0;color:#A32D2D;font-size:12px;line-height:1.8'>"
        "✗ 连不上后端：%1<br>"
        "先在项目目录里把服务起起来：<br>"
        "<code>cd <项目根目录></code><br>"
        "<code>.venv\\Scripts\\python.exe -m server.app</code><br>"
        "然后点「检查后端」。若服务在别的机器/端口，用 "
        "<code>--server http://主机:端口</code> 启动本客户端。</div>")
        .arg(error.toHtmlEscaped()));

    logLine(QStringLiteral("ERROR 连不上后端：%1").arg(error));
}

// ---------------------------------------------------------------------------
// 提问
// ---------------------------------------------------------------------------
void MainWindow::onSendClicked()
{
    const QString text = m_input->text().trimmed();
    if (text.isEmpty())
        return;

    appendSay(QStringLiteral("你"), text, QStringLiteral("#185FA5"));
    m_input->clear();

    if (!m_healthOk) {
        appendNote(QStringLiteral(
            "<div style='margin:6px 0;color:#A32D2D;font-size:12px'>"
            "（后端未连接，先点「检查后端」确认为什么连不上。）</div>"));
        logLine(QStringLiteral("SKIP 后端未连接，未发送：%1").arg(text));
        return;
    }

    logLine(QStringLiteral("ASK %1").arg(text));

    setBusy(true);
    setStatusText(QStringLiteral("后端：正在处理这一句…（可能要 20 秒以上）"),
                  QStringLiteral("color:#854F0B;"));
    m_chatGuard->start();
    m_backend->sendChat(text, m_sessionId);
}

void MainWindow::onChatReady(const QJsonObject &response)
{
    m_chatGuard->stop();
    setBusy(false);

    // 服务端返回的 id 必须存下来，下一句带上它才接得上上下文。
    m_sessionId = response.value(QStringLiteral("session_id")).toString();

    const QString stop = response.value(QStringLiteral("stop")).toString();
    const int nTools   = response.value(QStringLiteral("n_tools")).toInt();
    const double secs  = response.value(QStringLiteral("elapsed_s")).toDouble();
    const QString note = response.value(QStringLiteral("note")).toString();
    const QString answer = response.value(QStringLiteral("answer")).toString();

    appendSay(QStringLiteral("Agent"), answer, QStringLiteral("#04342C"));

    // ---- 这一轮的「过程」信息 ----
    QString html = QStringLiteral(
        "<div style='margin:2px 0 10px 0;color:#854F0B;font-size:12px;line-height:1.7'>"
        "（调了 %1 个工具，耗时 %2 秒，会话 %3）")
        .arg(QString::number(nTools))
        .arg(secs, 0, 'f', 2)
        .arg(m_sessionId.toHtmlEscaped());

    QStringList chain;
    const QJsonArray steps = response.value(QStringLiteral("steps")).toArray();
    for (const QJsonValue &v : steps) {
        const QJsonObject s = v.toObject();
        const bool ok = s.value(QStringLiteral("ok")).toBool();
        chain << QStringLiteral("%1%2(%3s)")
                     .arg(s.value(QStringLiteral("tool")).toString(),
                          ok ? QString() : QStringLiteral("失败"))
                     .arg(s.value(QStringLiteral("seconds")).toDouble(), 0, 'f', 1);
    }
    if (!chain.isEmpty())
        html += QStringLiteral("<br>工具调用链：") + chain.join(QStringLiteral(" → ")).toHtmlEscaped();
    html += QStringLiteral("</div>");
    appendNote(html);

    if (!note.isEmpty()) {
        appendNote(QStringLiteral(
            "<div style='margin:2px 0 10px 0;color:#A32D2D;font-size:12px'>⚠ %1</div>")
            .arg(note.toHtmlEscaped()));
    }

    setStatusText(QStringLiteral("后端：已连接　上次耗时 %1 秒").arg(secs, 0, 'f', 1),
                  QStringLiteral("color:#0F6E4F;"));

    logLine(QStringLiteral("CHAT ok session=%1 stop=%2 tools=%3 elapsed=%4s 链=%5")
                .arg(m_sessionId, stop, QString::number(nTools))
                .arg(secs, 0, 'f', 2)
                .arg(chain.isEmpty() ? QStringLiteral("-") : chain.join(QStringLiteral(" -> "))));
    logLine(QStringLiteral("ANSWER %1").arg(answer));
}

void MainWindow::onChatFailed(const QString &error)
{
    m_chatGuard->stop();
    setBusy(false);

    appendNote(QStringLiteral(
        "<div style='margin:6px 0;color:#A32D2D;font-size:12px;line-height:1.8'>"
        "✗ 这一句没答上来：%1</div>")
        .arg(error.toHtmlEscaped()));

    // 409 = 同一会话上一句还在跑。服务端把原因和人话都写在 detail 里了，
    // 客户端只要照着念一遍 —— 不要在客户端自己编一套判据。
    if (error.contains(QStringLiteral("409"))) {
        appendNote(QStringLiteral(
            "<div style='margin:0 0 10px 0;color:#888780;font-size:12px'>"
            "（这个会话上一句还在处理，等它答完再发；或者点「新会话」另开一条。）</div>"));
    }

    setStatusText(QStringLiteral("后端：这一句失败了"), QStringLiteral("color:#A32D2D;"));
    logLine(QStringLiteral("CHAT FAIL %1").arg(error));
}

void MainWindow::onChatTimeout()
{
    // 不 abort 请求（BackendClient 那边会继续等）：万一它一会儿回来了，
    // 答案照样会显示出来。这里只负责把界面解锁，不假装失败。
    setBusy(false);
    appendNote(QStringLiteral(
        "<div style='margin:6px 0;color:#A32D2D;font-size:12px;line-height:1.8'>"
        "⚠ 已经等了 180 秒还没回。界面先解锁了；如果稍后收到回复会直接显示出来。"
        "若一直没动静，去看服务端控制台（大概率是撞限流或网络卡住）。</div>"));

    setStatusText(QStringLiteral("后端：等待超时（请求仍在进行）"),
                  QStringLiteral("color:#A32D2D;"));
    logLine(QStringLiteral("TIMEOUT 180 秒未收到回复"));
}

void MainWindow::onNewSession()
{
    // 只丢掉本地这个 id。服务端那份会话留在内存里，超出上限（32 个）会
    // 按「最久没用过」自动淘汰 —— 实验室单机场景不需要客户端主动回收。
    m_sessionId.clear();
    m_chat->clear();
    setStatusText(QStringLiteral("后端：已连接　（已开新会话）"),
                  QStringLiteral("color:#0F6E4F;"));
    appendNote(QStringLiteral(
        "<div style='color:#888780;font-size:12px'>（已开新会话，下一句不带上下文）</div>"));
    logLine(QStringLiteral("NEWSESSION 本地会话 id 已清空"));
}

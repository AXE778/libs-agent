#pragma once

// BackendClient —— 与 Step 10 的 HTTP 服务说话的那一层。
//
// 为什么不把网络代码直接写进 MainWindow？
//   因为「窗口」和「协议」是两件不同的事。分开之后：
//     · 界面改版不会碰到协议代码；
//     · 命令行自检（main.cpp 的 --selftest）可以复用同一套请求逻辑，
//       不必为了验证接口而先起一个图形界面。
//   这跟 Python 侧「agent/ 只管决策、tools/ 只管计算」是同一种切法。
//
// 设计约定
//   · 异步、信号槽回调 —— 一次 /chat 可能等 20 秒以上，绝不能阻塞界面线程；
//   · 这个类不弹窗、不写日志、不改界面，只负责「发请求 → 把结果翻译成信号」；
//   · 出错时也走信号（healthFailed / chatFailed），不抛异常、不 qFatal。
//     Qt 的信号槽没有异常通道，把错误做成信号是这里唯一自洽的做法。

#include <QObject>
#include <QUrl>
#include <QJsonObject>
#include <QString>

class QNetworkAccessManager;
class QNetworkReply;

class BackendClient : public QObject
{
    Q_OBJECT

public:
    explicit BackendClient(QObject *parent = nullptr);

    void setBaseUrl(const QUrl &url);       // 例如 http://127.0.0.1:8000
    QUrl baseUrl() const { return m_base; }

    // GET /health —— 不调用大模型，毫秒级，可以随便点
    void checkHealth();

    // POST /chat —— 一次请求返回完整答案；sessionId 传空 = 新会话
    void sendChat(const QString &message, const QString &sessionId = QString());

signals:
    void healthReady(const QJsonObject &info);
    void healthFailed(const QString &error);

    void chatReady(const QJsonObject &response);
    void chatFailed(const QString &error);

private slots:
    void onHealthFinished();
    void onChatFinished();

private:
    QUrl endpoint(const QString &path) const;

    QUrl m_base;
    QNetworkAccessManager *m_net = nullptr;
    QNetworkReply *m_healthReply = nullptr;
    QNetworkReply *m_chatReply = nullptr;
};

#include "BackendClient.h"

#include <QJsonDocument>
#include <QJsonValue>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>

namespace {

// 把一次失败的回复翻译成「人话」。
//
// 服务端的错误体有两种形状，都是 FastAPI 的默认写法：
//   { "detail": { "kind": "...", "message": "...", "hint": "..." } }   ← 我们自己抛的
//   { "detail": [ { "loc": [...], "msg": "..." } ] }                   ← 参数校验失败
// 直接把原始 JSON 甩给用户，等于让用户去读协议；所以这里统一拆成一句话。
QString describeError(QNetworkReply *reply)
{
    const int code = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
    const QByteArray body = reply->readAll();

    QString message;
    const QJsonObject obj = QJsonDocument::fromJson(body).object();
    if (obj.contains(QStringLiteral("detail"))) {
        const QJsonValue detail = obj.value(QStringLiteral("detail"));
        if (detail.isObject()) {
            message = detail.toObject().value(QStringLiteral("message")).toString();
        } else if (detail.isString()) {
            message = detail.toString();
        }
    }
    if (message.isEmpty() && !body.isEmpty())
        message = QString::fromUtf8(body.left(300));
    if (message.isEmpty())
        message = reply->errorString();

    if (code > 0)
        return QStringLiteral("HTTP %1 —— %2").arg(code).arg(message);
    // 连 TCP 都没连上时 code 为 0，这时 errorString 才是最有用的信息
    // （典型：「Connection refused」= 服务压根没起来）。
    return QStringLiteral("%1(%2)").arg(reply->errorString()).arg(int(reply->error()));
}

} // namespace

BackendClient::BackendClient(QObject *parent)
    : QObject(parent)
    , m_base(QStringLiteral("http://127.0.0.1:8000"))
    , m_net(new QNetworkAccessManager(this))
{
}

void BackendClient::setBaseUrl(const QUrl &url)
{
    if (url.isValid() && !url.scheme().isEmpty())
        m_base = url;
}

QUrl BackendClient::endpoint(const QString &path) const
{
    // 基址可能带不带结尾斜杠，统一去掉再拼，避免出现 //health 这种路径。
    QString base = m_base.toString();
    while (base.endsWith(QLatin1Char('/')))
        base.chop(1);
    return QUrl(base + path);
}

void BackendClient::checkHealth()
{
    // 连点「检查后端」时不排队：上一次没回来就直接作废，以最后一次为准。
    if (m_healthReply) {
        m_healthReply->abort();
        m_healthReply->deleteLater();
        m_healthReply = nullptr;
    }

    QNetworkRequest req(endpoint(QStringLiteral("/health")));
    req.setHeader(QNetworkRequest::UserAgentHeader, QStringLiteral("libs-agent-client/0.1"));

    m_healthReply = m_net->get(req);
    connect(m_healthReply, &QNetworkReply::finished, this, &BackendClient::onHealthFinished);
}

void BackendClient::onHealthFinished()
{
    QNetworkReply *reply = m_healthReply;
    m_healthReply = nullptr;
    if (!reply)
        return;
    reply->deleteLater();   // 推迟释放：下面还要读它的内容

    if (reply->error() != QNetworkReply::NoError) {
        emit healthFailed(describeError(reply));
        return;
    }

    QJsonParseError perr{};
    const QJsonDocument doc =
        QJsonDocument::fromJson(reply->readAll(), &perr);
    if (perr.error != QJsonParseError::NoError || !doc.isObject()) {
        emit healthFailed(QStringLiteral("/health 返回的不是合法 JSON：%1").arg(perr.errorString()));
        return;
    }
    emit healthReady(doc.object());
}

void BackendClient::sendChat(const QString &message, const QString &sessionId)
{
    if (m_chatReply) {
        m_chatReply->abort();
        m_chatReply->deleteLater();
        m_chatReply = nullptr;
    }

    QJsonObject body;
    body.insert(QStringLiteral("message"), message);
    // session_id 是服务端会话的钥匙。会话历史（含 reasoning_content）全在服务端，
    // 客户端只负责把这个 id 原样带回来 —— 第 ① 条设计决定就是为这个。
    if (!sessionId.isEmpty())
        body.insert(QStringLiteral("session_id"), sessionId);

    QNetworkRequest req(endpoint(QStringLiteral("/chat")));
    req.setHeader(QNetworkRequest::ContentTypeHeader, QStringLiteral("application/json"));
    req.setHeader(QNetworkRequest::UserAgentHeader, QStringLiteral("libs-agent-client/0.1"));

    m_chatReply = m_net->post(req, QJsonDocument(body).toJson(QJsonDocument::Compact));
    connect(m_chatReply, &QNetworkReply::finished, this, &BackendClient::onChatFinished);
}

void BackendClient::onChatFinished()
{
    QNetworkReply *reply = m_chatReply;
    m_chatReply = nullptr;
    if (!reply)
        return;
    reply->deleteLater();

    if (reply->error() != QNetworkReply::NoError) {
        emit chatFailed(describeError(reply));
        return;
    }

    QJsonParseError perr{};
    const QJsonDocument doc = QJsonDocument::fromJson(reply->readAll(), &perr);
    if (perr.error != QJsonParseError::NoError || !doc.isObject()) {
        emit chatFailed(QStringLiteral("/chat 返回的不是合法 JSON：%1").arg(perr.errorString()));
        return;
    }
    emit chatReady(doc.object());
}

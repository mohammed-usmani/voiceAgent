import pytest

from voiceagent.outcomes import email as email_module
from voiceagent.outcomes.email import Email, LogMailer, SmtpMailer, default_mailer
from voiceagent.settings import Settings

MAIL = Email(to="priya@example.com", subject="Next steps", body="Hi Priya,\n\nThanks!")


class FakeSMTP:
    """Stands in for smtplib.SMTP / SMTP_SSL: records what the mailer did."""

    instances: list["FakeSMTP"] = []

    def __init__(self, host, port, timeout=None, context=None):
        self.host, self.port, self.timeout = host, port, timeout
        self.calls: list = []
        FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.calls.append("quit")

    def starttls(self, context=None):
        self.calls.append("starttls")

    def login(self, user, password):
        self.calls.append(("login", user, password))

    def send_message(self, msg):
        self.calls.append(("send", msg["From"], msg["To"], msg["Subject"], msg.get_content()))


@pytest.fixture
def smtp(monkeypatch):
    FakeSMTP.instances = []
    monkeypatch.setattr(email_module.smtplib, "SMTP", FakeSMTP)
    monkeypatch.setattr(email_module.smtplib, "SMTP_SSL", FakeSMTP)
    return FakeSMTP.instances


def test_smtp_mailer_uses_starttls_and_logs_in(smtp):
    SmtpMailer("smtp.example.com", 587, sender="hiring@acme.test", username="u", password="p").send(
        MAIL
    )
    (conn,) = smtp
    assert (conn.host, conn.port, conn.timeout) == ("smtp.example.com", 587, 30)
    assert conn.calls == [
        "starttls",
        ("login", "u", "p"),
        ("send", "hiring@acme.test", "priya@example.com", "Next steps", "Hi Priya,\n\nThanks!\n"),
        "quit",
    ]


def test_port_465_is_implicit_tls_and_no_login_without_a_username(smtp):
    SmtpMailer("smtp.example.com", 465, sender="hiring@acme.test").send(MAIL)
    assert [c if isinstance(c, str) else c[0] for c in smtp[0].calls] == ["send", "quit"]


def test_log_mailer_only_logs(monkeypatch):
    logged = []
    monkeypatch.setattr(email_module.logger, "info", lambda msg, *args: logged.append(msg % args))
    LogMailer().send(MAIL)
    assert logged == ["email not sent (no SMTP_HOST) to priya@example.com: Next steps"]


def settings(**kw) -> Settings:
    return Settings(database_url="", api_key="", together_api_key="", **kw)


def test_without_smtp_host_mail_is_only_logged():
    assert isinstance(default_mailer(settings()), LogMailer)


def test_smtp_host_needs_a_sender():
    with pytest.raises(RuntimeError, match="SMTP_FROM"):
        default_mailer(settings(smtp_host="smtp.example.com"))


def test_smtp_settings_build_an_smtp_mailer():
    m = default_mailer(
        settings(smtp_host="smtp.example.com", smtp_from="hiring@acme.test", smtp_port=2525)
    )
    assert isinstance(m, SmtpMailer) and (m.host, m.port, m.starttls) == (
        "smtp.example.com",
        2525,
        True,
    )

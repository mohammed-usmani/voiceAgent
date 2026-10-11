import smtplib

import pytest

from voiceagent.outcomes import email as email_module
from voiceagent.outcomes.email import Email, LogMailer, MaybeSent, SmtpMailer, default_mailer
from voiceagent.settings import Settings

MAIL = Email(to="priya@example.com", subject="Next steps", body="Hi Priya,\n\nThanks!")


class FakeSMTP:
    """Stands in for smtplib.SMTP / SMTP_SSL: records what the mailer did."""

    instances: list["FakeSMTP"] = []
    send_error: Exception | None = None
    quit_error: Exception | None = None

    def __init__(self, host, port, timeout=None, context=None):
        self.host, self.port, self.timeout = host, port, timeout
        self.calls: list = []
        FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.quit()

    def quit(self):
        self.calls.append("quit")
        if self.quit_error:
            raise self.quit_error

    def close(self):
        self.calls.append("close")

    def starttls(self, context=None):
        self.calls.append("starttls")

    def login(self, user, password):
        self.calls.append(("login", user, password))

    def send_message(self, msg):
        if self.send_error:
            raise self.send_error
        self.calls.append(("send", msg["From"], msg["To"], msg["Subject"], msg.get_content()))


@pytest.fixture
def smtp(monkeypatch):
    FakeSMTP.instances = []
    FakeSMTP.send_error = FakeSMTP.quit_error = None
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


def test_a_failed_quit_after_the_server_accepted_the_mail_is_still_sent(smtp):
    FakeSMTP.quit_error = smtplib.SMTPResponseException(451, b"bye")
    SmtpMailer("smtp.example.com", 587, sender="hiring@acme.test").send(MAIL)  # no raise
    assert smtp[0].calls[-2:] == ["quit", "close"]


@pytest.mark.parametrize("error", [TimeoutError("timed out"), smtplib.SMTPServerDisconnected()])
def test_a_connection_lost_mid_send_may_have_delivered(smtp, error):
    FakeSMTP.send_error = error
    with pytest.raises(MaybeSent, match="check it wasn't received"):
        SmtpMailer("smtp.example.com", 587, sender="hiring@acme.test").send(MAIL)


def test_a_refused_recipient_is_a_plain_failure(smtp):
    FakeSMTP.send_error = smtplib.SMTPRecipientsRefused({"priya@example.com": (550, b"no")})
    with pytest.raises(smtplib.SMTPRecipientsRefused):
        SmtpMailer("smtp.example.com", 587, sender="hiring@acme.test").send(MAIL)


def test_log_mailer_only_logs_and_never_the_address(monkeypatch):
    logged = []
    monkeypatch.setattr(email_module.logger, "info", lambda msg, *args: logged.append(msg % args))
    LogMailer().send(MAIL)
    assert logged == ["email not sent (no SMTP_HOST): Next steps"]
    assert LogMailer.delivers is False and SmtpMailer.delivers is True


def settings(**kw) -> Settings:
    return Settings(database_url="", api_key="", together_api_key="", **kw)


def test_without_smtp_host_mail_is_only_logged_with_a_warning(monkeypatch):
    warned = []
    monkeypatch.setattr(email_module.logger, "warning", lambda msg, *args: warned.append(msg))
    assert isinstance(default_mailer(settings()), LogMailer)
    assert warned == ["SMTP_HOST is not set: outcome emails are only logged, never sent"]


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


def test_an_email_can_show_the_hiring_companys_name(smtp):
    # One sending address for every job, but the candidate sees who is hiring.
    named = Email(
        to="priya@example.com", subject="Next steps", body="Hi", sender_name="Acme Hiring"
    )
    SmtpMailer("smtp.example.com", 587, sender="hiring@acme.test").send(named)
    assert next(c for c in smtp[0].calls if c[0] == "send")[1] == "Acme Hiring <hiring@acme.test>"
    SmtpMailer("smtp.example.com", 587, sender="Team <hiring@acme.test>").send(named)
    assert next(c for c in smtp[1].calls if c[0] == "send")[1] == "Acme Hiring <hiring@acme.test>"

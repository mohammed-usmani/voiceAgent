"""Outcome email. SMTP when SMTP_HOST is set; otherwise (development) nothing leaves the
machine — the message stays in the messages table and the log."""

import logging
import smtplib
import ssl
from dataclasses import dataclass
from email.message import EmailMessage
from typing import Protocol

from voiceagent.settings import Settings

logger = logging.getLogger(__name__)
SMTP_TIMEOUT_S = 30


@dataclass(frozen=True)
class Email:
    to: str
    subject: str
    body: str


class Mailer(Protocol):
    def send(self, email: Email) -> None:
        """Deliver or raise."""


class SmtpMailer:
    def __init__(
        self,
        host: str,
        port: int,
        *,
        sender: str,
        username: str = "",
        password: str = "",
        starttls: bool = True,
    ) -> None:
        self.host, self.port, self.sender = host, port, sender
        self.username, self.password, self.starttls = username, password, starttls

    def send(self, email: Email) -> None:
        msg = EmailMessage()
        msg["From"] = self.sender
        msg["To"] = email.to
        msg["Subject"] = email.subject
        msg.set_content(email.body)
        context = ssl.create_default_context()
        if self.port == 465:  # implicit TLS
            smtp = smtplib.SMTP_SSL(self.host, self.port, timeout=SMTP_TIMEOUT_S, context=context)
        else:
            smtp = smtplib.SMTP(self.host, self.port, timeout=SMTP_TIMEOUT_S)
        with smtp:
            if self.starttls and self.port != 465:
                smtp.starttls(context=context)
            if self.username:
                smtp.login(self.username, self.password)
            smtp.send_message(msg)


class LogMailer:
    """Development mail: logged, never sent."""

    def send(self, email: Email) -> None:
        logger.info("email not sent (no SMTP_HOST) to %s: %s", email.to, email.subject)


def default_mailer(settings: Settings) -> Mailer:
    if not settings.smtp_host:
        return LogMailer()
    if not settings.smtp_from:
        raise RuntimeError("SMTP_FROM must be set when SMTP_HOST is")
    return SmtpMailer(
        settings.smtp_host,
        settings.smtp_port,
        sender=settings.smtp_from,
        username=settings.smtp_username,
        password=settings.smtp_password,
        starttls=settings.smtp_starttls,
    )

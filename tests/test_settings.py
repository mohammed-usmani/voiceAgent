from voiceagent.settings import get_settings


def test_smtp_settings_come_from_the_environment(monkeypatch):
    monkeypatch.setattr("voiceagent.settings.load_dotenv", lambda: None)
    for name, value in {
        "SMTP_HOST": "smtp.example.com",
        "SMTP_PORT": "465",
        "SMTP_USERNAME": "u",
        "SMTP_PASSWORD": "p",
        "SMTP_FROM": "Acme Hiring <hiring@example.com>",
        "SMTP_STARTTLS": "false",
    }.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    try:
        s = get_settings()
        assert (s.smtp_host, s.smtp_port, s.smtp_username, s.smtp_password) == (
            "smtp.example.com",
            465,
            "u",
            "p",
        )
        assert (s.smtp_from, s.smtp_starttls) == ("Acme Hiring <hiring@example.com>", False)
        assert s.reviewer_model == "deepseek-ai/DeepSeek-V4-Pro-0813"
    finally:
        get_settings.cache_clear()


def test_without_smtp_host_mail_is_only_recorded(monkeypatch):
    monkeypatch.setattr("voiceagent.settings.load_dotenv", lambda: None)
    monkeypatch.delenv("SMTP_HOST", raising=False)
    monkeypatch.delenv("SMTP_PORT", raising=False)
    monkeypatch.delenv("SMTP_STARTTLS", raising=False)
    get_settings.cache_clear()
    try:
        s = get_settings()
        assert (s.smtp_host, s.smtp_port, s.smtp_starttls) == ("", 587, True)
    finally:
        get_settings.cache_clear()

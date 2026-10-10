"""screening-demo-seed: a demo workspace that never dials or sends."""

from sqlalchemy import func, select

from tests.seed import seed_job
from voiceagent import demo
from voiceagent.db.base import make_sessionmaker
from voiceagent.db.models import Call, Candidate, Decision, Job, Message, Task


def counts(session, column) -> dict:
    return dict(session.execute(select(column, func.count()).group_by(column)).all())


def run(engine, *argv: str) -> int:
    return demo.main(list(argv), sessionmaker=make_sessionmaker(engine))


def test_seeds_a_demo_workspace(engine, session):
    assert run(engine) == 0
    jobs = session.scalars(select(Job)).all()
    assert sorted(j.company_name for j in jobs) == [
        "Brightline Support Services (demo)",
        "Kestrel Pay (demo)",
    ]
    assert all(j.status == "active" and not j.auto_send_invites for j in jobs)
    assert counts(session, Candidate.status) == {
        "decided": 4,
        "notified": 1,
        "pending": 2,
        "callback": 1,
        "unreachable": 1,
        "declined_consent": 1,
        "withdrawn": 1,
        "no_answer": 1,
    }
    results = counts(session, Decision.result)
    assert results == {"suitable": 2, "not_suitable": 2, "needs_review": 1}
    assert counts(session, Message.status) == {"draft": 3, "sent": 1}
    finished = session.scalars(select(Call).where(Call.status == "completed")).all()
    assert len(finished) == 8 and all(c.transcript for c in finished)
    excluded = session.scalars(select(Candidate).where(Candidate.excluded_at.is_not(None)))
    assert [c.name for c in excluded] == ["Meghna Rao"]


def test_decisions_follow_the_real_rules(engine, session):
    run(engine)
    by_name = {
        name: d
        for d, name in session.execute(select(Decision, Candidate.name).join(Candidate)).all()
    }
    assert (by_name["Rahul Verma"].result, by_name["Rahul Verma"].hard_failures) == (
        "not_suitable",
        ["notice_days"],
    )
    assert by_name["Sana Sheikh"].result == "needs_review"  # night shifts: no clear answer
    vikram = by_name["Vikram Singh"]
    assert (vikram.result, vikram.final_result, vikram.status) == (
        "not_suitable",
        "suitable",
        "overridden",
    )
    assert all(r["quote"] for d in by_name.values() for r in d.reasons)


def test_never_dials_or_sends(engine, session, mailer):
    # Review focus 5: nothing queued, nothing emailed, no reachable address.
    run(engine)
    assert session.scalars(select(Task)).all() == []
    assert mailer.sent == []
    candidates = session.scalars(select(Candidate)).all()
    assert all(c.phone is None and c.sip_address.endswith("@demo.invalid") for c in candidates)
    assert all(c.email.endswith("@example.com") for c in candidates)


def test_refuses_a_database_with_real_jobs(engine, session, capsys):
    # Review focus 5: a real workspace stays untouched without --force.
    seed_job(session)
    assert run(engine) == 2
    assert "--force" in capsys.readouterr().err
    assert [j.company_name for j in session.scalars(select(Job))] == ["Acme"]


def test_force_adds_the_demo_beside_real_jobs_and_a_rerun_replaces_it(engine, session):
    seed_job(session)
    assert run(engine, "--force") == 0
    assert run(engine, "--force") == 0
    session.expire_all()
    assert sorted(j.company_name for j in session.scalars(select(Job))) == [
        "Acme",
        "Brightline Support Services (demo)",
        "Kestrel Pay (demo)",
    ]
    assert session.scalar(select(func.count()).select_from(Candidate)) == 12

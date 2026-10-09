from pathlib import Path

import pytest

from voiceagent.evals.jobs import load_job, load_jobs

SLUGS = [
    "customer-support",
    "data-analyst",
    "sales-executive",
    "software-engineer",
    "staff-nurse",
    "warehouse-supervisor",
]


def test_the_six_reference_jobs_load_and_validate():
    jobs = load_jobs()
    assert [j.slug for j in jobs] == SLUGS
    for job in jobs:
        reqs = job.criteria.requirements
        assert 150 <= len(job.description.split()) <= 250, job.slug
        assert any(r.hard for r in reqs), job.slug
        assert any(not r.hard and r.weight > 0 for r in reqs), job.slug
        assert sum(r.weight for r in reqs if not r.hard) == pytest.approx(1, abs=1e-3)
        assert job.overrides, job.slug
        assert {o.requirement_id for o in job.overrides} <= {r.id for r in reqs if not r.hard}


def test_some_jobs_carry_an_illegal_line_to_drop():
    assert sum(bool(j.illegal) for j in load_jobs()) >= 1


def test_a_range_of_years_is_labelled_as_its_lower_bound():
    sales = load_jobs(["sales-executive"])[0]
    rule = {r.id: r.rule for r in sales.criteria.requirements}["sales_years"]
    assert (rule.op, rule.value) == (">=", 1)


def test_an_unknown_slug_names_the_known_ones():
    with pytest.raises(ValueError, match="unknown jobs: nope .*software-engineer"):
        load_jobs(["nope"])


def test_a_label_that_quotes_words_not_in_the_description_is_rejected(tmp_path: Path):
    text = Path("evals/jobs/data-analyst.yaml").read_text()
    bad = tmp_path / "bad.yaml"
    source = "source: At least 2 years of hands-on SQL experience in an analyst role"
    assert source in text
    bad.write_text(text.replace(source, "source: At least 7 years of SQL in any role"))
    with pytest.raises(ValueError, match="sql_years: source not in the description"):
        load_job(bad)


def test_an_override_for_a_requirement_the_job_lacks_is_rejected(tmp_path: Path):
    text = Path("evals/jobs/data-analyst.yaml").read_text()
    bad = tmp_path / "bad.yaml"
    bad.write_text(text.replace("- requirement_id: ab_testing", "- requirement_id: nope"))
    with pytest.raises(ValueError, match="override for unknown requirement 'nope'"):
        load_job(bad)

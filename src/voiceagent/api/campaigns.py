"""Campaigns: call a job's candidates in the background (worker.campaign.tick does the
calling)."""

import uuid
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from voiceagent.api.deps import get_session
from voiceagent.api.jobs import get_job
from voiceagent.api.schemas import CampaignOut, CampaignStart
from voiceagent.db.models import ACTIVE_CALL_STATUSES, Call, Campaign, Candidate, CriteriaVersion
from voiceagent.events import emit

router = APIRouter()
DB = Annotated[Session, Depends(get_session)]


def campaign_out(session: Session, campaign: Campaign) -> CampaignOut:
    counts = session.execute(
        select(Candidate.status, func.count())
        .where(Candidate.job_id == campaign.job_id)
        .group_by(Candidate.status)
    )
    in_flight = session.scalar(
        select(func.count())
        .select_from(Call)
        .join(Candidate)
        .where(Candidate.job_id == campaign.job_id, Call.status.in_(ACTIVE_CALL_STATUSES))
    )
    return CampaignOut(
        job_id=campaign.job_id,
        status=campaign.status,
        max_concurrent=campaign.max_concurrent,
        started_at=campaign.started_at,
        paused_at=campaign.paused_at,
        finished_at=campaign.finished_at,
        in_flight=in_flight,
        candidates=dict(counts.all()),
    )


def get_campaign(session: Session, job_id: uuid.UUID, *, lock: bool = False) -> Campaign:
    campaign = session.get(Campaign, job_id, with_for_update=lock or None, populate_existing=lock)
    if campaign is None:
        raise HTTPException(status_code=404, detail="this job has no campaign")
    return campaign


@router.post("/jobs/{job_id}/campaign", response_model=CampaignOut)
def start_campaign(
    job_id: uuid.UUID, session: DB, body: CampaignStart | None = None
) -> CampaignOut:
    """Start calling the job's candidates within its calling hours, or resume a paused or
    finished campaign (e.g. after adding candidates). Calls up to `max_concurrent` at once;
    unanswered calls are retried up to 3 times, 2 hours apart; callbacks are made at the
    time the candidate gave (09:00-21:00 local)."""
    job = get_job(session, job_id, lock=True)  # lock order: job, then campaign
    approved = session.scalar(
        select(CriteriaVersion.id).where(
            CriteriaVersion.job_id == job.id, CriteriaVersion.status == "approved"
        )
    )
    if job.status != "active" or approved is None:
        raise HTTPException(
            status_code=409, detail=f"job is {job.status}: approve its criteria before calling"
        )
    now = datetime.now(UTC)
    campaign = session.get(Campaign, job.id, with_for_update=True, populate_existing=True)
    before = campaign and (campaign.status, campaign.max_concurrent)
    if campaign is None:
        campaign = Campaign(job_id=job.id, started_at=now)
        session.add(campaign)
        session.flush()  # fills in the defaults
    if body is not None and body.max_concurrent is not None:
        campaign.max_concurrent = body.max_concurrent
    if campaign.status != "active":
        campaign.started_at = now
    campaign.status = "active"
    campaign.paused_at = campaign.finished_at = None
    if (campaign.status, campaign.max_concurrent) != before:  # not a repeated start
        emit(
            session, "job", job.id, "campaign_started", {"max_concurrent": campaign.max_concurrent}
        )
    session.commit()
    return campaign_out(session, campaign)


@router.post("/jobs/{job_id}/campaign:pause", response_model=CampaignOut)
def pause_campaign(job_id: uuid.UUID, session: DB) -> CampaignOut:
    """Stop queueing new calls; calls already queued or in progress finish."""
    get_job(session, job_id, lock=True)
    campaign = get_campaign(session, job_id, lock=True)
    if campaign.status == "done":
        raise HTTPException(status_code=409, detail="the campaign is done; start it again instead")
    if campaign.status == "active":
        campaign.status = "paused"
        campaign.paused_at = datetime.now(UTC)
        emit(session, "job", job_id, "campaign_paused")
    session.commit()
    return campaign_out(session, campaign)


@router.get("/jobs/{job_id}/campaign", response_model=CampaignOut)
def read_campaign(job_id: uuid.UUID, session: DB) -> CampaignOut:
    get_job(session, job_id)
    return campaign_out(session, get_campaign(session, job_id))

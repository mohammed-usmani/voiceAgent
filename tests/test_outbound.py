from types import SimpleNamespace

import pytest

from voiceagent import outbound
from voiceagent.outbound import AGENT_NAME, CALLER_NAME, dial_request, parse_address


@pytest.mark.parametrize(
    ("address", "expected"),
    [
        ("candidate@sip.linphone.org", ("candidate", "sip.linphone.org")),
        ("sip:candidate@example.org", ("candidate", "example.org")),
        ("candidate", ("candidate", "sip.linphone.org")),
    ],
)
def test_parse_address(address, expected):
    assert parse_address(address) == expected


def test_parse_address_needs_a_user():
    with pytest.raises(ValueError):
        parse_address("@sip.linphone.org")


def test_dial_shows_the_company_and_waits_for_pickup():
    req = dial_request("ST_x", "candidate", "call-out-1")
    assert req.sip_call_to == "candidate"
    assert req.display_name == CALLER_NAME
    assert req.wait_until_answered


class FakeLiveKit:
    """Records the requests outbound.dial makes; the linphone trunk already exists."""

    def __init__(self):
        self.requests = []
        self.closed = False
        self.sip = SimpleNamespace(
            list_outbound_trunk=self._trunks,
            create_sip_participant=self._record,
        )
        self.agent_dispatch = SimpleNamespace(create_dispatch=self._record)

    async def _trunks(self, _req):
        trunk = SimpleNamespace(name="outbound-sip.linphone.org", sip_trunk_id="ST_linphone")
        return SimpleNamespace(items=[trunk])

    async def _record(self, req):
        self.requests.append(req)

    async def aclose(self):
        self.closed = True


@pytest.fixture
def livekit(monkeypatch):
    lk = FakeLiveKit()
    monkeypatch.setattr(outbound.api, "LiveKitAPI", lambda: lk)
    return lk


async def test_dial_sip_address_dispatches_the_agent_with_metadata(livekit):
    await outbound.dial(
        "priya@sip.linphone.org",
        "screen-1",
        display_name="Acme Hiring",
        metadata='{"call_id": "x"}',
    )
    participant, dispatch = livekit.requests
    assert (participant.sip_trunk_id, participant.sip_call_to, participant.room_name) == (
        "ST_linphone",
        "priya",
        "screen-1",
    )
    assert participant.display_name == "Acme Hiring" and participant.wait_until_answered
    assert (dispatch.agent_name, dispatch.room, dispatch.metadata) == (
        AGENT_NAME,
        "screen-1",
        '{"call_id": "x"}',
    )
    assert livekit.closed


async def test_dial_phone_number_uses_the_configured_trunk(livekit, monkeypatch):
    monkeypatch.setenv("SIP_OUTBOUND_TRUNK_ID", "ST_pstn")
    await outbound.dial("+919876543210", "screen-2")
    participant = livekit.requests[0]
    assert (participant.sip_trunk_id, participant.sip_call_to) == ("ST_pstn", "+919876543210")


async def test_dial_phone_number_without_a_trunk_fails_before_ringing(livekit, monkeypatch):
    monkeypatch.delenv("SIP_OUTBOUND_TRUNK_ID", raising=False)
    with pytest.raises(RuntimeError, match="SIP_OUTBOUND_TRUNK_ID"):
        await outbound.dial("+919876543210", "screen-3")
    assert livekit.requests == [] and livekit.closed

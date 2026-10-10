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
        self.options = None
        self.dispatch_error = None
        self.ring_error = None
        self.deleted_rooms = []
        self.sip = SimpleNamespace(
            list_outbound_trunk=self._trunks,
            create_sip_participant=self._record,
        )
        self.agent_dispatch = SimpleNamespace(create_dispatch=self._dispatch)
        self.room = SimpleNamespace(delete_room=self._delete_room)

    async def _dispatch(self, req):
        if self.dispatch_error:
            raise self.dispatch_error
        self.requests.append(req)

    async def _delete_room(self, req):
        self.deleted_rooms.append(req.room)

    async def _trunks(self, _req):
        trunk = SimpleNamespace(name="outbound-sip.linphone.org", sip_trunk_id="ST_linphone")
        return SimpleNamespace(items=[trunk])

    async def _record(self, req):
        if self.ring_error:
            raise self.ring_error
        self.requests.append(req)

    async def aclose(self):
        self.closed = True


@pytest.fixture
def livekit(monkeypatch):
    lk = FakeLiveKit()

    def make(**options):
        lk.options = options
        return lk

    monkeypatch.setattr(outbound.api, "LiveKitAPI", make)
    return lk


async def test_dial_sip_address_dispatches_the_agent_with_metadata(livekit):
    await outbound.dial(
        "priya@sip.linphone.org",
        "screen-1",
        display_name="Acme Hiring",
        metadata='{"call_id": "x"}',
    )
    # A screening call: the agent is in the room before the phone rings, so it hears the
    # first "Hello?" (it says nothing until its answering-machine check hears the pickup).
    dispatch, participant = livekit.requests
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


async def test_dial_is_never_replayed_to_another_region(livekit):
    # I5: the SDK's failover would resend CreateSIPParticipant and ring the candidate twice.
    await outbound.dial("priya@sip.linphone.org", "screen-4")
    assert livekit.options == {"failover": False}


async def test_a_failed_dispatch_hangs_up_the_answered_call(livekit):
    # M4: the candidate picked up but no agent is coming: don't leave them on a silent line.
    livekit.dispatch_error = RuntimeError("dispatch refused")
    with pytest.raises(RuntimeError, match="dispatch refused"):
        await outbound.dial("priya@sip.linphone.org", "screen-5")
    assert livekit.deleted_rooms == ["screen-5"] and livekit.closed


async def test_a_sip_address_with_a_phone_style_user_goes_to_its_host(livekit, monkeypatch):
    # M5: "+15551234@sip.example.com" is a SIP address, not a phone number.
    monkeypatch.setenv("SIP_OUTBOUND_TRUNK_ID", "ST_pstn")
    await outbound.dial("+15551234@sip.linphone.org", "screen-6")
    participant = livekit.requests[0]
    assert (participant.sip_trunk_id, participant.sip_call_to) == ("ST_linphone", "+15551234")


async def test_an_unanswered_screening_call_closes_the_room_the_agent_waits_in(livekit):
    livekit.ring_error = RuntimeError("486 busy")
    with pytest.raises(RuntimeError, match="busy"):
        await outbound.dial("priya@sip.linphone.org", "screen-7", metadata='{"call_id": "x"}')
    [dispatch] = livekit.requests
    assert dispatch.room == "screen-7"
    assert livekit.deleted_rooms == ["screen-7"] and livekit.closed


async def test_the_demo_agent_is_sent_in_only_after_pickup(livekit):
    # It greets as soon as it joins, so it must not join while the phone rings.
    await outbound.dial("priya@sip.linphone.org", "call-out-1")
    participant, dispatch = livekit.requests
    assert participant.room_name == dispatch.room == "call-out-1"

"""Have the agent call someone: ring a SIP address, then send the agent in once answered.

    uv run voiceagent-call                          # calls CALL_TO from .env
    uv run voiceagent-call candidate@sip.linphone.org
    uv run voiceagent-call +919876543210            # through SIP_OUTBOUND_TRUNK_ID

A free linphone.org account works as the callee, so no phone number has to be bought.
The outbound trunk for the callee's SIP host is created on first use.
"""

import asyncio
import contextlib
import os
import secrets
import sys

from dotenv import load_dotenv
from livekit import api

AGENT_NAME = "screening-agent"
CALLER_NAME = "Nimbus Labs Hiring"
DEFAULT_HOST = "sip.linphone.org"
# LiveKit needs a caller number on the trunk; only CALLER_NAME is shown to the callee.
CALLER_NUMBER = "+11111111111"


def parse_address(address: str) -> tuple[str, str]:
    """'user@host', 'sip:user@host' or plain 'user' (on DEFAULT_HOST) -> (user, host)."""
    address = address.removeprefix("sip:")
    user, _, host = address.partition("@")
    if not user:
        raise ValueError(f"no user in SIP address {address!r}")
    return user, host or DEFAULT_HOST


def dial_request(
    trunk_id: str, to: str, room: str, display_name: str = CALLER_NAME
) -> api.CreateSIPParticipantRequest:
    return api.CreateSIPParticipantRequest(
        sip_trunk_id=trunk_id,
        sip_call_to=to,
        room_name=room,
        participant_identity="candidate",
        participant_name="Candidate",
        display_name=display_name,
        # Returns once they answer, and raises if they don't.
        wait_until_answered=True,
    )


async def trunk_for(lk: api.LiveKitAPI, host: str) -> str:
    name = f"outbound-{host}"
    trunks = await lk.sip.list_outbound_trunk(api.ListSIPOutboundTrunkRequest())
    for trunk in trunks.items:
        if trunk.name == name:
            return trunk.sip_trunk_id
    trunk = await lk.sip.create_outbound_trunk(
        api.CreateSIPOutboundTrunkRequest(
            trunk=api.SIPOutboundTrunkInfo(
                name=name,
                address=host,
                numbers=[CALLER_NUMBER],
                # sip.linphone.org didn't answer over UDP; it did over TCP.
                transport=api.SIPTransport.SIP_TRANSPORT_TCP,
            )
        )
    )
    return trunk.sip_trunk_id


async def dial(
    address: str, room: str, *, display_name: str = CALLER_NAME, metadata: str = ""
) -> None:
    """Ring `address` and have the agent take the call in `room`, with `metadata`.

    `address` is a phone number (+E.164, through the trunk in SIP_OUTBOUND_TRUNK_ID) or a
    SIP address, anything with an "@" (through a per-host trunk created on first use).
    Raises api.SipCallError when the call isn't answered (busy, no answer, declined: see
    `sip_status_code`).
    """
    # failover=False: the SDK would otherwise replay a timed-out or failed request to
    # another region (livekit/api/livekit_api.py:35, _failover.py:32), and a replayed
    # CreateSIPParticipant rings the candidate a second time.
    lk = api.LiveKitAPI(failover=False)
    try:
        if "@" not in address:
            trunk_id = os.environ.get("SIP_OUTBOUND_TRUNK_ID", "")
            if not trunk_id:
                raise RuntimeError("set SIP_OUTBOUND_TRUNK_ID to call phone numbers")
            to = address
        else:
            to, host = parse_address(address)
            trunk_id = await trunk_for(lk, host)
        ring = dial_request(trunk_id, to, room, display_name)
        dispatch = api.CreateAgentDispatchRequest(
            agent_name=AGENT_NAME, room=room, metadata=metadata
        )
        if metadata:
            # A screening call: the agent joins while the phone rings, so it is listening the
            # moment they answer; its answering-machine detection waits for the pickup before
            # anything is said. Sent in after pickup, it took ~2.5 s to join.
            await lk.agent_dispatch.create_dispatch(dispatch)
            try:
                await lk.sip.create_sip_participant(ring)
            except Exception:
                # Nobody answered: close the room so the waiting agent leaves.
                with contextlib.suppress(Exception):
                    await lk.room.delete_room(api.DeleteRoomRequest(room=room))
                raise
            return
        # The demo agent greets as soon as it joins: send it in only after pickup.
        await lk.sip.create_sip_participant(ring)
        try:
            await lk.agent_dispatch.create_dispatch(dispatch)
        except Exception:
            # They answered but no agent is coming: hang up rather than leave them on a
            # silent line. The dispatch error is the one worth raising.
            with contextlib.suppress(Exception):
                await lk.room.delete_room(api.DeleteRoomRequest(room=room))
            raise
    finally:
        await lk.aclose()


async def call(address: str) -> None:
    room = f"call-out-{secrets.token_hex(4)}"
    print(f"ringing {address} ...", flush=True)
    try:
        await dial(address, room)
        print(f"answered; agent joining room {room}")
    except api.TwirpError as e:
        sys.exit(f"call failed: {e.code} {e.message}")


def main() -> None:
    load_dotenv()
    address = sys.argv[1] if len(sys.argv) > 1 else os.getenv("CALL_TO")
    if not address:
        sys.exit("usage: voiceagent-call <user@sip-host>, or set CALL_TO in .env")
    asyncio.run(call(address))

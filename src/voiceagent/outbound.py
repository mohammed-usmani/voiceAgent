"""Have the agent call someone: ring a SIP address, then send the agent in once answered.

    uv run voiceagent-call                          # calls CALL_TO from .env
    uv run voiceagent-call candidate@sip.linphone.org

A free linphone.org account works as the callee, so no phone number has to be bought.
The outbound trunk for the callee's SIP host is created on first use.
"""

import asyncio
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
        # Dispatch the agent only after pickup, so its greeting isn't spoken to a ringtone.
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
    """Ring `address` and, once answered, send the agent into `room` with `metadata`.

    `address` is a phone number (+E.164, through the trunk in SIP_OUTBOUND_TRUNK_ID) or a
    SIP address (through a per-host trunk created on first use). Raises api.SipCallError
    when the call isn't answered (busy, no answer, declined: see `sip_status_code`).
    """
    lk = api.LiveKitAPI()
    try:
        if address.startswith("+"):
            trunk_id = os.environ.get("SIP_OUTBOUND_TRUNK_ID", "")
            if not trunk_id:
                raise RuntimeError("set SIP_OUTBOUND_TRUNK_ID to call phone numbers")
            to = address
        else:
            to, host = parse_address(address)
            trunk_id = await trunk_for(lk, host)
        await lk.sip.create_sip_participant(dial_request(trunk_id, to, room, display_name))
        await lk.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(agent_name=AGENT_NAME, room=room, metadata=metadata)
        )
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

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

AGENT_NAME = "clinic-agent"
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


def dial_request(trunk_id: str, user: str, room: str) -> api.CreateSIPParticipantRequest:
    return api.CreateSIPParticipantRequest(
        sip_trunk_id=trunk_id,
        sip_call_to=user,
        room_name=room,
        participant_identity="candidate",
        participant_name="Candidate",
        display_name=CALLER_NAME,
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


async def call(address: str) -> None:
    user, host = parse_address(address)
    room = f"call-out-{secrets.token_hex(4)}"
    lk = api.LiveKitAPI()
    try:
        trunk_id = await trunk_for(lk, host)
        print(f"ringing sip:{user}@{host} ...", flush=True)
        await lk.sip.create_sip_participant(dial_request(trunk_id, user, room))
        await lk.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(agent_name=AGENT_NAME, room=room)
        )
        print(f"answered; agent joining room {room}")
    except api.TwirpError as e:
        sys.exit(f"call failed: {e.code} {e.message}")
    finally:
        await lk.aclose()


def main() -> None:
    load_dotenv()
    address = sys.argv[1] if len(sys.argv) > 1 else os.getenv("CALL_TO")
    if not address:
        sys.exit("usage: voiceagent-call <user@sip-host>, or set CALL_TO in .env")
    asyncio.run(call(address))

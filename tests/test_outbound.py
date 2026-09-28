import pytest

from voiceagent.outbound import CALLER_NAME, dial_request, parse_address


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

import pytest
from pydantic import ValidationError

from voiceagent.api.schemas import CandidateCreate

BASE = {"name": "Priya Sharma", "email": "priya@example.com", "phone": "+919876543210"}


def test_contact_details_are_normalised():
    c = CandidateCreate(**BASE | {"email": " Priya@Example.COM ", "phone": "+91 98765-43210"})
    assert (c.email, c.phone, c.sip_address, c.resume_text) == (
        "priya@example.com",
        "+919876543210",
        None,
        "",
    )


def test_sip_address_alone_is_enough_and_blank_cells_are_none():
    c = CandidateCreate(**BASE | {"phone": "", "sip_address": " priya@sip.linphone.org "})
    assert (c.phone, c.sip_address) == (None, "priya@sip.linphone.org")


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"phone": "9876543210"}, "international format"),
        ({"phone": "+0123"}, "international format"),
        ({"phone": None}, "phone number or a SIP address"),
        ({"phone": None, "sip_address": "no-at-sign"}, "SIP address"),
        ({"email": "not-an-email"}, "email"),
        ({"name": "  "}, "name"),
    ],
)
def test_invalid_candidates_are_rejected(patch, message):
    with pytest.raises(ValidationError, match=message):
        CandidateCreate(**BASE | patch)

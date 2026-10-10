"""candidate excluded_at

Revision ID: 5c2e8a7d1f43
Revises: 7a3c5e1f9b20
Create Date: 2026-10-10 12:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "5c2e8a7d1f43"
down_revision: str | Sequence[str] | None = "7a3c5e1f9b20"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # IF NOT EXISTS: safe on a database where the column was already added by hand.
    op.execute(
        "ALTER TABLE candidates ADD COLUMN IF NOT EXISTS excluded_at TIMESTAMP WITH TIME ZONE"
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("ALTER TABLE candidates DROP COLUMN IF EXISTS excluded_at")

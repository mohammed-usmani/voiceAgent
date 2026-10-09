"""campaigns and one active call per candidate

Revision ID: 7a3c5e1f9b20
Revises: eb6e3091a84f
Create Date: 2026-10-09 23:20:41.555568

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "7a3c5e1f9b20"
down_revision: str | Sequence[str] | None = "eb6e3091a84f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

ACTIVE = sa.text("status IN ('queued', 'ringing', 'in_progress')")


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "campaigns",
        sa.Column("job_id", sa.UUID(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("max_concurrent", sa.Integer(), server_default="2", nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("paused_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("job_id"),
    )
    # Fails if a candidate already has two active calls: close the extra one first
    # (it is a bug the API's checks were meant to prevent).
    op.create_index(
        "uq_calls_one_active_per_candidate",
        "calls",
        ["candidate_id"],
        unique=True,
        postgresql_where=ACTIVE,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("uq_calls_one_active_per_candidate", table_name="calls", postgresql_where=ACTIVE)
    op.drop_table("campaigns")

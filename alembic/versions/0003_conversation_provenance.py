from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_conversation_provenance"
down_revision: str | None = "0002_conversations"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("conversations", sa.Column("title", sa.String(length=200), nullable=True))
    op.add_column(
        "conversations",
        sa.Column("updated_at", sa.DateTime(), nullable=True),
    )
    op.execute("UPDATE conversations SET updated_at = created_at WHERE updated_at IS NULL")
    op.alter_column(
        "conversations",
        "updated_at",
        existing_type=sa.DateTime(),
        nullable=False,
    )
    op.create_index(
        op.f("ix_conversations_updated_at"),
        "conversations",
        ["updated_at"],
        unique=False,
    )

    op.add_column("messages", sa.Column("sources", sa.JSON(), nullable=True))
    op.add_column("messages", sa.Column("metrics", sa.JSON(), nullable=True))
    op.add_column(
        "messages",
        sa.Column("status", sa.String(), server_default="complete", nullable=False),
    )


def downgrade() -> None:
    op.drop_column("messages", "status")
    op.drop_column("messages", "metrics")
    op.drop_column("messages", "sources")

    op.drop_index(op.f("ix_conversations_updated_at"), table_name="conversations")
    op.drop_column("conversations", "updated_at")
    op.drop_column("conversations", "title")

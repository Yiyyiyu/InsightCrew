"""add pause_requested to tasks

闸门收敛配套（用户 m18246/m18259）：默认只有 plan / research / review 三道
闸门；`pause_requested=1` 让用户在任意阶段边界临时停一下（接口留宽，默认
路径收窄）。见 `src/core/gates.py::pause_pipeline` / `advance_stage`。

Revision ID: a1c7d9e4f2b8
Revises: 64f167b74622
Create Date: 2026-10-11 11:45:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1c7d9e4f2b8'
down_revision: Union[str, Sequence[str], None] = '64f167b74622'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # SQLite 走 batch 模式（render_as_batch=True 已在 env.py 设好），
    # 但加列本身不需要重建表；server_default 保证老行有值。
    with op.batch_alter_table('tasks', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column('pause_requested', sa.Integer(), nullable=True,
                      server_default='0')
        )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('tasks', schema=None) as batch_op:
        batch_op.drop_column('pause_requested')

"""add email to users

独立注册页配套（用户 m25143 第 2 项）：注册页除用户名/密码外增加**可选**邮箱。

为什么可空：老账号（如 `demo`）没有邮箱，注册页不填也不该拦住注册。
SQLite 里 UNIQUE 允许多行 NULL，所以「唯一 + 可空」正好表达「填了才唯一」。

Revision ID: b2d8e5f3a9c1
Revises: a1c7d9e4f2b8
Create Date: 2026-10-12 10:20:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b2d8e5f3a9c1'
down_revision: Union[str, Sequence[str], None] = 'a1c7d9e4f2b8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.add_column(sa.Column('email', sa.String(length=128), nullable=True))
        # 唯一约束走 batch 模式（SQLite 需重建表）；NULL 不参与唯一性比较，
        # 因此未填邮箱的历史行可以共存。
        batch_op.create_unique_constraint('uq_users_email', ['email'])


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.drop_constraint('uq_users_email', type_='unique')
        batch_op.drop_column('email')

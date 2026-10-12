"""password reset: users.token_version + password_resets table

找回密码功能（用户 m26093）的数据层。

两个改动：

1. **`users.token_version`** —— 改密后作废旧会话。
   JWT 是无状态的，服务端无法撤回已发出的 token；把版本号写进载荷、校验时
   与库中值比对，改密即 +1，此前签发的所有 token 立即失效。
   `server_default='0'` 让历史行（如 `demo`）不必回填。

2. **`password_resets`** —— 一次性重置凭据。
   只存 `sha256(token)`，明文仅存在于邮件里（与「密码只存 bcrypt 哈希」同理）。
   `used_at` 非空即作废，链接只能点一次。

Revision ID: c3e9f6a4b7d2
Revises: b2d8e5f3a9c1
Create Date: 2026-10-12 20:10:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c3e9f6a4b7d2'
down_revision: Union[str, Sequence[str], None] = 'b2d8e5f3a9c1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column('token_version', sa.Integer(), nullable=False, server_default='0')
        )

    op.create_table(
        'password_resets',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('token_hash', sa.String(length=64), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.Column('used_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    # 两个索引都有实际用途：按用户作废（改密时）、按摘要查（校验时）。
    # ★ token_hash 是 **UNIQUE 索引**（不是唯一约束 + 普通索引）——
    #   与 ORM 的 `unique=True, index=True` 完全对应，否则 `alembic check`
    #   会报 schema 漂移（verify_alembic.py 的 [6] 就是在盯这个）。
    op.create_index('ix_password_resets_user_id', 'password_resets', ['user_id'])
    op.create_index('ix_password_resets_token_hash', 'password_resets',
                    ['token_hash'], unique=True)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_password_resets_token_hash', table_name='password_resets')
    op.drop_index('ix_password_resets_user_id', table_name='password_resets')
    op.drop_table('password_resets')
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.drop_column('token_version')

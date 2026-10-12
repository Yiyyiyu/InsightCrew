"""email verification: users.email_verified_at + email_codes table

邮箱验证（用户 m26742）的数据层。

## 1. `users.email_verified_at`

在此之前「填了邮箱」就等于「绑定了邮箱」。但邮箱是找回密码的**唯一凭据**：
写错一个字母的邮箱和没绑邮箱一样收不到信，区别是前者看起来像绑定成功了，
用户到忘记密码那天才会发现 —— 那时已经登不进来。

所以把「填了」和「验证过」拆开：只有收到验证码并校验通过，才写这一列。
`nullable=True`，历史行为 NULL —— 既有用户（含 `demo`）需要重新验证一次邮箱
才能使用需要邮箱确认的操作（如改密码）。

## 2. `email_codes`

独立表而不是往 users 上加列，原因有二：

  * 一个用户会反复申请验证码（换绑、改密、改错重来），是一对多；
  * **注册时用户还不存在**，没有 user_id 可挂，只能按邮箱定位。

`purpose` 把不同用途的码隔离开：同一个人可能同时有「验证当前邮箱」与
「换绑到新邮箱」两个待办，只按邮箱查码的话，一个发给旧邮箱的码就能通过
换绑校验，验证就白做了。

码本身只存 `HMAC(密钥, code)` 而非明文，且带 `attempts` 计数 —— 6 位数字的
裸 sha256 一秒就能撞出全表，必须带密钥且限制尝试次数。

Revision ID: d4f0a7b5c8e3
Revises: c3e9f6a4b7d2
Create Date: 2026-10-13 10:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd4f0a7b5c8e3'
down_revision: Union[str, Sequence[str], None] = 'c3e9f6a4b7d2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column('email_verified_at', sa.DateTime(), nullable=True)
        )

    op.create_table(
        'email_codes',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('email', sa.String(length=128), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('purpose', sa.String(length=32), nullable=False),
        sa.Column('code_hash', sa.String(length=64), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.Column('used_at', sa.DateTime(), nullable=True),
        sa.Column('attempts', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    # ★ 两个索引都必须与 ORM 声明严格对应，否则 `alembic check` 报 schema 漂移：
    #   `email = Column(..., index=True)`            → ix_email_codes_email
    #   `__table_args__ = (Index("ix_email_codes_email_purpose", ...),)`
    #                                                 → ix_email_codes_email_purpose
    op.create_index('ix_email_codes_email', 'email_codes', ['email'])
    op.create_index('ix_email_codes_email_purpose', 'email_codes',
                    ['email', 'purpose'])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_email_codes_email_purpose', table_name='email_codes')
    op.drop_index('ix_email_codes_email', table_name='email_codes')
    op.drop_table('email_codes')
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.drop_column('email_verified_at')

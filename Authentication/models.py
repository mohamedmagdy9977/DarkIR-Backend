from datetime import date, datetime

from sqlalchemy import BigInteger, Boolean, Column, Date, DateTime, ForeignKey, String
from sqlalchemy.orm import relationship

from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(BigInteger, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    last_login = Column(DateTime, nullable=True)
    created_at = Column(Date, default=date.today, nullable=False)

    refresh_tokens = relationship("RefreshToken", back_populates="user")
    jobs = relationship("Job", back_populates="user")


class RefreshToken(Base):
    __tablename__ = "refresh_token"

    id = Column(BigInteger, primary_key=True, index=True)
    user_id = Column(BigInteger, ForeignKey("users.id"), nullable=False, index=True)
    token = Column(String, unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expire_at = Column(DateTime, nullable=False)

    user = relationship("User", back_populates="refresh_tokens")

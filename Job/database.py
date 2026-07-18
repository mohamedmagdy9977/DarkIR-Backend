from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from Authentication.config import settings
from Authentication.database import get_db
from models import Job


def create_job(db: Session, job: Job):
    db.add(job)
    db.commit()
    db.refresh(job)
    return job

def get_job_by_id(db: Session, job_id: int):
    return db.query(Job).filter(Job.id == job_id).first()

def get_jobs_by_user_id(db: Session, user_id: int, skip: int = 0, limit: int = 10):
    return db.query(Job).offset(skip).limit(limit).all()
from datetime import datetime

import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordRequestForm
from app.api.deps import oauth2_scheme
from app.db.session import get_db
from app.db.models import TokenDenylistRecord, UserRecord
from app.schemas.auth import LogoutResponse, UserCreate, UserResponse, Token
from app.core.security import (
    create_access_token,
    decode_access_token,
    get_password_hash,
    verify_password,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/register", response_model=UserResponse)
def register(user_in: UserCreate, db: Session = Depends(get_db)):
    user = db.query(UserRecord).filter(UserRecord.email == user_in.email).first()
    if user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    new_user = UserRecord(
        email=user_in.email,
        hashed_password=get_password_hash(user_in.password)
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@router.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(UserRecord).filter(UserRecord.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(subject=user.id)
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/logout", response_model=LogoutResponse)
def logout(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = decode_access_token(token)
        user_id: str | None = payload.get("sub")
        jti: str | None = payload.get("jti")
        expires_at_timestamp: int | None = payload.get("exp")
        if user_id is None or jti is None or expires_at_timestamp is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception

    denied_token = db.query(TokenDenylistRecord).filter(
        TokenDenylistRecord.jti == jti
    ).first()
    if denied_token is None:
        db.add(
            TokenDenylistRecord(
                jti=jti,
                user_id=user_id,
                expires_at=datetime.utcfromtimestamp(expires_at_timestamp),
            )
        )
        db.commit()

    return {"detail": "Logged out successfully"}

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import jwt
from sqlalchemy.orm import Session
from app.core.config import get_settings
from app.core.security import decode_access_token
from app.db.session import get_db
from app.db.models import TokenDenylistRecord, UserRecord

settings = get_settings()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

def get_current_user(
    db: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme)
) -> UserRecord:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_access_token(token)
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception

    jti: str | None = payload.get("jti")
    if jti is not None:
        denied_token = db.query(TokenDenylistRecord).filter(
            TokenDenylistRecord.jti == jti
        ).first()
        if denied_token is not None:
            raise credentials_exception
        
    user = db.query(UserRecord).filter(UserRecord.id == user_id).first()
    if user is None:
        raise credentials_exception
    return user

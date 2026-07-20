from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request

from app.api.deps import get_current_user
from app.core.config import get_settings
from app.core.rate_limit import limiter, user_or_ip
from app.db.models import UserRecord
from app.services.evaluation.rag_metrics import evaluate_dataset
from app.services.evaluation.test_datasets import DATASET2

router = APIRouter(
    prefix="/test",
    tags=["Test Metrics"],
)
settings = get_settings()


@router.get("/metrics")
@limiter.limit(settings.evaluation_rate_limit, key_func=user_or_ip)
async def test_metrics(
    request: Request,
    current_user: Annotated[UserRecord, Depends(get_current_user)],
    k: Annotated[int, Query(ge=1, le=20)] = 5,
):
    return await evaluate_dataset(
        dataset=DATASET2,
        k=k,
        user_id=current_user.id,
    )

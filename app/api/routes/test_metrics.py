from typing import Annotated

from fastapi import APIRouter, Query

from app.services.evaluation.rag_metrics import evaluate_dataset
from app.services.evaluation.test_datasets import DATASET2


router = APIRouter(
    prefix="/test",
    tags=["Test Metrics"],
)


@router.get("/metrics")
def test_metrics(
    k: Annotated[int, Query(ge=1, le=20)] = 5,
):
    return evaluate_dataset(
        dataset=DATASET2,
        k=k,
    )

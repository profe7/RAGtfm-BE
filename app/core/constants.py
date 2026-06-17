from enum import StrEnum


class DocumentStatus(StrEnum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    READY = "READY"
    FAILED = "FAILED"

"""Evidence assembly: candidate summaries and the requirement matrix."""

from .profiles import (
    CandidateRecord,
    JobRecord,
    ProfileSummariser,
    ProfileSummary,
    discover_candidates,
    discover_jobs,
)
from .requirements import (
    NOT_FOUND_RATIONALE,
    ExtractedRequirement,
    RequirementMatrixBuilder,
    extract_requirements,
)

__all__ = [
    "NOT_FOUND_RATIONALE",
    "CandidateRecord",
    "ExtractedRequirement",
    "JobRecord",
    "ProfileSummariser",
    "ProfileSummary",
    "RequirementMatrixBuilder",
    "discover_candidates",
    "discover_jobs",
    "extract_requirements",
]

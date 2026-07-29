"""Stable route names shared by Lesson 17 training and evaluation."""

from __future__ import annotations


SFT_ONLY_ROUTE = "Lesson17-SFT-Only"
RLVR_ROUTE = "Lesson17-RLVR"
DIRECT_RLAIF_ROUTE = "Lesson17-Direct-RLAIF"
COMBINED_ROUTE = "Lesson17-RLVR-Plus-Direct-RLAIF"

TRAINABLE_ROUTES = frozenset(
    {RLVR_ROUTE, DIRECT_RLAIF_ROUTE, COMBINED_ROUTE}
)
ALL_LESSON17_ROUTES = frozenset(
    {SFT_ONLY_ROUTE, *TRAINABLE_ROUTES}
)

ROUTE_LABELS = {
    SFT_ONLY_ROUTE: "SFT-only",
    RLVR_ROUTE: "RLVR",
    DIRECT_RLAIF_ROUTE: "direct-RLAIF",
    COMBINED_ROUTE: "RLVR + direct-RLAIF",
}

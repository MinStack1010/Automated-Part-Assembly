"""Hierarchical motion-planning strategy with bounded fallbacks.

Strategy A: helical (screw) motion estimated from or pinned to the real thread.
Strategy B: combined translation+rotation via bfs/bk-rrt (rotation=True), so
            the moving part may unscrew gradually while translating.
Strategy C: limited retry with alternative seed / planner.

Every strategy has an explicit time budget; the overall deadline is shared so
retries can never exceed it. A strategy timeout triggers the next eligible
strategy instead of failing the whole task.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from api.config import settings

logger = logging.getLogger(__name__)


def _validate_path_candidate(
    assembly_dir: Path,
    move_id: str,
    still_ids: list[str],
    path: list[np.ndarray],
    axis_estimate: Any = None,
) -> Optional[dict[str, Any]]:
    """Validate a candidate path and return its validation result, or None if invalid."""
    from api.services.axis_constrained import _load_parts, validate_full_path

    try:
        parts = _load_parts(assembly_dir)
        if move_id not in parts or not all(sid in parts for sid in still_ids):
            return None
        move_mesh = parts[move_id]
        still_meshes = [parts[sid] for sid in still_ids]
        result = validate_full_path(path, move_mesh, still_meshes, axis=axis_estimate)
        return {
            "valid": result.valid,
            "max_lateral_deviation": round(result.max_lateral_deviation, 6),
            "max_angular_deviation": round(result.max_angular_deviation, 6),
            "path_length": round(result.path_length, 4),
            "final_separated": result.final_separated,
            "collision_states": result.collision_states,
            "score": round(result.score, 6),
            "rejection_reason": result.rejection_reason,
        }
    except Exception:
        logger.warning("PATH_VALIDATION_FAILED move=%s reason=validation_error", move_id, exc_info=True)
        return None


@dataclass
class FallbackConfig:
    """Time budgets (seconds) for each planning strategy.

    All values are fractions/absolute seconds derived from the caller-supplied
    overall ``max_time``; the orchestrator never exceeds the overall deadline.
    """
    helical_fraction: float = 0.5
    combined_fraction: float = 0.35
    alternative_fraction: float = 0.15
    alternative_seeds: tuple[int, ...] = (7, 13)

    @classmethod
    def from_max_time(cls, max_time: float, seed: int = 1) -> FallbackConfig:
        return cls(
            helical_fraction=max_time * settings.helical_budget_fraction,
            combined_fraction=max_time * settings.combined_budget_fraction,
            alternative_fraction=max_time * settings.alternative_budget_fraction,
            alternative_seeds=(seed + 6, seed + 12),
        )


@dataclass
class StrategyResult:
    """Outcome of one planning strategy attempt."""
    strategy: str
    status: str
    elapsed_seconds: float
    path: list[np.ndarray] | None = None
    planner: str | None = None
    info: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None
    timed_out: bool = False


class PlanningStrategyRunner:
    """Run helical → combined → alternative strategies under a shared deadline."""

    def __init__(
        self,
        assembly_dir: Path,
        move_id: str,
        still_ids: list[str],
        max_time: float,
        seed: int = 1,
        rotation: bool = False,
        body_type: str = "sdf",
        sdf_dx: float = 0.05,
        collision_threshold: float = 0.01,
        force_magnitude: float = 100.0,
        frame_skip: int = 100,
        helical_spec: Optional[dict[str, Any]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        config: Optional[FallbackConfig] = None,
        axis_estimate: Optional[Any] = None,
    ) -> None:
        self.assembly_dir = assembly_dir
        self.move_id = move_id
        self.still_ids = still_ids
        self.max_time = max_time
        self.seed = seed
        self.rotation = rotation
        self.body_type = body_type
        self.sdf_dx = sdf_dx
        self.collision_threshold = collision_threshold
        self.force_magnitude = force_magnitude
        self.frame_skip = frame_skip
        self.helical_spec = helical_spec
        self.cancel_check = cancel_check
        self.config = config or FallbackConfig.from_max_time(max_time, seed)
        self.deadline = time.monotonic() + max_time
        self.attempts: list[StrategyResult] = []
        self.cancelled = False
        self.axis_estimate = axis_estimate
        if self.axis_estimate is None:
            try:
                from api.services.axis_constrained import estimate_assembly_axis

                self.axis_estimate = estimate_assembly_axis(
                    assembly_dir, move_id, still_ids, thread_info=None
                )
                if self.axis_estimate is not None:
                    logger.info(
                        "AXIS_ESTIMATED move=%s source=%s confidence=%.2f axis=%s",
                        move_id, self.axis_estimate.source,
                        self.axis_estimate.confidence,
                        np.round(self.axis_estimate.axis, 4).tolist(),
                    )
            except Exception:
                logger.info("AXIS_ESTIMATION_FAILED move=%s", move_id, exc_info=True)
                self.axis_estimate = None

    def _remaining(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    def _check_cancelled(self) -> bool:
        if self.cancel_check is not None and self.cancel_check():
            self.cancelled = True
            logger.info("PLANNING CANCELLED move=%s before next strategy", self.move_id)
            return True
        return False

    def _run_helical(self, time_budget: float) -> StrategyResult | None:
        logger.info(
            "HELICAL_ATTEMPT_STARTED move=%s budget=%.1fs spec=%s",
            self.move_id, time_budget, "explicit" if self.helical_spec else "auto",
        )
        from api.services.helical import build_helical_path

        started = time.perf_counter()
        status, elapsed, path, info = build_helical_path(
            self.assembly_dir,
            self.move_id,
            self.still_ids,
            max_time=time_budget,
            helical=self.helical_spec,
        )
        elapsed = time.perf_counter() - started
        timed_out = status == "Timeout"
        reason = info.get("reason")
        if status == "Success":
            logger.info(
                "HELICAL_PATH_FOUND move=%s elapsed=%.2fs steps=%d strategy=helical",
                self.move_id, elapsed, len(path) if path else 0,
            )
            return StrategyResult(
                strategy="helical", status="Success", elapsed_seconds=elapsed,
                path=path, planner="helical", info=info,
            )
        if timed_out:
            logger.warning(
                "HELICAL_TIMEOUT_FALLBACK move=%s elapsed=%.2fs reason=timeout",
                self.move_id, elapsed,
            )
        else:
            logger.warning(
                "HELICAL_PATH_NOT_FOUND move=%s elapsed=%.2fs reason=%s",
                self.move_id, elapsed, reason,
            )
        return StrategyResult(
            strategy="helical", status=status, elapsed_seconds=elapsed,
            info=info, reason=reason, timed_out=timed_out,
        )

    def _run_combined(self, planner_name: str, seed: int, time_budget: float, rotation: bool) -> StrategyResult | None:
        logger.info(
            "COMBINED_MOTION_ATTEMPT_STARTED move=%s planner=%s rotation=%s seed=%d budget=%.1fs",
            self.move_id, planner_name, rotation, seed, time_budget,
        )
        from examples.run_joint_plan import get_planner

        started = time.perf_counter()
        planner = get_planner(planner_name)(
            str(settings.assets_dir),
            str(self.assembly_dir),
            self.move_id,
            self.still_ids,
            rotation,
            self.body_type,
            self.sdf_dx,
            self.collision_threshold,
            self.force_magnitude,
            self.frame_skip,
            False,
        )
        status, elapsed, path = planner.plan(
            time_budget, seed=seed, return_path=True, render=False, record_path=None,
        )
        elapsed = time.perf_counter() - started
        timed_out = status == "Timeout"
        if status == "Success":
            logger.info(
                "COMBINED_MOTION_PATH_FOUND move=%s planner=%s seed=%d elapsed=%.2fs steps=%d",
                self.move_id, planner_name, seed, elapsed, len(path) if path else 0,
            )
            return StrategyResult(
                strategy="combined", status="Success", elapsed_seconds=elapsed,
                path=path, planner=planner_name,
                info={"rotation": rotation, "seed": seed},
            )
        if timed_out:
            logger.warning(
                "COMBINED_MOTION_TIMEOUT move=%s planner=%s seed=%d elapsed=%.2fs",
                self.move_id, planner_name, seed, elapsed,
            )
        else:
            logger.warning(
                "COMBINED_MOTION_NOT_FOUND move=%s planner=%s seed=%d status=%s",
                self.move_id, planner_name, seed, status,
            )
        return StrategyResult(
            strategy="combined", status=status, elapsed_seconds=elapsed,
            planner=planner_name, info={"rotation": rotation, "seed": seed},
            timed_out=timed_out,
        )

    def _run_alternative(self) -> StrategyResult | None:
        logger.info("ALTERNATIVE_PLANNER_ATTEMPT_STARTED move=%s", self.move_id)
        planners = ["bk-rrt", "bfs"] if self.config.alternative_seeds else ["bfs"]
        for seed in self.config.alternative_seeds:
            if self._check_cancelled() or self._remaining() <= 0.5:
                return None
            budget = min(self.config.alternative_fraction, self._remaining())
            if budget <= 0.5:
                return None
            for planner_name in planners:
                if self._check_cancelled() or self._remaining() <= 0.5:
                    return None
                budget = min(self.config.alternative_fraction, self._remaining())
                result = self._run_combined(planner_name, seed, budget, rotation=True)
                self.attempts.append(result)
                if result.status == "Success":
                    result.strategy = "alternative"
                    logger.info(
                        "ALTERNATIVE_PLANNER_PATH_FOUND move=%s planner=%s seed=%d",
                        self.move_id, planner_name, seed,
                    )
                    return result
        return None

    def _accept_candidate(self, result: StrategyResult) -> bool:
        """Validate a candidate path; return True if it should be accepted."""
        if result.path is None or not result.path:
            return False
        if self.axis_estimate is None:
            return True
        validation = _validate_path_candidate(
            self.assembly_dir, self.move_id, self.still_ids,
            result.path, self.axis_estimate,
        )
        if validation is None or not validation.get("valid"):
            logger.warning(
                "OFF_AXIS_PATH_REJECTED move=%s strategy=%s reason=%s",
                self.move_id, result.strategy,
                (validation or {}).get("rejection_reason", "unknown"),
            )
            return False
        result.info["validation"] = validation
        logger.info(
            "VALID_SEPARATION_PATH_FOUND move=%s strategy=%s lateral=%.4f score=%.3f",
            self.move_id, result.strategy,
            validation.get("max_lateral_deviation", 0.0),
            validation.get("score", 0.0),
        )
        return True

    def run(self) -> StrategyResult | None:
        """Execute strategies in order; return the first successful result."""
        # Strategy A: helical
        budget = min(self.config.helical_fraction, self._remaining())
        if budget > 0.5 and not self._check_cancelled():
            result = self._run_helical(budget)
            self.attempts.append(result)
            if result.status == "Success" and self._accept_candidate(result):
                return result

        # Strategy B: combined translation + rotation
        if not self._check_cancelled() and self._remaining() > 0.5:
            budget = min(self.config.combined_fraction, self._remaining())
            for planner_name in ("bfs", "bk-rrt"):
                if self._check_cancelled() or self._remaining() <= 0.5:
                    break
                budget = min(self.config.combined_fraction, self._remaining())
                result = self._run_combined(planner_name, self.seed, budget, rotation=True)
                self.attempts.append(result)
                if result.status == "Success" and self._accept_candidate(result):
                    return result

        # Strategy C: alternative seeds / planners
        if not self._check_cancelled() and self._remaining() > 0.5:
            result = self._run_alternative()
            if result is not None and result.status == "Success" and self._accept_candidate(result):
                return result

        logger.error(
            "ALL_STRATEGIES_EXHAUSTED move=%s attempts=%d cancelled=%s",
            self.move_id, len(self.attempts), self.cancelled,
        )
        return None

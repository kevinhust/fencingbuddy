"""
FencerAI Coaching Engine
=======================
Version: 2.0 | Last Updated: 2026-04-02

Rule-based coaching system for real-time fencing feedback.
Evaluates fencing metrics and generates actionable alerts.

Initial Alert Library (10 alerts):
    1. "Shorten recovery — riposte risk"
    2. "Attack now — distance open"
    3. "Extend arm fully"
    4. "Opponent favors 4th — attack 5th"
    5. "Son drops guard on retreat"
    6. "Counter-attack opportunity"
    7. "Distance closing — parry-riposte ready"
    8. "Watch for fleche attack"
    9. "Recovery stance too wide"
    10. "Trust your attack — you're fast enough"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Callable
from enum import Enum
import numpy as np

from src.coaching.coaching_metrics import (
    CoachingMetrics,
    FencingMetrics,
    extract_lunge_speed,
    extract_arm_extension,
    extract_recovery_speed,
    ARM_EXTENSION_MIN,
    ARM_EXTENSION_FULL,
    LUNGE_SPEED_THRESHOLD,
    DISTANCE_OPTIMAL_ATTACK,
    DISTANCE_TOO_FAR,
    PREDICTABILITY_HIGH,
)
from src.recognition.action_classifier import ActionClassifier, ActionType, ActionResult
from src.ui.alert_renderer import CoachingAlert


# =============================================================================
# Alert Categories
# =============================================================================

class AlertCategory(Enum):
    """Categories for organizing coaching alerts."""
    DISTANCE = "distance"
    ATTACK = "attack"
    DEFENSE = "defense"
    RECOVERY = "recovery"
    BLADE_WORK = "blade"
    GENERAL = "general"


# =============================================================================
# Alert Rules
# =============================================================================

@dataclass
class AlertRule:
    """
    A rule that triggers a coaching alert based on metric conditions.

    Attributes:
        name: Unique identifier for the rule
        message: Alert text shown to coach
        category: Alert category for filtering
        priority: 1=critical, 5=info
        fencer_id: 0=son, 1=opponent, None=both
        condition: Function that evaluates if alert should trigger
        cooldown_frames: Minimum frames between same alerts
    """
    name: str
    message: str
    category: AlertCategory
    priority: int = 3
    fencer_id: Optional[int] = None  # None = apply to both
    condition: Callable[[FencingMetrics, FencingMetrics, Optional[FencingMetrics]], bool] = field(
        default=lambda son, opp, rel: False
    )
    cooldown_frames: int = 30  # ~1 second at 30fps


# =============================================================================
# Coaching Engine
# =============================================================================

class CoachingEngine:
    """
    Real-time coaching alert system for fencing analysis.

    Evaluates fencing metrics against rule-based conditions to generate
    actionable coaching alerts displayed on the live viewer.

    Example:
        >>> engine = CoachingEngine()
        >>> alerts = engine.evaluate(features_2x101)
        >>> for alert in alerts:
        ...     viewer.add_alert(alert.message, priority=alert.priority)
    """

    def __init__(self):
        """Initialize coaching engine with alert rules."""
        self._metrics = CoachingMetrics(history_size=5)
        self._action_classifier = ActionClassifier(history_size=10)
        self._last_alert_frames: Dict[str, int] = {}
        self._frame_count = 0
        self._cooldown_frames = 30  # Default cooldown
        self._last_action_result: Optional[ActionResult] = None

        # Initialize alert rules
        self._rules = self._build_alert_rules()

    def evaluate(
        self,
        features: np.ndarray,
        min_priority: int = 1,
        timestamp: float = 0.0,
    ) -> List[CoachingAlert]:
        """
        Evaluate current frame and return list of active alerts.

        Args:
            features: (2, 101) feature matrix for both fencers
            min_priority: Minimum priority level to include (1-5)
            timestamp: Current timestamp for action classification

        Returns:
            List of CoachingAlert objects to display
        """
        self._frame_count += 1
        alerts = []

        # Compute metrics for both fencers
        son_metrics, opp_metrics, relative = self._metrics.compute_both_fencers_metrics(features)

        # Classify actions
        self._last_action_result = self._action_classifier.classify(
            features, son_metrics, opp_metrics, timestamp
        )

        # Evaluate each rule
        for rule in self._rules:
            if rule.priority > min_priority:
                continue

            # Check cooldown
            if self._is_in_cooldown(rule.name):
                continue

            # Evaluate condition
            try:
                triggered = rule.condition(son_metrics, opp_metrics, relative)
            except Exception:
                triggered = False

            if triggered:
                # Determine which fencer this applies to
                fencer_id = rule.fencer_id

                alert = CoachingAlert(
                    message=rule.message,
                    priority=rule.priority,
                    fencer_id=fencer_id,
                    category=rule.category.value,
                    duration=5.0,
                )
                alerts.append(alert)
                self._last_alert_frames[rule.name] = self._frame_count

        return alerts

    def get_last_action(self) -> Optional[ActionResult]:
        """Get the last action classification result."""
        return self._last_action_result

    def _is_in_cooldown(self, rule_name: str) -> bool:
        """Check if a rule is in cooldown period."""
        if rule_name not in self._last_alert_frames:
            return False
        frames_since = self._frame_count - self._last_alert_frames[rule_name]
        return frames_since < self._cooldown_frames

    def reset(self) -> None:
        """Reset engine state for new session."""
        self._metrics.reset()
        self._action_classifier.reset()
        self._last_alert_frames.clear()
        self._frame_count = 0

    def _build_alert_rules(self) -> List[AlertRule]:
        """
        Build the initial set of 10 coaching alert rules.

        Returns:
            List of AlertRule objects
        """
        rules = []

        # Rule 1: Shorten recovery — riposte risk
        rules.append(AlertRule(
            name="slow_recovery",
            message="Shorten recovery — riposte risk",
            category=AlertCategory.RECOVERY,
            priority=1,
            fencer_id=0,
            condition=lambda son, opp, rel: (
                son.is_recovering and
                son.acceleration_magnitude < 5.0 and
                opp.lunge_speed > LUNGE_SPEED_THRESHOLD
            ),
        ))

        # Rule 2: Attack now — distance open
        rules.append(AlertRule(
            name="attack_distance_open",
            message="Attack now — distance open",
            category=AlertCategory.DISTANCE,
            priority=2,
            fencer_id=0,
            condition=lambda son, opp, rel: (
                son.distance_to_opponent > DISTANCE_OPTIMAL_ATTACK and
                son.distance_to_opponent < DISTANCE_TOO_FAR and
                opp.lunge_speed < ATTACK_PREP_SPEED and
                son.lunge_speed > 0
            ),
        ))

        # Rule 3: Extend arm fully
        rules.append(AlertRule(
            name="arm_extension",
            message="Extend arm fully on next attack",
            category=AlertCategory.ATTACK,
            priority=2,
            fencer_id=0,
            condition=lambda son, opp, rel: (
                son.arm_extension_ratio < ARM_EXTENSION_MIN and
                son.lunge_speed > LUNGE_SPEED_THRESHOLD
            ),
        ))

        # Rule 4: Predictability warning
        rules.append(AlertRule(
            name="predictable",
            message="Too predictable — vary attacks",
            category=AlertCategory.ATTACK,
            priority=3,
            fencer_id=0,
            condition=lambda son, opp, rel: (
                son.predictability_score > PREDICTABILITY_HIGH and
                son.lunge_speed > LUNGE_SPEED_THRESHOLD
            ),
            cooldown_frames=60,  # Longer cooldown for this
        ))

        # Rule 5: Son drops guard on retreat
        rules.append(AlertRule(
            name="dropped_guard",
            message="Watch guard position on retreat",
            category=AlertCategory.DEFENSE,
            priority=2,
            fencer_id=0,
            condition=lambda son, opp, rel: (
                son.is_retreating and
                abs(son.torso_lateral_tilt) > 0.3
            ),
        ))

        # Rule 6: Counter-attack opportunity
        rules.append(AlertRule(
            name="counter_opportunity",
            message="Counter-attack — opponent overextended",
            category=AlertCategory.DEFENSE,
            priority=2,
            fencer_id=1,
            condition=lambda son, opp, rel: (
                opp.is_attacking and
                opp.arm_extension_ratio > ARM_EXTENSION_FULL and
                son.lunge_speed > 0
            ),
        ))

        # Rule 7: Distance closing — parry-riposte ready
        rules.append(AlertRule(
            name="distance_closing",
            message="Distance closing — parry-riposte ready",
            category=AlertCategory.DISTANCE,
            priority=3,
            fencer_id=0,
            condition=lambda son, opp, rel: (
                rel.opponent_distance_change < -0.01 and  # Closing
                opp.arm_extension_ratio > ARM_EXTENSION_MIN and
                son.is_retreating is False
            ),
        ))

        # Rule 8: Watch for fleche
        rules.append(AlertRule(
            name="fleche_warning",
            message="Watch for fleche — opponent leaning forward",
            category=AlertCategory.DEFENSE,
            priority=2,
            fencer_id=1,
            condition=lambda son, opp, rel: (
                opp.torso_forward_lean > 0.4 and
                opp.lunge_speed > LUNGE_SPEED_THRESHOLD * 0.8
            ),
        ))

        # Rule 9: Recovery stance too wide
        rules.append(AlertRule(
            name="wide_stance",
            message="Close stance after lunge",
            category=AlertCategory.RECOVERY,
            priority=3,
            fencer_id=0,
            condition=lambda son, opp, rel: (
                son.is_recovering and
                son.front_knee_angle < KNEE_ANGLE_DEEP
            ),
        ))

        # Rule 10: Trust your attack — you're fast enough
        rules.append(AlertRule(
            name="trust_attack",
            message="Trust your attack — you're faster",
            category=AlertCategory.ATTACK,
            priority=4,
            fencer_id=0,
            condition=lambda son, opp, rel: (
                son.lunge_speed > opp.lunge_speed * 1.5 and
                son.lunge_speed > LUNGE_SPEED_THRESHOLD and
                opp.arm_extension_ratio < ARM_EXTENSION_MIN
            ),
            cooldown_frames=90,  # Rare alert
        ))

        return rules

    def get_metrics(self) -> CoachingMetrics:
        """Get the metrics extractor for external access."""
        return self._metrics


# =============================================================================
# Default Alert Messages (Fencing-Specific)
# =============================================================================

DEFAULT_ALERT_MESSAGES = {
    # Distance
    "distance_open": "Attack now - distance open",
    "distance_closed": "Too close - parry ready",
    "distance_far": "Close distance first",

    # Recovery
    "slow_recovery": "Shorten recovery - riposte risk",
    "fast_recovery": "Good recovery speed",

    # Attack
    "attack_prep": "Attack in prep - now!",
    "arm_extension": "Extend arm fully",
    "predictable": "Too predictable - vary attacks",

    # Defense
    "guard_low": "Watch your guard position",
    "dropped_guard": "Dropped guard on retreat",

    # Blade work
    "weak_blade": "Weak blade control",
    "fleche_prep": "Watch for fleche attack",

    # Opponent
    "opp_overextended": "Opponent overextended - counter",
    "opp_weak_recovery": "Opponent slow recovery - attack",
}


# Knee angle constant for deep lunge check
KNEE_ANGLE_DEEP = 1.0

# Attack prep speed constant
ATTACK_PREP_SPEED = 5.0

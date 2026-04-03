"""
FencerAI Action Classifier
=========================
Version: 2.0 | Last Updated: 2026-04-02

Rule-based action classification for fencing actions.
Classifies actions as attack, parry, riposte, counter-attack, etc.

Action Classification Rules:
    - Attack: Forward lunge motion + arm extension
    - Parry: Blade movement indicating defensive action
    - Riposte: Attack following a parry
    - Counter-attack: Attack during opponent's attack preparation
    - Fleche: Rapid forward attack with body lean

Example:
    classifier = ActionClassifier()
    action = classifier.classify(features_2x101, son_metrics, opp_metrics)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List
import numpy as np

from src.coaching.coaching_metrics import FencingMetrics


# =============================================================================
# Action Types
# =============================================================================

class ActionType(Enum):
    """Fencing action types."""
    IDLE = "idle"
    ADVANCE = "advance"
    RETREAT = "retreat"
    ATTACK = "attack"
    ATTACK_PREP = "attack_prep"
    PARRY = "parry"
    RIPOSTE = "riposte"
    COUNTER_ATTACK = "counter_attack"
    FLECHE = "fleche"
    RECOVERY = "recovery"


# =============================================================================
# Action Result
# =============================================================================

@dataclass
class ActionResult:
    """Result of action classification."""
    action: ActionType
    confidence: float  # 0.0 - 1.0
    son_action: ActionType
    opp_action: ActionType
    timestamp: float
    notes: str = ""


# =============================================================================
# Thresholds
# =============================================================================

# Attack thresholds
ATTACK_LUNGE_SPEED = 15.0
ATTACK_ARM_EXTENSION = 0.80
ATTACK_FORWARD_MOTION = 5.0

# Fleche thresholds
FLECHE_TORSO_LEAN = 0.4
FLECHE_LUNGE_SPEED = 25.0

# Parry thresholds (blade movement patterns)
PARRY_BLADE_CHANGE = 0.3  # Significant change in blade angle

# Counter-attack thresholds
COUNTER_SPEED_RATIO = 1.2  # Must be faster than opponent

# Recovery thresholds
RECOVERY_DECEL = -5.0


# =============================================================================
# Action Classifier
# =============================================================================

class ActionClassifier:
    """
    Rule-based action classifier for fencing actions.

    Uses heuristics based on fencing domain knowledge:
    - Attack: Forward lunge + arm extension
    - Fleche: Deep forward lean + high speed
    - Parry: Specific blade position changes
    - Riposte: Attack following parry within time window

    Example:
        >>> classifier = ActionClassifier()
        >>> result = classifier.classify(features, son_metrics, opp_metrics)
        >>> print(f"Action: {result.son_action.value}")
    """

    def __init__(self, history_size: int = 10):
        """
        Initialize action classifier.

        Args:
            history_size: Number of past frames for sequence analysis
        """
        self.history_size = history_size
        self._action_history: List[ActionResult] = []
        self._last_son_action: Optional[ActionType] = None
        self._parry_timestamp: Optional[float] = None

    def classify(
        self,
        features: np.ndarray,
        son_metrics: FencingMetrics,
        opp_metrics: FencingMetrics,
        timestamp: float = 0.0,
    ) -> ActionResult:
        """
        Classify current fencing actions.

        Args:
            features: (2, 101) feature matrix
            son_metrics: Computed metrics for son
            opp_metrics: Computed metrics for opponent
            timestamp: Current timestamp

        Returns:
            ActionResult with classification
        """
        # Classify individual fencers
        son_action = self._classify_fencer(son_metrics, opp_metrics, is_son=True)
        opp_action = self._classify_fencer(opp_metrics, son_metrics, is_son=False)

        # Determine overall action
        overall_action = self._determine_overall_action(son_action, opp_action, son_metrics)

        # Compute confidence
        confidence = self._compute_confidence(son_action, opp_action, son_metrics)

        # Build result
        result = ActionResult(
            action=overall_action,
            confidence=confidence,
            son_action=son_action,
            opp_action=opp_action,
            timestamp=timestamp,
        )

        # Update history
        self._action_history.append(result)
        if len(self._action_history) > self.history_size:
            self._action_history.pop(0)

        # Track for riposte detection
        if son_action == ActionType.PARRY:
            self._parry_timestamp = timestamp
        self._last_son_action = son_action

        return result

    def _classify_fencer(
        self,
        fencer: FencingMetrics,
        opponent: FencingMetrics,
        is_son: bool,
    ) -> ActionType:
        """Classify action for a single fencer."""
        # Check for fleche first (most distinctive)
        if self._is_fleche(fencer):
            return ActionType.FLECHE

        # Check for attack
        if self._is_attacking(fencer):
            # Check if this is a riposte (following parry)
            if self._is_riposte(fencer):
                return ActionType.RIPOSTE
            return ActionType.ATTACK

        # Check for parry
        if self._is_parry(fencer, opponent):
            return ActionType.PARRY

        # Check for counter-attack
        if self._is_counter_attack(fencer, opponent):
            return ActionType.COUNTER_ATTACK

        # Check for movement states
        if self._is_retreating(fencer):
            return ActionType.RETREAT
        if self._is_advancing(fencer):
            return ActionType.ADVANCE

        # Check for attack preparation
        if self._is_attack_prep(fencer):
            return ActionType.ATTACK_PREP

        # Check for recovery
        if self._is_recovering(fencer):
            return ActionType.RECOVERY

        return ActionType.IDLE

    def _is_fleche(self, fencer: FencingMetrics) -> bool:
        """Check if fencer is performing a fleche."""
        return (
            fencer.torso_forward_lean > FLECHE_TORSO_LEAN
            and fencer.lunge_speed > FLECHE_LUNGE_SPEED
            and fencer.is_attacking
        )

    def _is_attacking(self, fencer: FencingMetrics) -> bool:
        """Check if fencer is performing an attack."""
        return (
            fencer.lunge_speed > ATTACK_LUNGE_SPEED
            and fencer.arm_extension_ratio > ATTACK_ARM_EXTENSION
            and fencer.com_velocity_x > ATTACK_FORWARD_MOTION
        )

    def _is_riposte(self, fencer: FencingMetrics) -> bool:
        """Check if fencer is performing a riposte (attack following parry)."""
        if self._last_son_action == ActionType.PARRY and self._parry_timestamp is not None:
            # Riposte typically happens within 1 second of parry
            # For simplicity, just check if attacking after parry
            return self._is_attacking(fencer)
        return False

    def _is_parry(self, fencer: FencingMetrics, opponent: FencingMetrics) -> bool:
        """Check if fencer is performing a parry."""
        # Parry indicates defensive blade position
        # When opponent is attacking and fencer has good blade position
        if opponent.is_attacking:
            # Good defensive stance: arm extended, blade ready
            return (
                fencer.arm_extension_ratio > ATTACK_ARM_EXTENSION * 0.8
                and abs(fencer.weapon_elbow_angle) > 0.5  # Elbow angle indicates guard position
            )
        return False

    def _is_counter_attack(self, fencer: FencingMetrics, opponent: FencingMetrics) -> bool:
        """Check if fencer is counter-attacking."""
        return (
            fencer.lunge_speed > opponent.lunge_speed * COUNTER_SPEED_RATIO
            and opponent.is_attacking
            and not fencer.is_retreating
        )

    def _is_retreating(self, fencer: FencingMetrics) -> bool:
        """Check if fencer is retreating."""
        return fencer.is_retreating and fencer.lunge_speed < -3.0

    def _is_advancing(self, fencer: FencingMetrics) -> bool:
        """Check if fencer is advancing."""
        return fencer.lunge_speed > 3.0 and not fencer.is_attacking

    def _is_attack_prep(self, fencer: FencingMetrics) -> bool:
        """Check if fencer is in attack preparation."""
        return (
            fencer.lunge_speed > 2.0
            and fencer.lunge_speed < ATTACK_LUNGE_SPEED
            and fencer.arm_extension_ratio < ATTACK_ARM_EXTENSION
        )

    def _is_recovering(self, fencer: FencingMetrics) -> bool:
        """Check if fencer is recovering after an action."""
        return (
            fencer.is_decelerating
            and fencer.acceleration_magnitude < RECOVERY_DECEL
        )

    def _determine_overall_action(
        self,
        son_action: ActionType,
        opp_action: ActionType,
        son_metrics: FencingMetrics,
    ) -> ActionType:
        """Determine overall action (prioritize significant actions)."""
        # If son is doing something significant, that's the overall action
        if son_action in (ActionType.ATTACK, ActionType.FLECHE, ActionType.RIPOSTE):
            return son_action
        if son_action == ActionType.COUNTER_ATTACK:
            return son_action
        if son_action == ActionType.PARRY:
            return son_action

        # Return son's action as default
        return son_action

    def _compute_confidence(
        self,
        son_action: ActionType,
        opp_action: ActionType,
        son_metrics: FencingMetrics,
    ) -> float:
        """Compute confidence in the classification."""
        confidence = 0.5  # Base confidence

        # Increase confidence based on action clarity
        if son_action == ActionType.FLECHE:
            confidence = 0.9
        elif son_action == ActionType.ATTACK:
            if son_metrics.lunge_speed > ATTACK_LUNGE_SPEED * 1.5:
                confidence = 0.9
            else:
                confidence = 0.7
        elif son_action == ActionType.IDLE:
            confidence = 0.6

        return min(confidence, 1.0)

    def get_action_sequence(self) -> List[ActionType]:
        """Get recent action sequence."""
        return [r.action for r in self._action_history]

    def reset(self) -> None:
        """Reset classifier state."""
        self._action_history.clear()
        self._last_son_action = None
        self._parry_timestamp = None

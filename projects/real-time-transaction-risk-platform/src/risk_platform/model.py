"""Versioned calibrated baseline risk model artifact with integrity verification and fast inference."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Protocol, Sequence, Tuple, Union

from risk_platform.errors import ArtifactIntegrityError
from risk_platform.features import FEATURE_NAMES, FEATURE_VERSION
from risk_platform.schemas import FeatureAttribution, FeatureVector


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-min(x, 60.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(x, -60.0))
    return z / (1.0 + z)


def _compute_artifact_hash(payload_without_hash: Dict[str, Any]) -> str:
    canonical = json.dumps(payload_without_hash, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class RiskModelBackend(Protocol):
    """Inference backend boundary for risk scoring."""

    @property
    def model_version(self) -> str: ...

    @property
    def feature_version(self) -> str: ...

    def predict(
        self, feature_vector: FeatureVector
    ) -> Tuple[float, float, List[FeatureAttribution]]: ...


@dataclass(frozen=True)
class CalibratedLinearRiskModel:
    """Fast, pure-math calibrated logistic risk model with SHA-256 artifact verification.

    Runs in sub-millisecond wall time with deterministic linear feature attributions.
    """

    model_version: str
    feature_version: str
    feature_names: Tuple[str, ...]
    scaler_mean: Tuple[float, ...]
    scaler_scale: Tuple[float, ...]
    weights: Tuple[float, ...]
    intercept: float
    platt_a: float
    platt_b: float
    training_metadata: Dict[str, Any]
    artifact_sha256: str = ""

    def _raw_logit_and_attributions(
        self, feature_vector: FeatureVector
    ) -> Tuple[float, List[FeatureAttribution]]:
        if feature_vector.feature_version != self.feature_version:
            raise ArtifactIntegrityError(
                f"Feature version mismatch: model expects '{self.feature_version}', "
                f"got '{feature_vector.feature_version}'"
            )

        logit = self.intercept
        attributions: List[FeatureAttribution] = []
        for idx, name in enumerate(self.feature_names):
            raw_val = float(feature_vector.values[name])
            scale = self.scaler_scale[idx] if self.scaler_scale[idx] != 0.0 else 1.0
            z = (raw_val - self.scaler_mean[idx]) / scale
            contrib = self.weights[idx] * z
            logit += contrib
            attributions.append(FeatureAttribution(feature=name, value=raw_val, contribution=contrib))

        attributions.sort(key=lambda a: abs(a.contribution), reverse=True)
        return logit, attributions

    def predict(
        self, feature_vector: FeatureVector
    ) -> Tuple[float, float, List[FeatureAttribution]]:
        """Return (calibrated_probability, raw_probability, top_attributions)."""
        logit, attributions = self._raw_logit_and_attributions(feature_vector)
        raw_prob = _sigmoid(logit)
        cal_logit = self.platt_a * logit + self.platt_b
        cal_prob = _sigmoid(cal_logit)
        return cal_prob, raw_prob, attributions[:5]

    def to_artifact_dict(self) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "artifact_format": "rtrp.model.v1",
            "model_version": self.model_version,
            "feature_version": self.feature_version,
            "feature_names": list(self.feature_names),
            "scaler_mean": [float(x) for x in self.scaler_mean],
            "scaler_scale": [float(x) for x in self.scaler_scale],
            "weights": [float(x) for x in self.weights],
            "intercept": float(self.intercept),
            "calibrator": {
                "method": "platt_sigmoid",
                "platt_a": float(self.platt_a),
                "platt_b": float(self.platt_b),
            },
            "training_metadata": self.training_metadata,
        }
        digest = _compute_artifact_hash(body)
        body["artifact_sha256"] = digest
        return body

    def save(self, path: Union[str, Path]) -> str:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_artifact_dict()
        out_path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
        return str(out_path)

    @classmethod
    def load(
        cls, path: Union[str, Path], expected_feature_version: str = FEATURE_VERSION
    ) -> "CalibratedLinearRiskModel":
        art_path = Path(path)
        if not art_path.exists():
            raise ArtifactIntegrityError(f"Model artifact not found: {art_path}")

        try:
            payload = json.loads(art_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ArtifactIntegrityError(f"Invalid JSON in model artifact '{art_path}': {exc}") from exc

        if not isinstance(payload, dict) or payload.get("artifact_format") != "rtrp.model.v1":
            raise ArtifactIntegrityError(
                f"Unsupported or missing artifact_format in '{art_path}'"
            )

        stored_hash = payload.get("artifact_sha256")
        if not isinstance(stored_hash, str) or not stored_hash:
            raise ArtifactIntegrityError(f"Missing artifact_sha256 in '{art_path}'")

        body = {k: v for k, v in payload.items() if k != "artifact_sha256"}
        computed_hash = _compute_artifact_hash(body)
        if computed_hash != stored_hash:
            raise ArtifactIntegrityError(
                f"Artifact SHA-256 mismatch for '{art_path}': expected {stored_hash[:12]}, "
                f"computed {computed_hash[:12]}"
            )

        feat_ver = str(payload["feature_version"])
        if feat_ver != expected_feature_version:
            raise ArtifactIntegrityError(
                f"Incompatible model feature_version '{feat_ver}' (expected '{expected_feature_version}')"
            )

        feature_names = tuple(str(x) for x in payload["feature_names"])
        if feature_names != FEATURE_NAMES:
            raise ArtifactIntegrityError(
                f"Feature schema mismatch: artifact has {feature_names}, runtime expects {FEATURE_NAMES}"
            )

        cal = payload.get("calibrator", {})
        return cls(
            model_version=str(payload["model_version"]),
            feature_version=feat_ver,
            feature_names=feature_names,
            scaler_mean=tuple(float(x) for x in payload["scaler_mean"]),
            scaler_scale=tuple(float(x) for x in payload["scaler_scale"]),
            weights=tuple(float(x) for x in payload["weights"]),
            intercept=float(payload["intercept"]),
            platt_a=float(cal["platt_a"]),
            platt_b=float(cal["platt_b"]),
            training_metadata=dict(payload.get("training_metadata", {})),
            artifact_sha256=stored_hash,
        )

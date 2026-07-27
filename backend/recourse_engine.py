"""
Brew Recovery Engine — Prescriptive / Actionable Recourse for the Coffee Predictor
------------------------------------------------------------------------------------
The classifier answers "what will this cup taste like?". This module answers the
more useful question: "what is the SMALLEST change I can make to fix it?"

Framing (standard in the algorithmic-recourse literature — Wachter et al. 2017,
Ustun et al. 2019):

    Given a starting brew x0 that the model predicts as Sour or Bitter, and a
    split of features into:
        - FIXED features   (the user genuinely cannot change them right now,
                             e.g. roast_level — that's the bag they already own)
        - MUTABLE features (grind_size, brew_time, water_temp, water_ratio —
                             things a dial/kettle/scale can actually change)

    find the mutable-feature vector x* that:
        minimizes   weighted_distance(x*, x0)
        subject to  P_model(target_class | x_fixed, x*) >= confidence

Why differential evolution instead of a gradient method
---------------------------------------------------------
GradientBoostingClassifier's probability surface is built from axis-aligned
tree splits — it's piecewise CONSTANT. Its gradient is exactly zero almost
everywhere and undefined at the splits themselves, so SLSQP/Newton-style
solvers will immediately report "no improvement possible" even one step from
a decision boundary. Differential evolution is derivative-free and treats the
classifier purely as a black box, which is exactly what real deployed models
require (you often don't get to see inside them at all).

Why the distance is normalized, not raw Euclidean
---------------------------------------------------------
Grind size lives on a ~0-1200 micron scale, brew time on a ~0-300 second
scale. Unweighted Euclidean distance across mismatched units would let a
900-micron grind change and a 3-second time change score as "equally cheap,"
which is physically absurd. Every mutable feature's delta is divided by its
observed standard deviation before being squared, which is the same idea
behind Mahalanobis distance under a diagonal covariance assumption — it puts
every feature's cost on a comparable, unitless footing.
"""

import numpy as np
import pandas as pd
import joblib
from scipy.optimize import differential_evolution


class BrewRecoveryEngine:
    def __init__(self, model_path="model.pkl"):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.encoder = bundle["encoder"]
        self.feature_names = bundle["feature_names"]
        self.bounds_dict = bundle["feature_bounds"]  # {feature: (min, max, std)}
        self.classes_ = list(self.model.classes_)

    def _vector_from_dict(self, cup: dict) -> np.ndarray:
        cup = dict(cup)
        cup["roast_level_encoded"] = self.encoder.transform([cup["roast_level"]])[0]
        return np.array([cup[f] for f in self.feature_names], dtype=float)

    def _predict_proba(self, x: np.ndarray) -> np.ndarray:
        row = pd.DataFrame([x], columns=self.feature_names)
        return self.model.predict_proba(row)[0]

    def diagnose(self, cup: dict) -> dict:
        """What does the model currently think of this cup, and how sure is it?"""
        x0 = self._vector_from_dict(cup)
        proba = self._predict_proba(x0)
        return {cls: round(float(p), 3) for cls, p in zip(self.classes_, proba)}

    def recommend_fix(
        self,
        cup: dict,
        mutable_features: list,
        target_class: str = "Balanced",
        confidence: float = 0.60,
        max_iter: int = 300,
    ) -> dict:
        x0 = self._vector_from_dict(cup)
        target_idx = self.classes_.index(target_class)

        mutable_idx = [self.feature_names.index(f) for f in mutable_features]

        # per-dimension cost weight = 1 / std, so a change is scored in
        # "how many standard deviations did I move this knob" units
        weights = np.array(
            [1.0 / (self.bounds_dict[self.feature_names[i]][2] or 1.0) for i in mutable_idx]
        )
        bounds = [self.bounds_dict[self.feature_names[i]][:2] for i in mutable_idx]

        current_proba = self._predict_proba(x0)
        current_pred = self.classes_[int(np.argmax(current_proba))]

        def objective(z):
            x = x0.copy()
            x[mutable_idx] = z
            proba = self._predict_proba(x)
            cost = float(np.sum((weights * (z - x0[mutable_idx])) ** 2))
            shortfall = max(0.0, confidence - proba[target_idx])
            # Large penalty for not clearing the confidence bar at all; once
            # cleared, the optimizer is free to keep minimizing pure distance
            # rather than overshooting to some needlessly extreme fix.
            return cost + 5000.0 * shortfall**2

        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=max_iter,
            polish=True,
            tol=1e-7,
            mutation=(0.4, 1.6),
            recombination=0.8,
        )

        x_star = x0.copy()
        x_star[mutable_idx] = result.x
        final_proba = self._predict_proba(x_star)
        achieved = bool(final_proba[target_idx] >= confidence)

        changes = {}
        for k, i in enumerate(mutable_idx):
            fname = self.feature_names[i]
            changes[fname] = {
                "from": round(float(x0[i]), 2),
                "to": round(float(result.x[k]), 2),
                "delta": round(float(result.x[k] - x0[i]), 2),
            }

        return {
            "starting_prediction": current_pred,
            "starting_confidence_in_target": round(float(current_proba[target_idx]), 3),
            "target_class": target_class,
            "confidence_requested": confidence,
            "confidence_achieved": achieved,
            "achieved_probability": round(float(final_proba[target_idx]), 3),
            "recommended_changes": changes,
            "full_probabilities_after_fix": {
                cls: round(float(p), 3) for cls, p in zip(self.classes_, final_proba)
            },
        }


if __name__ == "__main__":
    engine = BrewRecoveryEngine("model.pkl")

    print("=" * 70)
    print("CASE 1: Sour cup, light roast — grinder and clock are the only")
    print("adjustable knobs (roast is fixed, ratio is fixed).")
    print("=" * 70)
    bad_cup_1 = {
        "roast_level": "Light",
        "grind_size_microns": 1100.0,   # very coarse -> under-extracts
        "water_temp_c": 92.0,
        "brew_time_seconds": 45.0,       # short
        "water_ratio": 16.0,
    }
    print("Diagnosis:", engine.diagnose(bad_cup_1))
    fix = engine.recommend_fix(
        bad_cup_1,
        mutable_features=["grind_size_microns", "brew_time_seconds"],
        target_class="Balanced",
        confidence=0.60,
    )
    for k, v in fix.items():
        print(f"  {k}: {v}")

    print()
    print("=" * 70)
    print("CASE 2: Bitter cup, dark roast — user can ONLY change grind size")
    print("(fixed brew time because they use a pre-set espresso machine).")
    print("=" * 70)
    bad_cup_2 = {
        "roast_level": "Dark",
        "grind_size_microns": 350.0,   # very fine -> over-extracts
        "water_temp_c": 96.0,
        "brew_time_seconds": 240.0,
        "water_ratio": 15.0,
    }
    print("Diagnosis:", engine.diagnose(bad_cup_2))
    fix2 = engine.recommend_fix(
        bad_cup_2,
        mutable_features=["grind_size_microns"],
        target_class="Balanced",
        confidence=0.60,
    )
    for k, v in fix2.items():
        print(f"  {k}: {v}")

    print()
    print("=" * 70)
    print("CASE 3: Same bitter cup, but demand 90% confidence with only")
    print("grind size adjustable — testing whether a single knob is enough.")
    print("=" * 70)
    fix3 = engine.recommend_fix(
        bad_cup_2,
        mutable_features=["grind_size_microns"],
        target_class="Balanced",
        confidence=0.90,
    )
    for k, v in fix3.items():
        print(f"  {k}: {v}")

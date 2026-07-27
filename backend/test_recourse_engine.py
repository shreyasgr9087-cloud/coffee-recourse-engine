import pytest
from recourse_engine import BrewRecoveryEngine

@pytest.fixture
def engine():
    # Load the trained model once for all tests
    return BrewRecoveryEngine("model.pkl")

def test_probabilities_sum_to_one(engine):
    cup = {"roast_level": "Medium", "grind_size_microns": 800.0, "water_temp_c": 93.0, "brew_time_seconds": 120.0, "water_ratio": 16.0}
    diagnosis = engine.diagnose(cup)
    # The probabilities of Sour + Balanced + Bitter should equal 1.0
    assert pytest.approx(sum(diagnosis.values()), 0.01) == 1.0

def test_diagnose_obvious_sour(engine):
    # A very coarse grind and very short brew time on a light roast should be extremely sour
    cup = {"roast_level": "Light", "grind_size_microns": 1200.0, "water_temp_c": 85.0, "brew_time_seconds": 30.0, "water_ratio": 15.0}
    diagnosis = engine.diagnose(cup)
    assert max(diagnosis, key=diagnosis.get) == "Sour / Under-extracted"

def test_recommended_fix_respects_bounds(engine):
    cup = {"roast_level": "Light", "grind_size_microns": 1200.0, "water_temp_c": 85.0, "brew_time_seconds": 30.0, "water_ratio": 15.0}
    fix = engine.recommend_fix(cup, mutable_features=["grind_size_microns"], target_class="Balanced", confidence=0.60)
    
    # Extract the AI's suggested grind size
    new_grind = fix["recommended_changes"]["grind_size_microns"]["to"]
    min_bound = engine.bounds_dict["grind_size_microns"][0]
    max_bound = engine.bounds_dict["grind_size_microns"][1]
    
    # Check if the AI's suggestion is physically possible
    assert min_bound <= new_grind <= max_bound

def test_fixed_features_remain_untouched(engine):
    cup = {"roast_level": "Light", "grind_size_microns": 1200.0, "water_temp_c": 85.0, "brew_time_seconds": 30.0, "water_ratio": 15.0}
    # We only allow it to change grind size
    fix = engine.recommend_fix(cup, mutable_features=["grind_size_microns"], target_class="Balanced", confidence=0.60)
    
    # It should not suggest a change for brew_time_seconds
    assert "brew_time_seconds" not in fix["recommended_changes"]

def test_infeasible_target_reports_honestly(engine):
    # A horrifically bitter cup (dark roast, fine grind, boiling water, 5 minutes)
    cup = {"roast_level": "Dark", "grind_size_microns": 300.0, "water_temp_c": 100.0, "brew_time_seconds": 300.0, "water_ratio": 14.0}
    
    # Demand 99.9% confidence using ONLY water ratio (which has very little impact). 
    # The AI should fail to reach 99.9% and admit it.
    fix = engine.recommend_fix(cup, mutable_features=["water_ratio"], target_class="Balanced", confidence=0.999)
    assert fix["confidence_achieved"] is False

def test_diagnose_obvious_bitter(engine):
    # Dark roast + extremely fine grind + near-boiling water + long extraction
    # This is the textbook over-extraction scenario — the model must classify it as Bitter.
    cup = {"roast_level": "Dark", "grind_size_microns": 300.0, "water_temp_c": 99.0, "brew_time_seconds": 290.0, "water_ratio": 14.5}
    diagnosis = engine.diagnose(cup)
    assert max(diagnosis, key=diagnosis.get) == "Bitter / Over-extracted"

def test_already_balanced_trivial_case(engine):
    # This specific brew is empirically confirmed to classify as Balanced (~80% probability).
    cup = {"roast_level": "Medium", "grind_size_microns": 800.0, "water_temp_c": 90.0, "brew_time_seconds": 100.0, "water_ratio": 16.0}
    diagnosis = engine.diagnose(cup)
    assert max(diagnosis, key=diagnosis.get) == "Balanced", (
        f"Pre-condition failed: expected Balanced, got {max(diagnosis, key=diagnosis.get)}. "
        f"Retrain and re-check if the model changed."
    )

    # Ask the recourse engine to reach "Balanced" — it already is.
    # The optimizer should converge to near-zero deltas across all mutable features.
    fix = engine.recommend_fix(
        cup,
        mutable_features=["grind_size_microns", "brew_time_seconds", "water_temp_c", "water_ratio"],
        target_class="Balanced",
        confidence=0.60,
    )

    assert fix["confidence_achieved"] is True

    # Every recommended delta should be negligibly small (< 5% of the feature's range)
    for fname, change in fix["recommended_changes"].items():
        feature_range = engine.bounds_dict[fname][1] - engine.bounds_dict[fname][0]
        assert abs(change["delta"]) < 0.05 * feature_range, (
            f"Recourse engine proposed a non-trivial change to {fname} "
            f"(delta={change['delta']}) for an already-balanced cup."
        )
import numpy as np
import pandas as pd


def default_parameters():
    """Original exploratory scenario."""
    return pd.DataFrame({
        "factor": [
            "Awareness",
            "Accessibility",
            "Health attitudes",
            "Trust",
            "Social support",
            "Functional status",
            "Socioeconomic conditions",
            "Previous healthcare experience",
        ],
        "minimum": [0.30, 0.25, 0.35, 0.25, 0.30, 0.30, 0.25, 0.30],
        "maximum": [0.80, 0.85, 0.90, 0.85, 0.80, 0.85, 0.80, 0.85],
        "weight": [1.00, 1.20, 1.20, 1.20, 1.00, 1.00, 1.00, 0.90],
    })


def run_simulation(parameters, iterations=10000, threshold=0.60, seed=42):
    """Independent uniform inputs and a weighted participation score."""
    parameters = parameters.copy().reset_index(drop=True)

    if len(parameters) != 8:
        raise ValueError("Exactly eight factors are required.")

    names = parameters["factor"]
    if names.isna().any() or names.astype(str).str.strip().eq("").any():
        raise ValueError("Each factor must have a name.")
    if names.duplicated().any():
        raise ValueError("Factor names must be unique.")
    if names.isin(["participation_score", "threshold_reached"]).any():
        raise ValueError("Please use different factor names.")

    values = parameters[["minimum", "maximum", "weight"]].to_numpy(
        dtype=float
    )
    if not np.isfinite(values).all():
        raise ValueError("All parameters must be finite numbers.")

    low, high, weights = values.T

    if ((low < 0) | (high > 1) | (low > high)).any():
        raise ValueError("Each range must satisfy 0 <= minimum <= maximum <= 1.")
    if (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Weights must be nonnegative, with a positive total.")
    if not isinstance(iterations, (int, np.integer)) or not 1 <= iterations <= 1000000:
        raise ValueError("Iterations must be an integer from 1 to 1,000,000.")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("The threshold must be between 0 and 1.")
    if not isinstance(seed, (int, np.integer)) or not 0 <= seed <= 4294967295:
        raise ValueError("The seed must be an integer from 0 to 4,294,967,295.")

    # Preserve the original script's random-number sequence.
    rng = np.random.RandomState(seed)
    samples = np.column_stack([
        rng.uniform(a, b, iterations)
        for a, b in zip(low, high)
    ])

    normalized_weights = weights / weights.sum()
    scores = (samples * normalized_weights).sum(axis=1)
    reached = scores >= threshold

    simulation = pd.DataFrame(samples, columns=names.tolist())
    simulation["participation_score"] = scores
    simulation["threshold_reached"] = reached.astype(int)

    summary = {
        "iterations": int(iterations),
        "seed": int(seed),
        "threshold": float(threshold),
        "mean": float(scores.mean()),
        "median": float(np.median(scores)),
        "q1": float(np.quantile(scores, 0.25)),
        "q3": float(np.quantile(scores, 0.75)),
        "threshold_reached_share": float(reached.mean()),
    }

    # Equivalent to replacing one factor by its minimum/maximum
    # while retaining the other simulated factor values.
    sample_means = samples.mean(axis=0)
    tornado = pd.DataFrame({
        "factor": names.tolist(),
        "change_at_minimum": normalized_weights * (low - sample_means),
        "change_at_maximum": normalized_weights * (high - sample_means),
        "total_range": normalized_weights * (high - low),
    }).sort_values("total_range", ascending=False, kind="stable")

    return simulation, summary, tornado.reset_index(drop=True)

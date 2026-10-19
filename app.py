import io
import json
import zipfile

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
import streamlit as st

from model import default_parameters, run_simulation


st.set_page_config(
    page_title="Preventive Participation Framework",
    page_icon="📊",
    layout="wide",
)

st.title("Preventive Healthcare Participation")
st.subheader("An exploratory Monte Carlo framework")
st.caption("Peter Domjan | Exploratory scenario modelling")

st.info(
    "Explore how assumed factor ranges and weights influence a simulated "
    "participation score. The default settings are illustrative assumptions. "
    "The score is not an empirically calibrated participation probability."
)

with st.expander("Model assumptions and interpretation"):
    st.markdown(
        """
        - Eight factors are sampled independently from uniform distributions.
        - Higher factor values represent more favourable conditions.
        - The score is the weighted average of the eight sampled values.
        - Weights are automatically normalised by their sum.
        - The threshold classifies simulated scenarios.
        - Threshold attainment is not an observed population participation rate.
        - Tornado bars show changes in the mean score when one factor is
          fixed at its minimum or maximum, retaining the other simulated values.
        """
    )
    st.latex(r"S = \frac{\sum_{i=1}^{8} w_i x_i}{\sum_{i=1}^{8} w_i}")

# Changing this key restores all input widgets to their defaults.
if "input_version" not in st.session_state:
    st.session_state.input_version = 0

if st.button("Restore default scenario"):
    st.session_state.input_version += 1
    st.session_state.pop("result", None)
    st.rerun()

version = st.session_state.input_version

with st.form(f"settings_{version}"):
    st.subheader("1. Factor ranges and weights")

    parameters = st.data_editor(
        default_parameters(),
        hide_index=True,
        disabled=["factor"],
        num_rows="fixed",
        column_config={
            "factor": st.column_config.TextColumn("Factor", width="large"),
            "minimum": st.column_config.NumberColumn(
                "Minimum", min_value=0.0, max_value=1.0,
                step=0.01, format="%.2f", required=True,
            ),
            "maximum": st.column_config.NumberColumn(
                "Maximum", min_value=0.0, max_value=1.0,
                step=0.01, format="%.2f", required=True,
            ),
            "weight": st.column_config.NumberColumn(
                "Weight", min_value=0.0,
                step=0.05, format="%.2f", required=True,
            ),
        },
        key=f"parameters_{version}",
    )

    st.subheader("2. Simulation settings")
    c1, c2, c3 = st.columns(3)

    with c1:
        iterations = st.number_input(
            "Iterations", min_value=100, max_value=100000,
            value=10000, step=100,
            key=f"iterations_{version}",
        )
    with c2:
        threshold = st.number_input(
            "Score threshold", min_value=0.0, max_value=1.0,
            value=0.60, step=0.01, format="%.2f",
            key=f"threshold_{version}",
        )
    with c3:
        seed = st.number_input(
            "Random seed", min_value=0, max_value=4294967295,
            value=42, step=1,
            key=f"seed_{version}",
        )

    submitted = st.form_submit_button(
        "Run simulation", type="primary"
    )

if submitted:
    try:
        with st.spinner("Running simulation..."):
            simulation, summary, tornado = run_simulation(
                parameters,
                iterations=int(iterations),
                threshold=float(threshold),
                seed=int(seed),
            )
            st.session_state.result = {
                "parameters": parameters.copy(),
                "simulation": simulation,
                "summary": summary,
                "tornado": tornado,
            }
    except (ValueError, TypeError, OverflowError) as error:
        st.error(str(error))
        st.stop()

if "result" not in st.session_state:
    st.caption("Set your parameters and select Run simulation.")
    st.stop()

result = st.session_state.result
simulation = result["simulation"]
summary = result["summary"]
tornado = result["tornado"]

st.divider()
st.subheader("3. Results")
st.caption(
    f"Last completed run: {summary['iterations']:,} iterations | "
    f"seed {summary['seed']} | threshold {summary['threshold']:.2f}. "
    "Results and downloads reflect this completed run. "
    "After editing inputs, select Run simulation to update them."
)

c1, c2, c3 = st.columns(3)
c1.metric("Median score", f"{summary['median']:.3f}")
c2.metric(
    "Interquartile range",
    f"{summary['q1']:.3f}–{summary['q3']:.3f}",
)
c3.metric(
    "Scenarios reaching threshold",
    f"{summary['threshold_reached_share']:.1%}",
)

# Histogram
histogram, ax = plt.subplots(figsize=(9, 5))
ax.hist(
    simulation["participation_score"],
    bins=40, color="#287D8E",
    edgecolor="white", linewidth=0.5,
)
ax.axvline(
    summary["threshold"], color="#C44E52",
    linestyle="--", linewidth=2,
    label=f"Threshold = {summary['threshold']:.2f}",
)
ax.axvline(
    summary["median"], color="#26364A",
    linewidth=2, label=f"Median = {summary['median']:.3f}",
)
ax.set(
    xlabel="Simulated participation score",
    ylabel="Number of scenarios",
    title="Distribution of simulated participation scores",
)
ax.legend()
ax.grid(axis="y", alpha=0.2)
histogram.tight_layout()

# Tornado diagram
tornado_figure, ax = plt.subplots(figsize=(10, 6))
positions = range(len(tornado))
ax.barh(
    positions, tornado["change_at_minimum"],
    color="#D98B55", label="Factor fixed at minimum",
)
ax.barh(
    positions, tornado["change_at_maximum"],
    color="#287D8E", label="Factor fixed at maximum",
)
ax.set_yticks(list(positions))
ax.set_yticklabels(tornado["factor"])
ax.invert_yaxis()
ax.axvline(0, color="#26364A", linewidth=1)
ax.set(
    xlabel="Change in mean score (percentage points)",
    title="One-way sensitivity analysis",
)
ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
ax.grid(axis="x", alpha=0.2)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
tornado_figure.tight_layout()

tab1, tab2, tab3 = st.tabs([
    "Score distribution", "Sensitivity analysis", "Run parameters"
])
with tab1:
    st.pyplot(histogram)
with tab2:
    st.pyplot(tornado_figure)
    st.caption(
        "Ranking depends on the assumed weights and ranges; "
        "it does not establish the real-world importance of each factor. "
        "A 1% tick interval represents one percentage point of score change."
    )
with tab3:
    st.dataframe(result["parameters"], hide_index=True)

# Export the completed run, including both figures.
archive = io.BytesIO()
with zipfile.ZipFile(
    archive, "w", compression=zipfile.ZIP_DEFLATED
) as bundle:
    tables = {
        "parameters.csv": result["parameters"],
        "simulation.csv": simulation,
        "summary.csv": pd.DataFrame([summary]),
        "tornado.csv": tornado,
    }
    for filename, table in tables.items():
        bundle.writestr(
            filename,
            table.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
        )

    settings = {
        "model": "Independent uniform inputs; weighted average score",
        "iterations": summary["iterations"],
        "seed": summary["seed"],
        "threshold": summary["threshold"],
        "parameters": result["parameters"].to_dict(orient="records"),
    }
    bundle.writestr(
        "settings.json", json.dumps(settings, indent=2)
    )

    for filename, figure in [
        ("histogram.png", histogram),
        ("tornado.png", tornado_figure),
    ]:
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
        bundle.writestr(filename, buffer.getvalue())

plt.close(histogram)
plt.close(tornado_figure)

st.download_button(
    "Download results, parameters and figures (ZIP)",
    data=archive.getvalue(),
    file_name="participation_framework_results.zip",
    mime="application/zip",
)

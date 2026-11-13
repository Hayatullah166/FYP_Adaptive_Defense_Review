import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# =========================================================
# SCRIPT 21: LATENCY AND EFFICIENCY ANALYSIS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

GROUPED_DIR = os.path.join(
    BASE_DIR, "data", "processed", "grouped"
)

REPORT_DIR = os.path.join(
    BASE_DIR, "data", "reports"
)

RAW_VAL_PATH = os.path.join(
    GROUPED_DIR, "validation_lgbm_grouped.csv"
)

SCALED_VAL_PATH = os.path.join(
    GROUPED_DIR, "validation_mlp_scaled_grouped.csv"
)

FEATURE_NAMES_PATH = os.path.join(
    GROUPED_DIR, "models", "feature_names_grouped.joblib"
)

RISK_MODEL_PATH = os.path.join(
    BASE_DIR, "models", "risk_assessment",
    "xgboost_risk_model.joblib"
)

RISK_FEATURES_PATH = os.path.join(
    BASE_DIR, "models", "risk_assessment",
    "risk_model_feature_names.joblib"
)

MLP_CHECKPOINT_PATH = os.path.join(
    BASE_DIR, "models", "mlp",
    "mlp_adversarial_best.pth"
)

OUTPUT_CSV = os.path.join(
    REPORT_DIR, "e3_latency_efficiency_results.csv"
)

OUTPUT_JSON = os.path.join(
    REPORT_DIR, "e3_latency_efficiency_summary.json"
)

DEVICE = torch.device("cpu")

# Provisional validation-selected threshold
RISK_THRESHOLD = 0.01

MLP_THRESHOLD = 0.9073078036308289
SQUEEZE_DECIMALS = 2

WARMUP_RUNS = 5
MEASURED_RUNS = 30


# =========================================================
# MLP ARCHITECTURE
# =========================================================

class MLP(nn.Module):

    def __init__(self, input_features):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)


# =========================================================
# BENCHMARK HELPER
# =========================================================

def benchmark(function, warmups=5, repeats=30):

    for _ in range(warmups):
        function()

    timings = []

    for _ in range(repeats):
        start = time.perf_counter()
        function()
        elapsed = time.perf_counter() - start
        timings.append(elapsed)

    timings = np.asarray(timings, dtype=np.float64)

    return {
        "mean_seconds": float(np.mean(timings)),
        "median_seconds": float(np.median(timings)),
        "std_seconds": float(np.std(timings)),
        "min_seconds": float(np.min(timings)),
        "max_seconds": float(np.max(timings))
    }


def add_latency_metrics(result, records):

    mean_seconds = result["mean_seconds"]

    result["mean_milliseconds"] = mean_seconds * 1000

    if records > 0:
        result["amortized_ms_per_record"] = (
            mean_seconds / records * 1000
        )

        result["records_per_second"] = (
            records / mean_seconds
            if mean_seconds > 0 else None
        )
    else:
        result["amortized_ms_per_record"] = None
        result["records_per_second"] = None

    return result


# =========================================================
# MAIN
# =========================================================

def main():

    os.makedirs(REPORT_DIR, exist_ok=True)

    print("=" * 75)
    print("SCRIPT 21 - E3 LATENCY AND EFFICIENCY ANALYSIS")
    print("=" * 75)

    # -----------------------------------------------------
    # 1. Load and validate data
    # -----------------------------------------------------

    print("\n[1/9] Loading validation data...")

    required_paths = [
        RAW_VAL_PATH,
        SCALED_VAL_PATH,
        FEATURE_NAMES_PATH,
        RISK_MODEL_PATH,
        RISK_FEATURES_PATH,
        MLP_CHECKPOINT_PATH
    ]

    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required file missing: {path}"
            )

    raw_df = pd.read_csv(RAW_VAL_PATH)
    scaled_df = pd.read_csv(SCALED_VAL_PATH)

    feature_cols = list(
        joblib.load(FEATURE_NAMES_PATH)
    )

    risk_feature_cols = list(
        joblib.load(RISK_FEATURES_PATH)
    )

    if feature_cols != risk_feature_cols:
        raise ValueError(
            "Grouped and XGBoost feature names do not match."
        )

    if len(raw_df) != len(scaled_df):
        raise ValueError(
            "Raw and scaled validation rows do not match."
        )

    if not np.array_equal(
        raw_df["Label"].to_numpy(),
        scaled_df["Label"].to_numpy()
    ):
        raise ValueError(
            "Raw and scaled labels are not aligned."
        )

    if not raw_df["AttackType"].astype(str).equals(
        scaled_df["AttackType"].astype(str)
    ):
        raise ValueError(
            "Raw and scaled AttackType rows are not aligned."
        )

    X_raw = raw_df[feature_cols].to_numpy(
        dtype=np.float32
    )

    X_scaled = scaled_df[feature_cols].to_numpy(
        dtype=np.float32
    )

    y = raw_df["Label"].to_numpy(dtype=np.int64)

    if not np.isfinite(X_raw).all():
        raise ValueError("Invalid raw feature values.")

    if not np.isfinite(X_scaled).all():
        raise ValueError("Invalid scaled feature values.")

    total_records = len(y)

    print("Validation records:", total_records)
    print("Features:", len(feature_cols))
    print("Malicious:", int((y == 1).sum()))
    print("Benign:", int((y == 0).sum()))

    # -----------------------------------------------------
    # 2. Load models
    # -----------------------------------------------------

    print("\n[2/9] Loading models...")

    risk_model = joblib.load(RISK_MODEL_PATH)

    try:
        checkpoint = torch.load(
            MLP_CHECKPOINT_PATH,
            map_location=DEVICE,
            weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(
            MLP_CHECKPOINT_PATH,
            map_location=DEVICE
        )

    mlp_features = list(checkpoint["feature_names"])

    if mlp_features != feature_cols:
        raise ValueError(
            "MLP and XGBoost feature order does not match."
        )

    mlp_model = MLP(len(mlp_features)).to(DEVICE)

    mlp_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    mlp_model.eval()

    print("XGBoost loaded.")
    print("MLP loaded.")
    print("Device:", DEVICE)

    # -----------------------------------------------------
    # 3. Prepare fixed input data
    # -----------------------------------------------------

    print("\n[3/9] Preparing benchmark inputs...")

    raw_frame = pd.DataFrame(
        X_raw,
        columns=feature_cols
    )

    # -----------------------------------------------------
    # 4. Warm-up and initial routing
    # -----------------------------------------------------

    print("\n[4/9] Establishing routing distribution...")

    with torch.no_grad():

        xgb_prob = risk_model.predict_proba(
            raw_frame
        )[:, 1]

        xgb_class = (
            xgb_prob >= 0.5
        ).astype(np.int64)

        uncertainty = np.minimum(
            xgb_prob,
            1.0 - xgb_prob
        )

        fast_mask = (
            (xgb_class == 0) &
            (uncertainty < RISK_THRESHOLD)
        )

        deep_mask = ~fast_mask

        deep_indices = np.flatnonzero(deep_mask)

        X_deep = X_scaled[deep_indices]

        X_deep_squeezed = np.round(
            X_deep,
            decimals=SQUEEZE_DECIMALS
        ).astype(np.float32)

        deep_tensor = torch.tensor(
            X_deep_squeezed,
            dtype=torch.float32,
            device=DEVICE
        )

    fast_count = int(fast_mask.sum())
    deep_count = int(deep_mask.sum())

    print("Threshold:", RISK_THRESHOLD)
    print("Fast records:", fast_count)
    print("Deep records:", deep_count)

    # -----------------------------------------------------
    # 5. Benchmark XGBoost
    # -----------------------------------------------------

    print("\n[5/9] Benchmarking XGBoost...")

    def run_xgboost():

        return risk_model.predict_proba(
            raw_frame
        )[:, 1]

    xgb_result = benchmark(
        run_xgboost,
        WARMUP_RUNS,
        MEASURED_RUNS
    )

    xgb_result = add_latency_metrics(
        xgb_result,
        total_records
    )

    # -----------------------------------------------------
    # 6. Benchmark feature squeezing
    # -----------------------------------------------------

    print("\n[6/9] Benchmarking feature squeezing...")

    def run_squeezing():

        return np.round(
            X_deep,
            decimals=SQUEEZE_DECIMALS
        ).astype(np.float32)

    squeezing_result = benchmark(
        run_squeezing,
        WARMUP_RUNS,
        MEASURED_RUNS
    )

    squeezing_result = add_latency_metrics(
        squeezing_result,
        deep_count
    )

    # -----------------------------------------------------
    # 7. Benchmark Deep MLP
    # -----------------------------------------------------

    print("\n[7/9] Benchmarking Deep MLP...")

    def run_mlp():

        with torch.no_grad():

            logits = mlp_model(deep_tensor)

            return torch.sigmoid(
                logits
            ).cpu().numpy().reshape(-1)

    mlp_result = benchmark(
        run_mlp,
        WARMUP_RUNS,
        MEASURED_RUNS
    )

    mlp_result = add_latency_metrics(
        mlp_result,
        deep_count
    )

    # -----------------------------------------------------
    # 8. Benchmark complete adaptive pipeline
    # -----------------------------------------------------

    print("\n[8/9] Benchmarking complete adaptive pipeline...")

    def run_complete_pipeline():

        probabilities = risk_model.predict_proba(
            raw_frame
        )[:, 1]

        predicted = (
            probabilities >= 0.5
        ).astype(np.int64)

        risk = np.minimum(
            probabilities,
            1.0 - probabilities
        )

        current_fast_mask = (
            (predicted == 0) &
            (risk < RISK_THRESHOLD)
        )

        current_deep_mask = ~current_fast_mask

        current_indices = np.flatnonzero(
            current_deep_mask
        )

        if len(current_indices) > 0:

            current_deep = X_scaled[current_indices]

            squeezed = np.round(
                current_deep,
                decimals=SQUEEZE_DECIMALS
            ).astype(np.float32)

            tensor = torch.tensor(
                squeezed,
                dtype=torch.float32,
                device=DEVICE
            )

            with torch.no_grad():

                logits = mlp_model(tensor)

                deep_predictions = (
                    torch.sigmoid(logits)
                    .cpu()
                    .numpy()
                    .reshape(-1)
                    >= MLP_THRESHOLD
                ).astype(np.int64)

        return (
            current_fast_mask,
            current_deep_mask
        )

    complete_result = benchmark(
        run_complete_pipeline,
        WARMUP_RUNS,
        MEASURED_RUNS
    )

    complete_result = add_latency_metrics(
        complete_result,
        total_records
    )

    # -----------------------------------------------------
    # 9. CPU and memory observations
    # -----------------------------------------------------

    print("\n[9/9] Collecting resource observations...")

    resource_info = {
        "psutil_available": False,
        "rss_before_benchmark_mb": None,
        "rss_after_benchmark_mb": None,
        "process_cpu_seconds_during_benchmark": None
    }

    try:

        import psutil

        process = psutil.Process(os.getpid())

        resource_info["psutil_available"] = True

        # Current RSS after model/data loading and benchmarking.
        resource_info["rss_after_benchmark_mb"] = (
            process.memory_info().rss / (1024 ** 2)
        )

    except ImportError:

        print(
            "psutil not installed; memory measurement skipped."
        )

    # -----------------------------------------------------
    # Summary and report
    # -----------------------------------------------------

    # Batch-amortized path estimates:
    # XGBoost scoring is shared across the full validation batch.
    xgb_per_record = (
        xgb_result["mean_seconds"] / total_records
        if total_records else 0.0
    )

    fast_path_amortized_ms = xgb_per_record * 1000

    deep_path_amortized_ms = (
        xgb_per_record
        + squeezing_result["mean_seconds"] / deep_count
        + mlp_result["mean_seconds"] / deep_count
    ) * 1000 if deep_count else None

    results = [

        {
            "component": "XGBoost risk engine",
            **xgb_result
        },

        {
            "component": "Feature squeezing",
            **squeezing_result
        },

        {
            "component": "Deep MLP",
            **mlp_result
        },

        {
            "component": "Complete adaptive pipeline",
            **complete_result
        }
    ]

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    summary = {

        "experiment": "E3 Latency and Efficiency Analysis",

        "dataset": "CICIDS-2018 grouped validation",

        "validation_only": True,

        "independent_test_evaluation": False,

        "packet_level_latency_measured": False,

        "threshold": RISK_THRESHOLD,

        "threshold_status": "Provisional validation-selected value",

        "records": int(total_records),

        "features": int(len(feature_cols)),

        "fast_records": fast_count,

        "deep_records": deep_count,

        "fast_share_percent": (
            fast_count / total_records * 100
        ),

        "deep_share_percent": (
            deep_count / total_records * 100
        ),

        "warmup_runs": WARMUP_RUNS,

        "measured_runs": MEASURED_RUNS,

        "device": str(DEVICE),

        "path_latency_method": (
            "Batch-amortized estimate. XGBoost scores the full batch; "
            "feature squeezing and MLP operate only on Deep records. "
            "These are not independent single-record request timings."
        ),

        "fast_path_amortized_ms_per_record": (
            fast_path_amortized_ms
        ),

        "deep_path_amortized_ms_per_record": (
            deep_path_amortized_ms
        ),

        "xgboost": xgb_result,

        "feature_squeezing": squeezing_result,

        "deep_mlp": mlp_result,

        "complete_adaptive_pipeline": complete_result,

        "resources": resource_info,

        "limitations": [
            "Offline validation-batch benchmark.",
            "Model loading and dataset loading are excluded.",
            "No live network packet capture or packet processing.",
            "No independent test-set evaluation.",
            "Batch-amortized path estimates are not single-record latency.",
            "CPU and memory observations depend on available system tools."
        ],

        "results_csv": OUTPUT_CSV

    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    # -----------------------------------------------------
    # Print results
    # -----------------------------------------------------

    print("\n" + "=" * 75)
    print("SCRIPT 21 - FINAL BENCHMARK RESULTS")
    print("=" * 75)

    print("\nPATH DISTRIBUTION")

    print("Total records:", total_records)

    print(
        "Fast: {} ({:.4f}%)".format(
            fast_count,
            fast_count / total_records * 100
        )
    )

    print(
        "Deep: {} ({:.4f}%)".format(
            deep_count,
            deep_count / total_records * 100
        )
    )

    print("\nLATENCY RESULTS")

    for result in results:

        print("\n" + result["component"])

        print(
            "Mean batch latency: {:.6f} ms".format(
                result["mean_milliseconds"]
            )
        )

        print(
            "Median batch latency: {:.6f} ms".format(
                result["median_seconds"] * 1000
            )
        )

        print(
            "Std deviation: {:.6f} ms".format(
                result["std_seconds"] * 1000
            )
        )

        print(
            "Amortized ms per record: {}".format(
                result["amortized_ms_per_record"]
            )
        )

        print(
            "Records per second: {}".format(
                result["records_per_second"]
            )
        )

    print("\nBATCH-AMORTIZED PATH ESTIMATES")

    print(
        "Fast path ms/record:",
        round(fast_path_amortized_ms, 8)
    )

    print(
        "Deep path ms/record:",
        round(deep_path_amortized_ms, 8)
        if deep_path_amortized_ms is not None
        else "Not available"
    )

    print("\nRESOURCE OBSERVATIONS")

    print(
        "psutil available:",
        resource_info["psutil_available"]
    )

    print(
        "RSS after benchmark (MB):",
        resource_info["rss_after_benchmark_mb"]
    )

    print("\nSAVED REPORTS")

    print(OUTPUT_CSV)
    print(OUTPUT_JSON)

    print("\nSCRIPT 21 COMPLETED SUCCESSFULLY")
    print("=" * 75)


if __name__ == "__main__":
    main()


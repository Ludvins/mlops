# monitoring/drift_monitor.py
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset, ClassificationPreset
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

# Load reference (training) data and current (simulated production) data
reference_data = pd.read_csv("/app/data/reference.csv")   # the training dataset stats
current_data = pd.read_csv("/app/data/current.csv")       # new data or test set

# Configure Evidently report for data drift (and optionally performance)
metrics = [DataDriftPreset()]
# If we had ground truth for predictions, we could add performance metrics as well (ClassificationPreset for classification)
report = Report(metrics=metrics)
report.run(reference_data=reference_data, current_data=current_data)

# Extract drift results
drift_result = report.as_dict()["metrics"][0]["result"]  # DataDriftPreset is first metric
drift_score = drift_result["dataset_drift"]  # true/false if drift detected
drift_ratio = drift_result["number_of_drifted_columns"] / drift_result["number_of_columns"]

# Prepare Prometheus metrics
registry = CollectorRegistry()
# Gauge for whether dataset drift is detected (0/1)
drift_flag = Gauge("dataset_drift_detected", "Data drift detected (1=yes, 0=no)", registry=registry)
# Gauge for fraction of features drifting
drift_fraction = Gauge("data_drift_fraction", "Fraction of features drifted", registry=registry)
drift_flag.set(1.0 if drift_score else 0.0)
drift_fraction.set(drift_ratio)

# Optionally, if performance metrics with ground truth available:
# accuracy = ... (compute from model predictions vs true labels)
# perf_gauge = Gauge("model_accuracy", "Model accuracy on current data", registry=registry)
# perf_gauge.set(accuracy)

# Push metrics to Prometheus (via Pushgateway)
push_to_gateway("prometheus-pushgateway.monitoring.svc.cluster.local:9091", job="evidently_drift_monitor", registry=registry)
print(f"Drift fraction = {drift_ratio:.2f}. Pushed to Prometheus.")

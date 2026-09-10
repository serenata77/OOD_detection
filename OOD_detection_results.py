import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# 1. compute Expected Calibration Error (ECE)

def compute_ece(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    probs = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(labels)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(1, device=logits.device)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = confidences.gt(bin_lower) * confidences.le(bin_upper)
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item()


# 2.  Temperature Scaling

class ModelWithTemperature(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor, lr: float = 0.01, max_iter: int = 100):
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_step():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(val_logits), val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        print(f"[Calibration] Optimal Temperature T: {self.temperature.item():.4f}")

# 3. OOD Scoring Functions
def get_msp_scores(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    scores, _ = torch.max(probs, dim=1)
    return scores

def get_energy_scores(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    return temperature * torch.logsumexp(logits / temperature, dim=1)

# 4. OOD Metrics (AUROC & FPR95% TPR)

def compute_ood_metrics(id_scores: np.ndarray, ood_scores: np.ndarray):
    y_true = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    y_scores = np.concatenate([id_scores, ood_scores])

    auroc = roc_auc_score(y_true, y_scores)

    total_id = len(id_scores)
    sorted_id_scores = np.sort(id_scores)
    threshold_idx = int(np.floor(0.05 * total_id))
    threshold_95 = sorted_id_scores[threshold_idx]

    fpr95 = np.sum(ood_scores >= threshold_95) / len(ood_scores)
    return auroc, fpr95

# 5. Visualization Functions

def plot_ood_distributions(msp_id, msp_ood, energy_id, energy_ood):
    os.makedirs("assets", exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # MSP Plot
    axes[0].hist(msp_id, bins=30, alpha=0.6, label="In-Distribution (ID)", color="#2b5c8f", density=True)
    axes[0].hist(msp_ood, bins=30, alpha=0.6, label="Out-of-Distribution (OOD)", color="#d95f02", density=True)
    axes[0].set_title("Maximum Softmax Probability (MSP)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Confidence Score", fontsize=10)
    axes[0].set_ylabel("Density", fontsize=10)
    axes[0].legend(loc="upper left")

    # Energy Plot
    axes[1].hist(energy_id, bins=30, alpha=0.6, label="In-Distribution (ID)", color="#2b5c8f", density=True)
    axes[1].hist(energy_ood, bins=30, alpha=0.6, label="Out-of-Distribution (OOD)", color="#d95f02", density=True)
    axes[1].set_title("Energy-Based Score", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Score (-Energy, Higher = ID)", fontsize=10)
    axes[1].set_ylabel("Density", fontsize=10)
    axes[1].legend(loc="upper left")

    plt.tight_layout()
    plt.savefig("assets/ood_distributions.png")
    plt.close()
    print(" Saved: assets/ood_distributions.png")

def plot_reliability_diagram(logits_raw, logits_calibrated, labels, n_bins=10):
    os.makedirs("assets", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect Calibration")

    for name, logits, color in [("Uncalibrated", logits_raw, "#d95f02"), 
                                 ("Calibrated (T-Scaled)", logits_calibrated, "#2b5c8f")]:
        probs = torch.softmax(logits, dim=1)
        confidences, predictions = torch.max(probs, dim=1)
        accuracies = predictions.eq(labels)

        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_accs, bin_confs = [], []

        for i in range(n_bins):
            in_bin = confidences.gt(bin_boundaries[i]) * confidences.le(bin_boundaries[i + 1])
            if in_bin.sum() > 0:
                bin_accs.append(accuracies[in_bin].float().mean().item())
                bin_confs.append(confidences[in_bin].mean().item())

        ax.plot(bin_confs, bin_accs, marker="o", linewidth=2, label=name, color=color)

    ax.set_title("Reliability Diagram (Calibration Curve)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Confidence", fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig("assets/calibration_curve.png")
    plt.close()
    print(" Saved: assets/calibration_curve.png")

# ==========================================
# 6. Pipeline Execution
# ==========================================
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    # synthetic dataset and model
    d_feat, n_classes = 32, 5
    model = nn.Sequential(nn.Linear(d_feat, 64), nn.ReLU(), nn.Linear(64, n_classes))

    val_x = torch.randn(500, d_feat)
    val_y = torch.randint(0, n_classes, (500,))

    test_id_x = torch.randn(1000, d_feat)
    test_id_y = torch.randint(0, n_classes, (1000,))
    test_ood_x = torch.randn(1000, d_feat) * 3.0 + 2.0

    # Inference
    model.eval()
    with torch.no_grad():
        val_logits = model(val_x)
        test_id_logits = model(test_id_x)
        test_ood_logits = model(test_ood_x)

    # 1. Calibration
    print("--- 1. Uncertainty Calibration (ECE) ---")
    ece_before = compute_ece(test_id_logits, test_id_y)
    print(f"ECE (Uncalibrated): {ece_before * 100:.2f}%")

    temp_scaler = ModelWithTemperature()
    temp_scaler.fit(val_logits, val_y)

    with torch.no_grad():
        calibrated_test_logits = temp_scaler(test_id_logits)
    ece_after = compute_ece(calibrated_test_logits, test_id_y)
    print(f"ECE (Calibrated with Temp Scaling): {ece_after * 100:.2f}%\n")

    # 2. OOD Evaluation
    print("--- 2. OOD Detection Evaluation ---")
    msp_id = get_msp_scores(test_id_logits).numpy()
    msp_ood = get_msp_scores(test_ood_logits).numpy()
    auroc_msp, fpr95_msp = compute_ood_metrics(msp_id, msp_ood)
    print(f"[MSP Baseline]   AUROC: {auroc_msp * 100:.2f}% | FPR95: {fpr95_msp * 100:.2f}%")

    energy_id = get_energy_scores(test_id_logits, temperature=1.0).numpy()
    energy_ood = get_energy_scores(test_ood_logits, temperature=1.0).numpy()
    auroc_energy, fpr95_energy = compute_ood_metrics(energy_id, energy_ood)
    print(f"[Energy Score]   AUROC: {auroc_energy * 100:.2f}% | FPR95: {fpr95_energy * 100:.2f}%\n")

    # 3. Generate Plots
    print("--- 3. Generating Figures ---")
    plot_ood_distributions(msp_id, msp_ood, energy_id, energy_ood)
    plot_reliability_diagram(test_id_logits, calibrated_test_logits, test_id_y)
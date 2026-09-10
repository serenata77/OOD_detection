import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Ensure output directory exists
os.makedirs("assets", exist_ok=True)

# ==========================================
# 1. Uncertainty Awareness Metrics
# ==========================================
def compute_uncertainty_metrics(logits: torch.Tensor):
    """
    Computes confidence score and predictive entropy for awareness handling.
    - Confidence: max softmax probability (Higher = more certain)
    - Predictive Entropy: H(p) = -sum(p * log(p)) (Higher = more uncertain)
    """
    probs = F.softmax(logits, dim=1)
    
    # Maximum Softmax Probability (Confidence score)
    confidences, predictions = torch.max(probs, dim=1)
    
    # Shannon Entropy (nats)
    entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1)
    
    return predictions, confidences, entropy

# ==========================================
# 2. Selective Prediction & Risk-Coverage (RC)
# ==========================================
def compute_risk_coverage(accuracies: np.ndarray, uncertainties: np.ndarray, higher_is_uncertain: bool = True):
    """
    Computes Selective Risk (error rate) as a function of Coverage (percentage of accepted samples).
    
    Sorts samples by uncertainty and progressively abstains/defers the most uncertain predictions.
    """
    n_samples = len(accuracies)
    
    # Sort samples: least uncertain (most confident) first
    if higher_is_uncertain:
        sorted_indices = np.argsort(uncertainties)  # low entropy first
    else:
        sorted_indices = np.argsort(-uncertainties)  # high confidence first
        
    sorted_acc = accuracies[sorted_indices]
    
    coverages = []
    selective_risks = []  # Error rate on accepted subset
    
    # Progressive evaluation from 10% to 100% coverage
    for cutoff in range(int(n_samples * 0.1), n_samples + 1, max(1, n_samples // 100)):
        accepted_acc = sorted_acc[:cutoff]
        coverage = cutoff / n_samples
        risk = 1.0 - np.mean(accepted_acc)  # Error rate on retained samples
        
        coverages.append(coverage)
        selective_risks.append(risk)
        
    coverages = np.array(coverages)
    selective_risks = np.array(selective_risks)
    
    # Area Under Risk-Coverage Curve (AURC) via trapezoidal rule
    aurc = np.trapezoid(selective_risks, coverages) if hasattr(np, 'trapezoid') else np.trapz(selective_risks, coverages)
    
    return coverages, selective_risks, aurc

# ==========================================
# 3. Visualization: Risk-Coverage Curve
# ==========================================
def plot_risk_coverage_curve(coverages_ent, risks_ent, aurc_ent, 
                             coverages_rnd, risks_rnd, aurc_rnd):
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    
    ax.plot(coverages_ent * 100, risks_ent * 100, 
            label=f"Uncertainty-Aware Deferral (AURC: {aurc_ent * 100:.2f}%)", 
            color="#2b5c8f", linewidth=2.5)
    
    ax.plot(coverages_rnd * 100, risks_rnd * 100, 
            label=f"Random Deferral Baseline (AURC: {aurc_rnd * 100:.2f}%)", 
            color="#d95f02", linestyle="--", linewidth=2)
    
    ax.set_title("Risk-Coverage Tradeoff (Selective Classification)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Coverage (% of Queries Handled by Model)", fontsize=10)
    ax.set_ylabel("Selective Risk (% Error on Accepted Queries)", fontsize=10)
    ax.set_xlim([10, 100])
    ax.set_ylim([0, max(risks_rnd.max(), risks_ent.max()) * 110])
    ax.legend(loc="upper left")
    
    plt.tight_layout()
    plt.savefig("assets/risk_coverage_curve.png")
    plt.close()
    print(" Saved: assets/risk_coverage_curve.png")

# ==========================================
# 4. Pipeline Execution
# ==========================================
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    n_samples = 2000
    n_classes = 5
    
    # 1. Simulate prediction distribution: mix of clear inputs & ambiguous/noisy inputs
    # Clean samples (high confidence, low error)
    clean_logits = torch.randn(1500, n_classes) * 1.2
    clean_labels = torch.randint(0, n_classes, (1500,))
    # Inject correlated signal for clean data
    clean_logits[torch.arange(1500), clean_labels] += 3.5
    
    # Highly ambiguous/noisy samples (high uncertainty, high error)
    noisy_logits = torch.randn(500, n_classes) * 0.4
    noisy_labels = torch.randint(0, n_classes, (500,))
    
    logits = torch.cat([clean_logits, noisy_logits], dim=0)
    labels = torch.cat([clean_labels, noisy_labels], dim=0)
    
    # 2. Extract predictions & uncertainty metrics
    predictions, confidences, entropies = compute_uncertainty_metrics(logits)
    accuracies = (predictions == labels).numpy().astype(float)
    entropies_np = entropies.numpy()
    
    # Raw system error (100% coverage, no deferral)
    full_error_rate = 1.0 - np.mean(accuracies)
    print(f"--- Awareness & Deferral Metrics ---")
    print(f"Raw System Accuracy: {(1 - full_error_rate) * 100:.2f}% (Error: {full_error_rate * 100:.2f}%)")
    
    # 3. Compute Risk-Coverage for Uncertainty-Aware vs Random Deferral
    cov_ent, risk_ent, aurc_ent = compute_risk_coverage(accuracies, entropies_np, higher_is_uncertain=True)
    
    # Random baseline: shuffle uncertainty so deferral is arbitrary
    random_scores = np.random.permutation(len(accuracies))
    cov_rnd, risk_rnd, aurc_rnd = compute_risk_coverage(accuracies, random_scores, higher_is_uncertain=True)
    
    print(f"AURC (Uncertainty-Aware): {aurc_ent * 100:.2f}% | AURC (Random): {aurc_rnd * 100:.2f}%")
    
    # Demonstrate target precision guarantee: e.g. finding coverage required for < 5% error
    idx_5pct = np.where(risk_ent <= 0.05)[0]
    if len(idx_5pct) > 0:
        achieved_cov = cov_ent[idx_5pct[-1]]
        print(f"Guarantee: System achieves <5.0% error rate at {achieved_cov * 100:.1f}% coverage (deferring { (1 - achieved_cov) * 100:.1f}% to human expert).")
        
    # 4. Generate visual artifact
    plot_risk_coverage_curve(cov_ent, risk_ent, aurc_ent, cov_rnd, risk_rnd, aurc_rnd)
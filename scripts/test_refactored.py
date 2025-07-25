#!/usr/bin/env python3

import torch
import shap
import sys
import os

# Add the project root to Python path so we can import shap_enhanced
sys.path.insert(0, '/home/niyang/repos/enhanced_shap/src')

try:
    from shap_enhanced.tools.predefined_models import RealisticLSTM
    from shap_enhanced.tools.datasets import generate_synthetic_seqregression
    from shap_enhanced.tools.evaluation import compute_shapley_gt_seq
    from shap_enhanced.tools.comparison import Comparison
    from shap_enhanced.tools.visulization import plot_mse_pearson, plot_3d_bars

    from shap_enhanced.explainers.CASHAP import CoalitionAwareSHAPExplainer
    from shap_enhanced.explainers.AttnSHAP import AttnSHAPExplainer
    from shap_enhanced.explainers.BSHAP import BShapExplainer
    from shap_enhanced.explainers.CMSHAP import ContextualMaskingSHAPExplainer
    from shap_enhanced.explainers.ESSHAP import EnsembleSHAPWithNoise
    from shap_enhanced.explainers.SurroSHAP import SurrogateSHAPExplainer
    from shap_enhanced.explainers.RLSHAP import RLShapExplainer
    from shap_enhanced.explainers.LatentSHAP import LatentSHAPExplainer, Conv1dEncoder, Conv1dDecoder, train_conv1d_autoencoder
    from shap_enhanced.explainers.MBSHAP import NearestNeighborMultiBaselineSHAP
    from shap_enhanced.explainers.TimeSHAP import TimeSHAPExplainer
    from shap_enhanced.explainers.ERSHAP import ERSHAPExplainer
    from shap_enhanced.explainers.hSHAP import HShapExplainer, generate_hierarchical_groups
    
    print("✅ All imports successful!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_basic_functionality():
    """Test basic functionality of a subset of explainers"""
    print("\n🧪 Testing basic functionality...")
    
    # 1. Generate synthetic sequential data
    seq_len, n_features, n_samples = 5, 2, 50  # Smaller for testing
    X, y = generate_synthetic_seqregression(seq_len, n_features, n_samples)
    print(f"✅ Generated data: X.shape={X.shape}, y.shape={y.shape}")

    # 2. Train a simple LSTM model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")
    
    model = RealisticLSTM(input_dim=n_features).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)

    model.train()
    print("🏋️ Training model...")
    for epoch in range(20):  # Reduced epochs for testing
        opt.zero_grad()
        loss = loss_fn(model(X_t), y_t)
        loss.backward()
        opt.step()
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}, Loss: {loss.item():.4f}")

    model.eval()
    print("✅ Model trained!")

    # 3. Pick a test sample
    x_test = X[0]
    print(f"✅ Test sample shape: {x_test.shape}")

    # 4. Test a few key explainers
    explainers_to_test = {
        "CMSHAP": lambda: ContextualMaskingSHAPExplainer(model=model, device=device),
        "ERSHAP": lambda: ERSHAPExplainer(
            model=model,
            background=X[:20],
            n_coalitions=10,
            mask_strategy="mean",
            device=device
        ),
        "TimeSHAP": lambda: TimeSHAPExplainer(
            model=model,
            background=X[:20],
            mask_strategy="mean",
            device=device
        ),
        "hSHAP": lambda: HShapExplainer(
            model=model,
            background=X[:20],
            hierarchy=generate_hierarchical_groups(T=seq_len, F=n_features, time_block=2),
            mask_strategy="mean",
            device=device
        )
    }
    
    results = {}
    for name, explainer_fn in explainers_to_test.items():
        try:
            print(f"\n🔬 Testing {name}...")
            explainer = explainer_fn()
            
            # Test SHAP values computation
            shap_vals = explainer.shap_values(x_test, nsamples=5)  # Small nsamples for testing
            print(f"  ✅ {name} SHAP values shape: {shap_vals.shape}")
            print(f"  ✅ {name} SHAP values sum: {shap_vals.sum():.4f}")
            
            results[name] = shap_vals
            
        except Exception as e:
            print(f"  ❌ {name} failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎉 Successfully tested {len(results)}/{len(explainers_to_test)} explainers!")
    return results

if __name__ == "__main__":
    test_basic_functionality()
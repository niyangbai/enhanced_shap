from shap_enhanced.simulation.simulator import ModelSimulation

def compare_explainers(explainers, X, y, true_shap_function, metric='mse'):
    """Compare different explainers and summarize the results."""
    results = {}
    
    for explainer_name, explainer in explainers.items():
        simulation = ModelSimulation(model=None, explainer=explainer, true_shap_function=true_shap_function, metric=metric)
        mse = simulation.run_simulation(n_samples=X.shape[0], n_features=X.shape[1])
        results[explainer_name] = mse
    
    return results

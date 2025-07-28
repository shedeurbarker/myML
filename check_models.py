import json

# Load the training metadata
with open('results/train_optimization_models/training_metadata_improved.json', 'r') as f:
    data = json.load(f)

print("=== BEST MODELS SELECTED ===")
print("\nEfficiency Models:")
for target, info in data['efficiency_models'].items():
    print(f"  {target}: {info['best_model']}")

print("\nRecombination Models:")
for target, info in data['recombination_models'].items():
    print(f"  {target}: {info['best_model']}")

print("\n=== ALGORITHM PERFORMANCE SUMMARY ===")
print("\nEfficiency Models - CV Scores:")
for target, info in data['efficiency_models'].items():
    print(f"\n{target}:")
    for model_name, scores in info['all_scores'].items():
        cv_score = scores['cv_mean']
        test_r2 = scores['test_metrics']['r2']
        print(f"  {model_name}: CV={cv_score:.4f}, Test R²={test_r2:.4f}")

print("\nRecombination Models - CV Scores:")
for target, info in data['recombination_models'].items():
    print(f"\n{target}:")
    for model_name, scores in info['all_scores'].items():
        cv_score = scores['cv_mean']
        test_r2 = scores['test_metrics']['r2']
        print(f"  {model_name}: CV={cv_score:.4f}, Test R²={test_r2:.4f}") 
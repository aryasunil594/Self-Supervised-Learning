
import pandas as pd
import matplotlib.pyplot as plt

models = ["SimCLR", "Supervised ResNet18"]

# Replace these values with your actual evaluation results
accuracy = [47.18, 0.0]
precision = [0.4900, 0.0]
recall = [0.4718, 0.0]
f1_score = [0.4681, 0.0]

results = pd.DataFrame({
    "Model": models,
    "Accuracy (%)": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1_score
})

print("\nModel Comparison Results")
print(results.to_string(index=False))

results.to_csv("model_comparison.csv", index=False)
print("\nSaved: model_comparison.csv")

metrics = ["Accuracy (%)", "Precision", "Recall", "F1-Score"]

for metric in metrics:
    plt.figure(figsize=(7, 4))
    plt.bar(results["Model"], results[metric])
    plt.title(f"Model Comparison - {metric}")
    plt.ylabel(metric)
    plt.tight_layout()

    filename = metric.lower().replace(" ", "_").replace("-", "_") + ".png"
    plt.savefig(filename)
    plt.close()

print("Comparison graphs saved successfully!")

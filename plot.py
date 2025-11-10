from sklearn.tree import  plot_tree
import matplotlib.pyplot as plt

# Plot
plt.figure(figsize=(15,8))
plot_tree(
    model,
    feature_names=X.columns,
    # # class_names=[str(c) for c in model.classes_],
    filled=True,
    rounded=True,
    # fontsize=10
)
plt.show()

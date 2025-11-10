# Q.2) Decision tree (ID3 algorithm using entropy)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier ,plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
df = pd.read_csv("2 university_admission.csv")
new = df.columns
new = new.str.strip()

x = df.drop('Chance_of_Admission', axis=1)
y = (df['Chance_of_Admission'] > 0.5).astype(int)

xt, xtst, yt, ytst = train_test_split(x, y)
m = DecisionTreeClassifier(criterion='entropy')
m.fit(xt, yt)

yp = m.predict(xtst)
print("Accuracy:", accuracy_score(ytst, yp))

s = [list(xt.iloc[0])]  # example
res = m.predict(s)
print("Predicted class:", res[0])
feature_names = ['Outlook', 'Temperature', 'Humidity', 'Wind']
print("Decision Tree Structure:\n")
print(export_text(clf, feature_names=feature_names))

# Plot the tree
plt.figure()
plot_tree(clf, feature_names=feature_names, class_names=['No', 'Yes'], filled=True)
plt.show()

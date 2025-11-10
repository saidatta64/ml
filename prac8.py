# Q.8) Market Basket Optimization 
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder    # (install this module: )
from mlxtend.frequent_patterns import apriori, association_rules

data = pd.read_csv("8 Market_Basket_Optimisation.csv", header=None)
rec = [[str(i) for i in row if pd.notna(i)] for row in data.values]

te = TransactionEncoder()
df = pd.DataFrame(te.fit(rec).transform(rec))

freq = apriori(df, min_support=0.01)
r = association_rules(freq, metric="lift")

print("Top Association Rules:\n")
print(r.head(10))

plt.scatter(r['support'], r['confidence'])
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()

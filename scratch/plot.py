import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
x = np.array([1,2,3,4,5])
y = np.sin(x)
z = np.dstack((x,y))
print(z)
df = pd.DataFrame(z[0],columns=['x','y'])
print(df)

# Show the results of a linear regression within each dataset
sns.boxplot(data=df, x="x", y="y")
plt.show()
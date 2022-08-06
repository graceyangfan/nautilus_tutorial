import matplotlib.pyplot as plt
import pandas as pd 
df = pd.read_csv("account.csv")
df["date"] = df['Unnamed: 0'].apply(pd.Timestamp).values
plt.plot(df['date'],df['total'])
plt.show()

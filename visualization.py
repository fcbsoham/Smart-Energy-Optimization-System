import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("smart_energy_subset.csv")

hourly = df.groupby("hour")["meter_reading"].mean()

plt.plot(hourly)
plt.title("Energy Consumption by Hour")
plt.xlabel("Hour")
plt.ylabel("Energy")
plt.show()
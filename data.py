from ucimlrepo import fetch_ucirepo 
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
  
# fetch dataset 
metro_interstate_traffic_volume = fetch_ucirepo(id=492) 
  
# data (as pandas dataframes) 
X = metro_interstate_traffic_volume.data.features 
y = metro_interstate_traffic_volume.data.targets 
  
# metadata 
print(metro_interstate_traffic_volume.metadata) 
  
# variable information 
print(metro_interstate_traffic_volume.variables) 

# create histogram of traffic volume
plt.hist(y, bins=20)
plt.title('Distribution of Traffic Volume')
plt.xlabel('Traffic Volume')
plt.ylabel('Frequency')
plt.show()


data = pd.concat([X, y], axis=1)

corr = data.corr(numeric_only=True)

plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Feature Correlation Matrix")
plt.show()

sns.boxplot(x=X["weather_main"], y=y["traffic_volume"])
plt.xticks(rotation=45)
plt.title("Traffic Volume by Weather Condition")
plt.show()

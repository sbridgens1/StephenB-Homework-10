# StephenB-Homework-10


# Import required libraries and dependencies
import pandas as pd
import hvplot.pandas
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the data into a Pandas DataFrame
df_market_data = pd.read_csv(
    Path("Resources/crypto_market_data.csv"),
    index_col="coin_id")

# Display sample data
df_market_data.head(10)

# Generate summary statistics
df_market_data.describe()

# Plot your data to see what's in your DataFrame
df_market_data.hvplot.line(
    width=800,
    height=400,
    rot=90
)


Prepare the Data

# Use the `StandardScaler()` module from scikit-learn to normalize the data from the CSV file
scaled_data = StandardScaler().fit_transform(df_market_data)

# Create a DataFrame with the scaled data
df_market_data_scaled = pd.DataFrame(
    scaled_data,
    columns=df_market_data.columns
)

# Copy the crypto names from the original data
df_market_data_scaled["coin_id"] = df_market_data.index

# Set the coinid column as index
df_market_data_scaled = df_market_data_scaled.set_index("coin_id")

# Display sample data
df_market_data_scaled.head()


Find the Best Value for k Using the Original Data


# Create a list with the number of k-values to try
# Use a range from 1 to 11
k = list(range(1, 11))

# Create an empy list to store the inertia values
inertia = []

# Create a for loop to compute the inertia with each possible value of k

for i in k:
    k_model = KMeans(n_clusters=i, random_state=2)
    k_model.fit(df_market_data_scaled)
    inertia.append(k_model.inertia_)

# Create a dictionary with the data to plot the Elbow curve

elbow_data = {"k": k, "inertia": inertia}

# Create a DataFrame with the data to plot the Elbow curve

df_elbow = pd.DataFrame(elbow_data)

df_elbow.head()

# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.

df_elbow.hvplot.line(
    x="k", 
    y="inertia", 
    title="Elbow Curve", 
    xticks=k
)

**Question:** What is the best value for `k`?

**Answer:** 3 


Cluster Cryptocurrencies with K-means Using the Original Data


# Initialize the K-Means model using the best value for k

model = KMeans(n_clusters=3, random_state=1)

# Fit the K-Means model using the scaled data

model.fit(df_market_data_scaled)

# Predict the clusters to group the cryptocurrencies using the scaled data

k_lower = model.predict(df_market_data_scaled)

# Create a copy of the DataFrame
df_market_data_scaled_predictions = df_market_data_scaled.copy()

# Add a new column to the DataFrame with the predicted clusters
df_market_data_scaled_predictions['clusters_lower'] = k_lower

# Display sample data
df_market_data_scaled_predictions.head(3)

# Create a scatter plot using hvPlot by setting 
# `x="price_change_percentage_24h"` and `y="price_change_percentage_7d"`. 
# Color the graph points with the labels found using K-Means and 
# add the crypto name in the `hover_cols` parameter to identify 
# the cryptocurrency represented by each data point.

df_market_data_scaled_predictions.hvplot.scatter(
    x="price_change_percentage_24h",
    y="price_change_percentage_7d",
    by="clusters_lower",
    hover_cols =["coin_id"]
).opts(yformatter="%.0f")


Optimize Clusters with Principal Component Analysis


# Create a PCA model instance and set `n_components=3`.
pca = PCA(n_components=3)

# Use the PCA model with `fit_transform` to reduce to 
# three principal components.
market_data_scaled_pca = pca.fit_transform(df_market_data_scaled_predictions)

# View the first five rows of the DataFrame. 
market_data_scaled_pca[:5]

# Retrieve the explained variance to determine how much information 
# can be attributed to each principal component.
pca.explained_variance_ratio_

**Question:** What is the total explained variance of the three principal components?

**Answer:** 0.9

# Create a new DataFrame with the PCA data.

df_market_data_scaled_pca = pd.DataFrame(
    market_data_scaled_pca,
    columns=["PCA1", "PCA2","PCA3"]
)

# Copy the crypto names from the original data
df_market_data_scaled_pca["coin_id"] = df_market_data.index

# Set the coinid column as index
df_market_data_scaled_pca = df_market_data_scaled_pca.set_index("coin_id")

# Display sample data
df_market_data_scaled_pca.head()


Find the Best Value for k Using the PCA Data

# Create a list with the number of k-values to try
# Use a range from 1 to 11
k = list(range(1, 11))

# Create an empy list to store the inertia values
inertia = []

# Create a for loop to compute the inertia with each possible value of k

for i in k:
    model = KMeans(n_clusters=i, random_state=0)
    model.fit(market_data_scaled_pca)
    inertia.append(model.inertia_)

# Create a dictionary with the data to plot the Elbow curve
elbow_data_pca = {
    "k": k,
    "inertia": inertia
}

# Create a DataFrame with the data to plot the Elbow curve
df_elbow_pca = pd.DataFrame(elbow_data_pca)

# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.
elbow_plot_pca = df_elbow_pca.hvplot.line(x="k", y="inertia", title="Elbow Curve Using PCA Data", xticks=k)
elbow_plot_pca


**Question:** What is the best value for `k` when using the PCA data?

**Answer:** # Based on this Elbow Curve, it looks like k=3 is still the correct one.


**Question:** Does it differ from the best k value found using the original data?

**Answer:** # No


Cluster Cryptocurrencies with K-means Using the PCA Data

# Initialize the K-Means model using the best value for k
model = KMeans(n_clusters=4)

# Fit the K-Means model using the PCA data
model.fit(market_data_scaled_pca)

# Predict the clusters to group the cryptocurrencies using the PCA data
cryptocurrencies_clusters = model.predict(market_data_scaled_pca)

# View the resulting array of cluster values.
print(cryptocurrencies_clusters)

# Create a copy of the DataFrame with the PCA data
market_data_scaled_pca_predictions = market_data_scaled_pca.copy()


# Add a new column to the DataFrame with the predicted clusters
df_market_data_scaled_predictions['clusters_lower'] = cryptocurrencies_clusters


# Display sample data
df_market_data_scaled_predictions.head()

# Create a scatter plot using hvPlot by setting 
# `x="PC1"` and `y="PC2"`. 
# Color the graph points with the labels found using K-Means and 
# add the crypto name in the `hover_cols` parameter to identify 
# the cryptocurrency represented by each data point.
df_market_data_scaled_predictions.hvplot.scatter(
    x="price_change_percentage_24h",
    y="price_change_percentage_7d",
    by="clusters_lower"

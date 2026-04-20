import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# =============================================================================
# Question (1)(a) - Loading and Cleaning the Dataset
# =============================================================================

column_names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight',
                'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings']
df = pd.read_csv('data/abalone-1.data', names=column_names)

# Check duplicate data
duplicate = df[df.duplicated()]
print(duplicate)
print("There are no duplicate records found.")

print("Number of observations of original data set:", df.shape[0])
print("\nAny NA value:", df.isnull().values.any())
print("No NA values were found. No need to drop NA rows.")

# Checking any continuous variables with invalid data
print("\nAny Length less than or equal Zero :", (df["Length"] <= 0).values.any())
print("\nAny Diameter less than or equal Zero :", (df["Diameter"] <= 0).values.any())
print("\nAny Height less than or equal Zero :", (df["Height"] <= 0).values.any())
print("\nAny Whole_weight less than or equal Zero :", (df["Whole_weight"] <= 0).values.any())
print("\nAny Shucked_weight less than or equal Zero :", (df["Shucked_weight"] <= 0).values.any())
print("\nAny Viscera_weight less than or equal Zero :", (df["Viscera_weight"] <= 0).values.any())
print("\nAny Shell_weight less than or equal Zero :", (df["Shell_weight"] <= 0).values.any())
print("\nAny Rings less than or equal Zero :", (df["Rings"] <= 0).values.any())

invalid_height_rows = df[df['Height'] <= 0]
print('Number of rows with Height value Zero :', len(invalid_height_rows))
print("There are two invalid rows where the Height value is zero. These rows will be removed.")

df = df[df['Height'] > 0].reset_index(drop=True)
print("Number of observations after cleaning data set:", df.shape[0])

# Descriptive Statistics
print(df.describe())

# =============================================================================
# Question (1)(b) - Histogram of Length
# =============================================================================

sns.histplot(df['Length'], bins=30, kde=True)
plt.title('Distribution of Abalone Length')
plt.xlabel('Length (scaled)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# =============================================================================
# Question (1)(c) - Correlation Matrix and VIF
# =============================================================================

X = df.drop(columns=['Rings', 'Sex'])

# Correlation matrix heatmap
correlation_matrix = X.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Predictor Variables')
plt.tight_layout()
plt.show()

# Identify pairs of variables with high correlation (|r| > 0.6)
high_correlation_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.6:
            high_correlation_pairs.append(
                (correlation_matrix.index[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j])
            )

high_corr_df = pd.DataFrame(high_correlation_pairs, columns=["Variable 1", "Variable 2", "Correlation"])
print("Highly Correlated Variable Pairs (|r| > 0.6):\n", high_corr_df)

# Calculate VIF
X_const = add_constant(X)
vif_data = pd.DataFrame()
vif_data["Variable"] = X_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
print("\nVIF values:\n", vif_data)

# =============================================================================
# Question (1)(d) - Linear Model with Interaction
# =============================================================================

model_interaction = smf.ols('Rings ~ Whole_weight * C(Sex)', data=df).fit()
print(model_interaction.summary())

# =============================================================================
# Question (1)(e) - Full OLS Regression Model
# =============================================================================

train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

model = smf.ols(
    'Rings ~ C(Sex) + Length + Diameter + Height + Whole_weight + Shucked_weight + Viscera_weight + Shell_weight',
    data=train_df
).fit()
print(model.summary())

selected_columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight',
                    'Shucked_weight', 'Viscera_weight', 'Shell_weight']

ypred = model.predict(test_df[selected_columns])
mse = mean_squared_error(test_df['Rings'], ypred)
print("Test MSE:", mse)

# Residuals vs Fitted
residuals = np.array(test_df["Rings"]) - np.array(ypred)
plt.scatter(ypred, residuals, s=15)
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.axhline(y=0.0, color="b", linestyle="-")
plt.title("Residuals vs Fitted Values Plot for Homoscedasticity Check")
plt.tight_layout()
plt.show()

# =============================================================================
# Question (1)(f) - PCA-based Regression
# =============================================================================

X_pca_input = df[['Length', 'Diameter', 'Height', 'Whole_weight',
                   'Shucked_weight', 'Viscera_weight', 'Shell_weight']]
y = df['Rings']

X_scaled = StandardScaler().fit_transform(X_pca_input)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained = pca.explained_variance_ratio_
df_for_explained = pd.DataFrame([explained], columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7'])
print(df_for_explained)
print('The first three components explains 97.18% of the variability.')

X_pca_selected = X_pca[:, :7]
PC_df = pd.DataFrame(data=X_pca_selected, columns=["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7"])
PC_df_with_rings = pd.concat([PC_df, df[['Rings', 'Sex']]], axis=1)

PC_train_df, PC_test_df = train_test_split(PC_df_with_rings, test_size=0.3, random_state=42)

PC_model = smf.ols('Rings ~ PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + C(Sex)', data=PC_train_df).fit()
print(PC_model.summary())

PC_ypred = PC_model.predict(PC_test_df)
PC_mse = mean_squared_error(PC_test_df['Rings'], PC_ypred)
print("PC Test MSE:", PC_mse)

PC_residuals = np.array(PC_test_df["Rings"]) - np.array(PC_ypred)
plt.scatter(PC_ypred, PC_residuals, s=15)
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.axhline(y=0.0, color="b", linestyle="-")
plt.title("Residuals vs Fitted Values (PCA Model)")
plt.tight_layout()
plt.show()

# =============================================================================
# Question (1)(g) - Predict Age of New Sample
# =============================================================================

new_sample = pd.DataFrame([{
    'Sex': 'M',
    'Length': 0.52,
    'Diameter': 0.41,
    'Height': 0.14,
    'Whole_weight': 0.83,
    'Shucked_weight': 0.36,
    'Viscera_weight': 0.18,
    'Shell_weight': 0.24
}])

scaler_new = StandardScaler().fit(X_pca_input)
new_X_scaled = scaler_new.transform(
    new_sample[['Length', 'Diameter', 'Height', 'Whole_weight',
                'Shucked_weight', 'Viscera_weight', 'Shell_weight']]
)

new_sample_pcs = pca.transform(new_X_scaled)
new_PC_df = pd.DataFrame(new_sample_pcs[:, :7], columns=[f'PC{i+1}' for i in range(7)])
new_PC_df['Sex'] = new_sample['Sex'].values

predicted_rings = PC_model.predict(new_PC_df)
print("Predicted Rings:", predicted_rings.iloc[0])
print("Estimated Age (Rings + 1.5):", predicted_rings.iloc[0] + 1.5, "Years")

# =============================================================================
# Question (2)(a) - Logistic Regression: Sensitivity and Specificity
# =============================================================================

df_binary = df[df['Sex'] != 'I'].copy()
df_binary['Sex'] = df_binary['Sex'].map({'M': 1, 'F': 0})

X_bin = df_binary.drop(columns=['Sex', 'Rings'])
y_bin = df_binary['Sex']

X_train, X_test, y_train, y_test = train_test_split(X_bin, y_bin, test_size=0.3, random_state=42)

sns.countplot(x='Sex', data=y_train.to_frame())
plt.title("Check Data Balancing\nDistribution of response: Male [1] vs Female [0]")
plt.xlabel("Sex")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

class_counts = y_train.value_counts(normalize=True) * 100
print("Male and Female data distribution in Percentage\nMale [1] vs Female [0]\n")
print(class_counts)

log_reg = LogisticRegression(max_iter=1000, solver='liblinear')
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

sensitivity = round(cm[1, 1] / np.sum(cm[1, :]), 3)
specificity = round(cm[0, 0] / np.sum(cm[0, :]), 3)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)

# =============================================================================
# Question (2)(b) - Accuracy, Sensitivity, Specificity
# =============================================================================

accuracy = round(np.sum(np.diagonal(cm)) / np.sum(cm), 3)
print("Accuracy:", accuracy)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)

# =============================================================================
# Question (3)(a) - Load and Clean Q3 Dataset
# =============================================================================

df_q3 = pd.read_csv("data/data_q3Updated.csv")

selected_variables = ["VA", "GE", "RQ", "RL", "CR", "Top200", "Top201-500",
                      "Top501-800", "Top801-1000", "ISCED5", "ISCED6", "ISCED7", "ISCED8", "TL"]
df_q3 = df_q3[selected_variables]

duplicate_q3 = df_q3[df_q3.duplicated()]
print(duplicate_q3)
print("Number of observations of original data set:", df_q3.shape[0])

print("\nGeneral overview of the data to check any missing data\n")
df_q3.info()

print("\nAny NA value:", df_q3.isnull().values.any())

df_q3 = df_q3.dropna()
df_q3.info()

# =============================================================================
# Question (3)(b) - Elbow Method and Dendrogram
# =============================================================================

scaler_q3 = StandardScaler()
fitted_q3 = scaler_q3.fit(df_q3)
X_std = pd.DataFrame(fitted_q3.transform(df_q3))

# Elbow Method
def wcss(x, kmax):
    wcss_s = []
    for k in range(2, kmax + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(x)
        wcss_s.append(kmeans.inertia_)
    return wcss_s

fig = plt.figure(figsize=(19, 11))
ax = fig.add_subplot(1, 1, 1)
kmax = 10
ax.plot(range(2, kmax + 1), wcss(X_std, kmax))
ax.tick_params(axis="both", which="major", labelsize=20)
ax.set_xlabel("Number of Clusters (k)", fontsize=25)
ax.set_ylabel("Sum of Squared Error (WCSS)", fontsize=25)
ax.set_title("Sum of Squared Error by Number of Clusters (Elbow Method)", fontsize=25)
plt.tight_layout()
plt.show()

# Dendrogram
linked = linkage(X_std, method='ward')
plt.figure(figsize=(10, 6))
dendrogram(linked, orientation='top', distance_sort='ascending')
plt.title('Dendrogram for Agglomerative Clustering')
plt.xlabel('Countries')
plt.ylabel('Distance')
plt.grid(True)
plt.tight_layout()
plt.show()

# =============================================================================
# Question (3)(c) - K-means and Hierarchical Clustering with PCA Visualization
# =============================================================================

# K-means
pca_q3 = PCA(n_components=2)
X_pca_q3 = pca_q3.fit_transform(X_std)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_std)

pca_df = pd.DataFrame(X_pca_q3, columns=['PC1', 'PC2'])
pca_df['Cluster'] = clusters

plt.figure(figsize=(8, 6))
for cluster in pca_df['Cluster'].unique():
    plt.scatter(
        pca_df[pca_df['Cluster'] == cluster]['PC1'],
        pca_df[pca_df['Cluster'] == cluster]['PC2'],
        label=f'Cluster {cluster}'
    )
plt.title('K-means Clusters Visualized with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=3)
clusters_hier = hierarchical.fit_predict(X_std)

pca_df_hier = pd.DataFrame(X_pca_q3, columns=['PC1', 'PC2'])
pca_df_hier['Cluster'] = clusters_hier

plt.figure(figsize=(8, 6))
for cluster in pca_df_hier['Cluster'].unique():
    plt.scatter(
        pca_df_hier[pca_df_hier['Cluster'] == cluster]['PC1'],
        pca_df_hier[pca_df_hier['Cluster'] == cluster]['PC2'],
        label=f'Cluster {cluster}'
    )
plt.title('Hierarchical Clustering (PCA Scatter Plot)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

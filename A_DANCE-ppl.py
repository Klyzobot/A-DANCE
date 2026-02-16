#Original file is located at
    #https://colab.research.google.com/drive/1wvIT8Tw9w7l3BczCuvqK0xYNI3QFAnwZ

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# Step 1: Load Data
# -----------------------------
X = pd.read_csv("/content/selected_expression_matrix.csv", index_col=0)
metadata = pd.read_csv("/content/metadata_binary.csv")
metadata.columns = ["Sample", "Condition"]

# Align samples
X = X.loc[metadata["Sample"]]
y = metadata["Condition"].values

print("Feature matrix shape:", X.shape)
print("Target vector shape:", y.shape)

# -----------------------------
# Step 2: Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -----------------------------
# Step 3: Train Model
# -----------------------------
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# -----------------------------
# Step 4: Evaluate
# -----------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

print(f"Accuracy: {acc:.3f}")
print(f"ROC AUC: {roc:.3f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Confusion matrix heatmap
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
!pip install xgboost

from xgboost import XGBClassifier

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:, 1]

print("\nXGBoost Results")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("ROC AUC:", roc_auc_score(y_test, y_prob_xgb))
print("Classification Report:\n", classification_report(y_test, y_pred_xgb))

!pip install lightgbm
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Create folder to save confusion matrices
os.makedirs("confusion_matrices", exist_ok=True)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "MLP Neural Net": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "LightGBM": LGBMClassifier()
}

# Train, Predict, and Save Confusion Matrix
for name, model in models.items():
    print(f"\n🔵 Model: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    print(f"✅ Accuracy: {acc:.3f}")
    print(f"✅ ROC AUC: {roc:.3f}")
    print("📄 Classification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)

    # Plot Confusion Matrix
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    # Save image
    filename = f"confusion_matrices/{name.replace(' ', '_')}_confusion_matrix.png"
    plt.savefig(filename)
    plt.show()
    print(f"📸 Saved confusion matrix as: {filename}")

!pip install catboost
from catboost import CatBoostClassifier

model = CatBoostClassifier(iterations=300, random_seed=42, verbose=0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation
acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

print("\n🧠 Model: CatBoost")
print(f"✅ Accuracy: {acc:.3f}")
print(f"✅ ROC AUC: {roc:.3f}")
print("📄 Classification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("CatBoost - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("CatBoost_confusion_matrix.png")
plt.show()

!pip install pytorch_tabnet
from pytorch_tabnet.tab_model import TabNetClassifier

clf = TabNetClassifier(verbose=0, seed=42)
clf.fit(X_train.values, y_train, eval_set=[(X_test.values, y_test)])

y_pred = clf.predict(X_test.values)
y_prob = clf.predict_proba(X_test.values)[:, 1]

# Evaluation
acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

print("\n🧠 Model: TabNet")
print(f"✅ Accuracy: {acc:.3f}")
print(f"✅ ROC AUC: {roc:.3f}")
print("📄 Classification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
plt.title("TabNet - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("TabNet_confusion_matrix.png")
plt.show()

# --------------------------------------
# 📦 Step 1: Install TPOT (Google Colab)
# --------------------------------------
!pip install tpot
# --------------------------------------
# 📚 Step 2: Import Libraries
# --------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tpot import TPOTClassifier
import os

# --------------------------------------
# 📂 Step 3: Load Data
# --------------------------------------
# Load expression matrix
X = pd.read_csv("/content/selected_expression_matrix.csv", index_col=0)

# Load metadata and align
meta = pd.read_csv("/content/metadata_binary.csv")
meta.columns = ["Sample", "Condition"]
X = X.loc[meta["Sample"]]  # align
y = meta["Condition"].values  # labels

# --------------------------------------
# ✂️ Step 4: Train-Test Split
# --------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --------------------------------------
# ⚙️ Step 5: Run TPOT AutoML
# (use n_jobs=1 to avoid dask issues on Colab)
# --------------------------------------
tpot = TPOTClassifier(
    generations=5,
    population_size=50,
    random_state=42,
    n_jobs=1  # important fix for Colab
)
tpot.fit(X_train, y_train)

# --------------------------------------
# 📊 Step 6: Evaluate Model
# --------------------------------------
print("\n✅ Best pipeline:\n", tpot.fitted_pipeline_)

y_pred = tpot.predict(X_test)
y_prob = tpot.predict_proba(X_test)[:, 1]

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("✅ ROC AUC:", roc_auc_score(y_test, y_prob))
print("📄 Classification Report:\n", classification_report(y_test, y_pred))

# --------------------------------------
# 📸 Step 7: Confusion Matrix Plot
# --------------------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges")
plt.title("TPOT AutoML - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

# Save the plot
os.makedirs("confusion_matrices_tpot", exist_ok=True)
plt.savefig("TPOT_confusion_matrix.png")
plt.show()

# --------------------------------------
# 💾 Step 8: Export Best Model Code
# --------------------------------------
tpot.export("tpot_best_pipeline.py")

!pip install imbalanced-learn

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Load your already processed expression data (280 selected genes)
X = pd.read_csv("/content/selected_expression_matrix.csv", index_col=0)

# Load labels
meta = pd.read_csv("/content/metadata_binary.csv")
meta.columns = ["Sample", "Condition"]
X = X.loc[meta["Sample"]]
y = meta["Condition"].values

# Before balancing: check class distribution
from collections import Counter
print("Before balancing:", Counter(y))

# Apply SMOTE to balance
sm = SMOTE(random_state=42)
X_balanced, y_balanced = sm.fit_resample(X, y)

print("After balancing:", Counter(y_balanced))

!pip install imblearn

X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, stratify=y_balanced, random_state=42
)

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from collections import Counter

# X: your gene expression DataFrame (319 samples × 280 genes)
# y: condition labels (0 = Control, 1 = AD)

# Split first
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Apply SMOTE on training set
sm = SMOTE(random_state=42)
X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

# Print class balance
print("Training set after SMOTE:", Counter(y_train_bal))
print("Testing set:", Counter(y_test))

# Convert to DataFrames
train_df = pd.DataFrame(X_train_bal, columns=X.columns)
train_df["Condition"] = y_train_bal

test_df = pd.DataFrame(X_test, columns=X.columns)
test_df["Condition"] = y_test

# Save to CSV
train_df.to_csv("balanced_training_data.csv", index=False)
test_df.to_csv("testing_data.csv", index=False)

print("✅ Files saved:")
print("- balanced_training_data.csv")
print("- testing_data.csv")

import pandas as pd

# Load your gene expression matrix
df = pd.read_csv("/content/selected_expression_matrix.csv", index_col=0)

# Get the gene names (i.e., column names)
gene_list = df.columns.tolist()

# Save to file for Enrichr
with open("genes_for_enrichr.txt", "w") as f:
    for gene in gene_list:
        f.write(f"{gene}\n")

import joblib
joblib.dump(tpot.fitted_pipeline_, 'best_tpot_model.pkl')

model = joblib.load('best_tpot_model.pkl')
pred = model.predict(new_gene_expression_data)

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 🔁 Predict if not already done
y_pred = tpot.predict(X_test)

# ✅ Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# 📊 Plot
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Control (0)", "AD (1)"],
            yticklabels=["Control (0)", "AD (1)"])
plt.title("TPOT AutoML - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

# 💾 Save the plot
os.makedirs("confusion_matrices_tpot", exist_ok=True)
plt.savefig("TPOT_confusion_matrix.png", dpi=300)
plt.show()

import pandas as pd

# Step 1: Load data
expr_df = pd.read_csv("/content/selected_expression_matrix.csv", index_col=0)
mapping_df = pd.read_csv("/content/Gene-Exp.csv")  # Adjust name if different

# Step 2: Ensure correct column names
# Let's assume mapping_df has columns: "gene_number", "gene_symbol"
mapping_df.columns = ["GeneID", "Symbol"]

# Step 3: Convert columns (gene numbers) to strings to match mapping
expr_df.columns = expr_df.columns.astype(str)
mapping_df["GeneID"] = mapping_df["GeneID"].astype(str)

# Step 4: Create dictionary: number → symbol
gene_dict = dict(zip(mapping_df["GeneID"], mapping_df["Symbol"]))

# Step 5: Rename expression dataframe columns using the gene mapping
expr_df.rename(columns=gene_dict, inplace=True)

# Step 6: Save new matrix and extract gene list for Enrichr
expr_df.to_csv("expression_with_gene_symbols.csv")

# Save only gene symbols to .txt file
with open("genes_for_enrichr.txt", "w") as f:
    for gene in expr_df.columns:
        f.write(f"{gene}\n")

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Load expression matrix (selected genes)
df = pd.read_csv("/content/expression_with_gene_symbols.csv", index_col=0)

# Compute correlation
corr_matrix = df.corr()

# Filter strong correlations
threshold = 0.7
edges = [(gene1, gene2) for gene1 in corr_matrix.columns for gene2 in corr_matrix.columns
         if gene1 != gene2 and corr_matrix.loc[gene1, gene2] > threshold]

# Build and draw network
G = nx.Graph()
G.add_edges_from(edges)
plt.figure(figsize=(10, 8))
nx.draw_networkx(G, with_labels=True, node_color='lightblue', edge_color='gray', font_size=8)
plt.title("Gene Co-Expression Network (r > 0.7)")
plt.show()

import pandas as pd
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
file_path = "/content/expression_with_gene_symbols.csv"
expression_df = pd.read_csv(file_path, index_col=0)
top_n = 320
gene_variances = expression_df.var(axis=0)
top_genes = gene_variances.sort_values(ascending=False).head(top_n).index
filtered_df = expression_df[top_genes]
correlation_matrix = filtered_df.corr()
corr_pairs = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
edges_df = corr_pairs.stack().reset_index()
edges_df.columns = ['Gene1', 'Gene2', 'Correlation']
threshold = 0.7
edges_filtered = edges_df[edges_df["Correlation"] > threshold]
G = nx.Graph()
G.add_edges_from(edges_filtered[["Gene1", "Gene2"]].values)
largest_cc = max(nx.connected_components(G), key=len)
G_sub = G.subgraph(largest_cc)
plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G_sub, seed=42)
nx.draw(G_sub, pos, with_labels=True, node_color='lightblue',
        edge_color='gray', font_size=10, node_size=1000)
plt.title(f"Co-expression Network (Top {top_n} Genes, r > {threshold})")
plt.show()
edges_filtered.to_csv("coexpression_edges_for_cytoscape.csv", index=False)
sns.clustermap(correlation_matrix, cmap='coolwarm', figsize=(12, 12))
plt.title(f"Hierarchical Clustering of Top {top_n} Genes")
plt.show()

import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
file_path = "/content/expression_with_gene_symbols.csv"
expression_df = pd.read_csv(file_path, index_col=0)
max_genes = expression_df.shape[1]
top_n = min(280, max_genes)  # cap at total available
gene_variances = expression_df.var(axis=0)
top_genes = gene_variances.sort_values(ascending=False).head(top_n).index
filtered_df = expression_df[top_genes]
correlation_matrix = filtered_df.corr()
corr_pairs = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
edges_df = corr_pairs.stack().reset_index()
edges_df.columns = ['Gene1', 'Gene2', 'Correlation']

threshold = 0.7
edges_filtered = edges_df[edges_df["Correlation"] > threshold]
G = nx.Graph()
G.add_edges_from(edges_filtered[["Gene1", "Gene2"]].values)
if nx.is_empty(G):
    raise ValueError("No strong co-expressed genes found. Try lowering the threshold.")
largest_cc = max(nx.connected_components(G), key=len)
G_sub = G.subgraph(largest_cc).copy()
pos_3d = nx.spring_layout(G_sub, dim=3, seed=42)
node_x, node_y, node_z = [], [], []
for node in G_sub.nodes():
    x, y, z = pos_3d[node]
    node_x.append(x)
    node_y.append(y)
    node_z.append(z)
edge_x, edge_y, edge_z = [], [], []
for edge in G_sub.edges():
    x0, y0, z0 = pos_3d[edge[0]]
    x1, y1, z1 = pos_3d[edge[1]]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]
    edge_z += [z0, z1, None]
fig = go.Figure()
fig.add_trace(go.Scatter3d(
    x=edge_x, y=edge_y, z=edge_z,
    mode='lines',
    line=dict(color='gray', width=2),
    hoverinfo='none'
))
fig.add_trace(go.Scatter3d(
    x=node_x, y=node_y, z=node_z,
    mode='markers+text',
    text=list(G_sub.nodes),
    textposition="top center",
    marker=dict(
        size=6,
        color='lightblue',
        line=dict(width=0.5, color='darkblue')
    ),
    hoverinfo='text'
))
fig.update_layout(
    title=f"3D Co-expression Network (Top {top_n} Genes, r > {threshold})",
    margin=dict(l=0, r=0, b=0, t=40),
    showlegend=False,
    scene=dict(
        xaxis=dict(showbackground=False),
        yaxis=dict(showbackground=False),
        zaxis=dict(showbackground=False)
    )
)

fig.show()

!pip install python-louvain

import community.community_louvain as community_louvain

# Apply Louvain clustering
partition = community_louvain.best_partition(G_sub)

# Add cluster info to nodes
nx.set_node_attributes(G_sub, partition, 'module')

# Visualize Louvain Clusters
plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G_sub, seed=42)
colors = [partition[node] for node in G_sub.nodes()]
nx.draw(G_sub, pos, node_color=colors, with_labels=True, cmap=plt.cm.tab20, node_size=1000, edge_color='gray')
plt.title("Louvain Gene Modules in Co-expression Network")
plt.show()

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

# Load original co-expression network G_sub
# (already filtered by correlation threshold and connected component)

# Step 1: Compute degree centrality
degree_centrality = nx.degree_centrality(G_sub)
centrality_df = pd.DataFrame({
    'Gene': list(degree_centrality.keys()),
    'Degree': list(degree_centrality.values())
}).sort_values(by='Degree', ascending=False)

# Step 2: Select top N hub genes
top_n_hubs = 20
hub_genes = set(centrality_df.head(top_n_hubs)['Gene'])

# Step 3: Create subgraph with only hub genes and their mutual edges
hub_subgraph = G_sub.subgraph(hub_genes)

# Step 4: Visualize
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(hub_subgraph, seed=42)
nx.draw(hub_subgraph, pos, with_labels=True, node_color='skyblue',
        node_size=1200, edge_color='gray', font_size=10)
plt.title(f"Network of Top {top_n_hubs} Central Genes")
plt.show()

# Step 5: Export if needed
nx.write_edgelist(hub_subgraph, "hub_gene_network.edgelist")

import networkx as nx
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ---- Step 1: Use your existing graph (G_sub from co-expression pipeline)
# If not already created, build G_sub using NetworkX from filtered correlations

# ---- Step 2: Centrality + Select Top Genes
degree_centrality = nx.degree_centrality(G_sub)
centrality_df = pd.DataFrame({
    'Gene': list(degree_centrality.keys()),
    'Degree': list(degree_centrality.values())
}).sort_values(by='Degree', ascending=False)

top_n = 280  # adjust as needed
hub_genes = set(centrality_df.head(top_n)['Gene'])

hub_subgraph = G_sub.subgraph(hub_genes)

# ---- Step 3: Get 3D layout
pos = nx.spring_layout(hub_subgraph, dim=3, seed=42)
xyz = np.array([pos[v] for v in hub_subgraph.nodes()])
node_x, node_y, node_z = xyz[:,0], xyz[:,1], xyz[:,2]

# ---- Step 4: Build Edges for Plotly
edge_x = []
edge_y = []
edge_z = []
for edge in hub_subgraph.edges():
    x0, y0, z0 = pos[edge[0]]
    x1, y1, z1 = pos[edge[1]]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]
    edge_z += [z0, z1, None]

# ---- Step 5: Build Node Trace
node_trace = go.Scatter3d(
    x=node_x, y=node_y, z=node_z,
    mode='markers+text',
    text=list(hub_subgraph.nodes()),
    textposition="top center",
    marker=dict(
        size=8,
        color=[degree_centrality[node] for node in hub_subgraph.nodes()],
        colorscale='Viridis',
        colorbar=dict(title='Degree Centrality'),
        line=dict(width=1, color='black')
    )
)

# ---- Step 6: Build Edge Trace
edge_trace = go.Scatter3d(
    x=edge_x, y=edge_y, z=edge_z,
    mode='lines',
    line=dict(color='gray', width=2),
    hoverinfo='none'
)

# ---- Step 7: Combine Plot
fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=f"3D Co-expression Network (Top {top_n} Central Genes)",
                    titlefont_size=16,
                    showlegend=False,
                    margin=dict(l=0, r=0, b=0, t=50),
                    scene=dict(
                        xaxis=dict(showbackground=False),
                        yaxis=dict(showbackground=False),
                        zaxis=dict(showbackground=False)
                    )
                ))
fig.show()

# ----------------------------
# STEP 6.5: Compute Centrality & Rank Genes
# ----------------------------
# Degree centrality
centrality = nx.degree_centrality(G_sub)

# Convert to DataFrame for ranking
centrality_df = pd.DataFrame.from_dict(centrality, orient='index', columns=['DegreeCentrality'])
centrality_df = centrality_df.sort_values(by='DegreeCentrality', ascending=False)

# Save to CSV
centrality_df.to_csv("centrality_ranking.csv")

# Optional: Print top 10 central genes
print("\nTop 10 Central Genes Based on Degree Centrality:")
print(centrality_df.head(280))

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import community.community_louvain as community_louvain

# -------------------------
# FIXED GLOBAL PARAMETERS
# -------------------------
TOP_N_GENES = 280
CORR_THRESHOLD = 0.7
RANDOM_SEED = 42
file_path = "/content/expression_with_gene_symbols.csv"
df = pd.read_csv(file_path, index_col=0)

print("Samples:", df.shape[0])
print("Genes:", df.shape[1])
gene_variances = df.var(axis=0)
top_genes = gene_variances.sort_values(ascending=False).head(TOP_N_GENES).index
df_filtered = df[top_genes]

print("Selected genes:", df_filtered.shape[1])
corr_matrix = df_filtered.corr(method="pearson")
G = nx.Graph()

for i in corr_matrix.columns:
    for j in corr_matrix.columns:
        if i < j:
            corr_value = corr_matrix.loc[i, j]
            if corr_value > CORR_THRESHOLD:
                G.add_edge(i, j, weight=corr_value)

print("Network nodes:", G.number_of_nodes())
print("Network edges:", G.number_of_edges())
largest_cc = max(nx.connected_components(G), key=len)
G_core = G.subgraph(largest_cc).copy()

print("Core network nodes:", G_core.number_of_nodes())
print("Core network edges:", G_core.number_of_edges())
degree_centrality = nx.degree_centrality(G_core)
betweenness_centrality = nx.betweenness_centrality(G_core, weight='weight')
eigenvector_centrality = nx.eigenvector_centrality(G_core, weight='weight', max_iter=2000)

centrality_df = pd.DataFrame({
    "Gene": list(G_core.nodes()),
    "DegreeCentrality": [degree_centrality[g] for g in G_core.nodes()],
    "BetweennessCentrality": [betweenness_centrality[g] for g in G_core.nodes()],
    "EigenvectorCentrality": [eigenvector_centrality[g] for g in G_core.nodes()]
})

centrality_df["CombinedScore"] = (
    centrality_df["DegreeCentrality"] +
    centrality_df["BetweennessCentrality"] +
    centrality_df["EigenvectorCentrality"]
)

centrality_df = centrality_df.sort_values("CombinedScore", ascending=False)
centrality_df.to_csv("ranked_genes_by_centrality.csv", index=False)

centrality_df.head(10)
partition = community_louvain.best_partition(G_core, weight='weight', random_state=RANDOM_SEED)
nx.set_node_attributes(G_core, partition, 'module')
plt.figure(figsize=(12, 10))
pos_2d = nx.spring_layout(G_core, seed=RANDOM_SEED, weight='weight')

node_colors = [partition[node] for node in G_core.nodes()]

nx.draw(
    G_core, pos_2d,
    node_color=node_colors,
    cmap=plt.cm.tab20,
    node_size=900,
    edge_color="gray",
    with_labels=True,
    font_size=8
)

plt.title("Alzheimer’s Gene Co-expression Network (Louvain Modules)")
plt.show()
pos_3d = nx.spring_layout(G_core, dim=3, seed=RANDOM_SEED, weight='weight')

node_x, node_y, node_z = [], [], []
for node in G_core.nodes():
    x, y, z = pos_3d[node]
    node_x.append(x)
    node_y.append(y)
    node_z.append(z)

edge_x, edge_y, edge_z = [], [], []
for u, v in G_core.edges():
    x0, y0, z0 = pos_3d[u]
    x1, y1, z1 = pos_3d[v]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]
    edge_z += [z0, z1, None]

fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=edge_x, y=edge_y, z=edge_z,
    mode="lines",
    line=dict(color="gray", width=2),
    hoverinfo="none"
))

fig.add_trace(go.Scatter3d(
    x=node_x, y=node_y, z=node_z,
    mode="markers+text",
    text=list(G_core.nodes()),
    textposition="top center",
    marker=dict(
        size=7,
        color=[eigenvector_centrality[g] for g in G_core.nodes()],
        colorscale="Viridis",
        colorbar=dict(title="Eigenvector Centrality"),
        line=dict(width=0.5, color="black")
    )
))

fig.update_layout(
    title="3D Alzheimer’s Gene Co-expression Network",
    showlegend=False,
    margin=dict(l=0, r=0, t=40, b=0),
    scene=dict(
        xaxis=dict(showbackground=False),
        yaxis=dict(showbackground=False),
        zaxis=dict(showbackground=False)
    )
)

fig.show()
edges_export = pd.DataFrame([
    (u, v, d['weight']) for u, v, d in G_core.edges(data=True)
], columns=["Source", "Target", "Weight"])

edges_export.to_csv("coexpression_edges_cytoscape.csv", index=False)

# ==============================
# Gene Centrality Ranking Script
# ==============================

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler

# ------------------------------
# 1. Load Gene Expression Data
# ------------------------------
# Rows    -> Sample IDs
# Columns -> Genes

df = pd.read_csv(
    "/content/expression_with_gene_symbols.csv",
    index_col=0
)

print("Original data shape:", df.shape)

# ------------------------------
# 2. Data Cleaning & Normalization
# ------------------------------

# Drop samples with missing values
df = df.dropna(axis=0)

# Assume data is already log-transformed
df_log = df.copy()

# Z-score normalization across samples (per gene)
scaler = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df_log),
    index=df_log.index,
    columns=df_log.columns
)

print("Preprocessed data shape:", df_scaled.shape)

# ------------------------------
# 3. Build Co-expression Network
# ------------------------------

print("Computing correlation matrix...")

# Gene × Gene correlation matrix
corr_matrix = df_scaled.corr()

# Remove self-correlations
np.fill_diagonal(corr_matrix.values, 0)

correlation_threshold = 0.6
adjacency = corr_matrix.abs() >= correlation_threshold

G = nx.Graph()
G.add_nodes_from(corr_matrix.columns)

for i, gene1 in enumerate(corr_matrix.columns):
    for j, gene2 in enumerate(corr_matrix.columns[i+1:], i+1):
        if adjacency.iat[i, j]:
            G.add_edge(
                gene1,
                gene2,
                weight=abs(corr_matrix.iat[i, j])
            )

print(f"Network created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

if G.number_of_edges() == 0:
    raise ValueError("No edges found. Lower correlation threshold.")

# ------------------------------
# 4. Centrality Computation
# ------------------------------

degree_centrality = nx.degree_centrality(G)

betweenness_centrality = nx.betweenness_centrality(
    G,
    weight="weight",
    normalized=True
)

eigenvector_centrality = nx.eigenvector_centrality(
    G,
    weight="weight",
    max_iter=2000
)

# ------------------------------
# 5. Centrality Table
# ------------------------------

centrality_df = pd.DataFrame({
    "Gene": list(G.nodes()),
    "DegreeCentrality": [degree_centrality[g] for g in G.nodes()],
    "BetweennessCentrality": [betweenness_centrality[g] for g in G.nodes()],
    "EigenvectorCentrality": [eigenvector_centrality[g] for g in G.nodes()]
})

# ------------------------------
# 6. Ranking
# ------------------------------

centrality_df["DegreeRank"] = centrality_df["DegreeCentrality"].rank(ascending=False)
centrality_df["BetweennessRank"] = centrality_df["BetweennessCentrality"].rank(ascending=False)
centrality_df["EigenvectorRank"] = centrality_df["EigenvectorCentrality"].rank(ascending=False)

centrality_df["FinalRankScore"] = centrality_df[
    ["DegreeRank", "BetweennessRank", "EigenvectorRank"]
].mean(axis=1)

centrality_df = centrality_df.sort_values("FinalRankScore")

# ------------------------------
# 7. Save Results
# ------------------------------

centrality_df.to_csv("ranked_genes_by_centrality.csv", index=False)
nx.write_gml(G, "gene_coexpression_network.gml")

print("Top 10 ranked genes:")
print(centrality_df.head(10))
print("Files saved:")
print("- ranked_genes_by_centrality.csv")
print("- gene_coexpression_network.gml")

# ==============================
# Gene Centrality Ranking Script
# ==============================

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler

# ------------------------------
# 1. Load Gene Expression Data
# ------------------------------
# Rows    -> Sample IDs
# Columns -> Genes

df = pd.read_csv(
    "/content/expression_with_gene_symbols.csv",
    index_col=0
)

print("Original data shape:", df.shape)

# ------------------------------
# 2. Data Cleaning & Normalization
# ------------------------------

df = df.dropna(axis=0)

# Assume data already log-transformed
df_log = df.copy()

scaler = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df_log),
    index=df_log.index,
    columns=df_log.columns
)

print("Preprocessed data shape:", df_scaled.shape)

# ------------------------------
# 3. Build Co-expression Network
# ------------------------------

corr_matrix = df_scaled.corr()
np.fill_diagonal(corr_matrix.values, 0)

correlation_threshold = 0.6
adjacency = corr_matrix.abs() >= correlation_threshold

G = nx.Graph()
G.add_nodes_from(corr_matrix.columns)

for i, gene1 in enumerate(corr_matrix.columns):
    for j, gene2 in enumerate(corr_matrix.columns[i+1:], i+1):
        if adjacency.iat[i, j]:
            G.add_edge(
                gene1,
                gene2,
                weight=abs(corr_matrix.iat[i, j])
            )

print(f"Network created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

if G.number_of_edges() == 0:
    raise ValueError("No edges found. Lower correlation threshold.")

# ------------------------------
# 4. Centrality Computation
# ------------------------------

degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G, weight="weight", normalized=True)
eigenvector_centrality = nx.eigenvector_centrality(G, weight="weight", max_iter=2000)

# ------------------------------
# 5. Centrality Table
# ------------------------------

centrality_df = pd.DataFrame({
    "Gene": list(G.nodes()),
    "DegreeCentrality": [degree_centrality[g] for g in G.nodes()],
    "BetweennessCentrality": [betweenness_centrality[g] for g in G.nodes()],
    "EigenvectorCentrality": [eigenvector_centrality[g] for g in G.nodes()]
})

# ------------------------------
# 6. Final Score & Ranking
# ------------------------------

# Normalize centralities before combining
for col in ["DegreeCentrality", "BetweennessCentrality", "EigenvectorCentrality"]:
    centrality_df[col] = (
        centrality_df[col] - centrality_df[col].min()
    ) / (
        centrality_df[col].max() - centrality_df[col].min()
    )

# Final combined score
centrality_df["FinalScore"] = centrality_df[
    ["DegreeCentrality", "BetweennessCentrality", "EigenvectorCentrality"]
].mean(axis=1)

# Rank ONLY by final score
centrality_df["FinalRank"] = centrality_df["FinalScore"].rank(
    ascending=False,
    method="dense"
)

centrality_df = centrality_df.sort_values("FinalRank")

# ------------------------------
# 7. Save Results
# ------------------------------

centrality_df.to_csv("ranked_genes_by_final_score.csv", index=False)
nx.write_gml(G, "gene_coexpression_network.gml")

print("Top 10 ranked genes (by final score):")
print(centrality_df.head(10))
print("Files saved:")
print("- ranked_genes_by_final_score.csv")
print("- gene_coexpression_network.gml")

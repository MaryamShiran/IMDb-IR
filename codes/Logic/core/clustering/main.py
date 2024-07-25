import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import fasttext


import sys
import os
import string
# Add the root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(project_root)

# Add the directory containing indexes_enum.py to sys.path
DimensionReduction_model_dir = os.path.join(project_root, 'Logic', 'core','clustering','dimension_reduction' )
sys.path.append(DimensionReduction_model_dir)


#from ..word_embedding.fasttext_data_loader import FastTextDataLoader
#from ..word_embedding.fasttext_model import FastText
from Logic.core.clustering.dimension_reduction import DimensionReduction


clustering_metrics_dir = os.path.join(project_root, 'Logic', 'core','clustering','clustering_metrics' )
sys.path.append(clustering_metrics_dir)



from Logic.core.clustering.clustering_metrics import ClusteringMetrics


clustering_utils_dir = os.path.join(project_root, 'Logic', 'core','clustering','clustering_utils' )
sys.path.append(clustering_utils_dir)




from Logic.core.clustering.clustering_utils import ClusteringUtils
CU = ClusteringUtils()
CM = ClusteringMetrics()



# Main Function: Clustering Tasks

# 0. Embedding Extraction
#  Using the previous preprocessor and fasttext model, collect all the embeddings of our data and store them.
loaded = np.load("C:/Users/Haghani/Desktop/mir_proj/mir_project/arrays.npz",allow_pickle=True)
model = fasttext.load_model("C:/Users/Haghani/Desktop/mir_proj/mir_project/FastText_model.bin")

X = loaded['arr1']
y = loaded['arr2']
embeddings = []


for sentence in X:
    embeddings.append(model.get_sentence_vector(sentence))

embeddings = np.array(embeddings)



# 1. Dimension Reduction
# Perform Principal Component Analysis (PCA):
#     - Reduce the dimensionality of features using PCA. (you can use the reduced feature afterward or use to the whole embeddings)
#     - Find the Singular Values and use the explained_variance_ratio_ attribute to determine the percentage of variance explained by each principal component.
#     - Draw plots to visualize the results.

dimension_reduction = DimensionReduction()
singular_values , explained_variance_ratio =  dimension_reduction.wandb_plot_explained_variance_by_components(embeddings, 'clustering', 'explained_variance_by_components')


#  Implement t-SNE (t-Distributed Stochastic Neighbor Embedding):
#     - Create the convert_to_2d_tsne function, which takes a list of embedding vectors as input and reduces the dimensionality to two dimensions using the t-SNE method.
#     - Use the output vectors from this step to draw the diagram.



embeddings=dimension_reduction.wandb_plot_2d_tsne(embeddings, 'Clustering', '2D_t-SNE')


# 2. Clustering
## K-Means Clustering
#   Implement the K-means clustering algorithm from scratch.
#   Create document clusters using K-Means.
#   Run the algorithm with several different values of k.
#    For each run:
#     - Determine the genre of each cluster based on the number of documents in each cluster.
#     - Draw the resulting clustering using the two-dimensional vectors from the previous section.
#     - Check the implementation and efficiency of the algorithm in clustering similar documents.
#   : Draw the silhouette score graph for different values of k and perform silhouette analysis to choose the appropriate k.
#   : Plot the purity value for k using the labeled data and report the purity value for the final k. (Use the provided functions in utilities)

print("part1")

for k in range(2,10):
    CU.visualize_kmeans_clustering_wandb(embeddings, k, 'Clustering', 'Kmeans_Clustering')

    print("part1.",k)


CU.plot_kmeans_cluster_scores(embeddings, y, [k for k in range(2,10)], 'Clustering', 'Scores_Kmeans_Cluster')

#print("part1.9999.")


CU.visualize_elbow_method_wcss(embeddings, [k for k in range(2,10)], 'Clustering', 'WCSS_Elbow_Method')


print("part2")

## Hierarchical Clustering
#    Perform hierarchical clustering with all different linkage methods.
#    Visualize the results.
CU.wandb_plot_hierarchical_clustering_dendrogram(embeddings, 'Clustering', 'complete', 'Dendrogram-complete')

print("part3")


CU.wandb_plot_hierarchical_clustering_dendrogram(embeddings, 'Clustering', 'average', 'Dendrogram-average')
CU.wandb_plot_hierarchical_clustering_dendrogram(embeddings, 'Clustering', 'single', 'Dendrogram-single')
CU.wandb_plot_hierarchical_clustering_dendrogram(embeddings, 'Clustering', 'ward', 'Dendrogram-ward')




# 3. Evaluation
#    Using clustering metrics, evaluate how well your clustering method is performing.


methods=['Kmeans','Single','Complete','Average','Ward']

labels={}
_, _labels = CU.cluster_kmeans(embeddings, 4)
labels['Kmeans']=_labels
labels['Single'] = CU.cluster_hierarchical_single(embeddings)
labels['Complete'] = CU.cluster_hierarchical_complete(embeddings)
labels['Average'] = CU.cluster_hierarchical_average(embeddings)
labels['Ward'] = CU.cluster_hierarchical_ward(embeddings)






for key in labels.keys():
    print("results for method", key, ":")
    silhouette = CM.silhouette_score(embeddings, labels[key])
    purity = CM.purity_score(y, labels[key])
    rs = CM.adjusted_rand_score(y, labels[key])
    print("Purity Score:", purity)
    print("Adjusted Rand Score:", rs)   
    print("Silhouette Score:", silhouette)
 


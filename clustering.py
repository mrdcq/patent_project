import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from clustering import *
from sklearn.metrics import silhouette_score
from joblib import Parallel, delayed
import umap

# DATA PRE-PROCESSING

def delete_stopwords(text, stopwords):
    tokens = text.split()
    words = [word for word in tokens if word not in stopwords]
    return " ".join(words)  

def preprocess_corpus_text(text):
    """
    Given a string, cleans the string by:
    - Converting to lowercase
    - Tokenizing the text (splitting text to words)
    - Removing the punctuations and special characters
    - Removing predetermined set of stopwords (from nltk)
    - Lemmatozing words to their base form (for example, delete plurals)
    """
    if pd.isna(text): 
        return ""
    
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation & special characters
    text = re.sub(r'\b\d+\b', '', text) # Remove standalone numbers
    words = word_tokenize(text)  # Tokenization
    words = [word for word in words if word not in stopwords.words('english') and len(word) > 1]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words] 
    return " ".join(words)  

def corpus_cleaning(df):
    """
    Cleans all the corpuses of a given DataFrame, needs to contain a "corpus" column.
    Returns:
    - DataFrame with an added "cleaned_corpus" column.
    """
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download("wordnet")

    df["cleaned_corpus"] = Parallel(n_jobs=-1)(delayed(preprocess_corpus_text)(text) for text in df["corpus"])
    return df

# VECTORIZATION UTILS

def vectorize_word2vec(list_of_docs, model):
    """Generate vectors for list of documents using a Word Embedding

    Args:
        list_of_docs: List of documents
        model: Gensim's Word Embedding

    Returns:
        List of document vectors
    """
    features = []

    for tokens in list_of_docs:
        zero_vector = np.zeros(model.vector_size)
        vectors = []
        for token in tokens:
            if token in model.wv:
                try:
                    vectors.append(model.wv[token])
                except KeyError:
                    continue
        if vectors:
            vectors = np.asarray(vectors)
            avg_vec = vectors.mean(axis=0)
            features.append(avg_vec)
        else:
            features.append(zero_vector)
    return features

## K-MEANS CLUSTERING

def elbow_method(X, max_range):
    """
    Function to present the result of the elbow method for choosing 
    k in the K-Means clustering algorithm. 
    Returns:
        - The Within-Cluster Sum of Squares per amount of cluster
    """

    k_values = range(2, max_range+1)  
    inertia_values = []

    inertia_values = Parallel(n_jobs=-1)(
        delayed(lambda k: KMeans(n_clusters=k, random_state=42, n_init=10).fit(X).inertia_)(k)
        for k in k_values
    )

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertia_values, marker='o', linestyle='--', color='b')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
    plt.title("Elbow Method for Optimal k")
    plt.show()

    return inertia_values

def compute_silhouette_method(X, max_range, n_jobs=-1):
    """
    Computes the Silhouette Score for different values of k in parallel.

    Parameters:
    - X: Feature matrix (TF-IDF or Word2Vec embeddings)
    - max_range: Maximum number of clusters to test
    - n_jobs: Number of parallel jobs (-1 uses all available CPUs)

    Returns:
    - optimal_k_silhouette: The k value with the highest silhouette score.
    - silhouette_scores: List of silhouette scores for each k.
    """
    k_values = range(2, max_range + 1)  

    def compute_silhouette(k):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        return silhouette_score(X, kmeans.labels_) if k > 1 else None

    silhouette_scores = Parallel(n_jobs=n_jobs)(
        delayed(compute_silhouette)(k) for k in k_values
    )

    valid_k_values = k_values[:len(silhouette_scores)]
    silhouette_scores = [score for score in silhouette_scores if score is not None]

    plt.figure(figsize=(8, 5))
    plt.plot(valid_k_values, silhouette_scores, marker='s', linestyle='-', color='g')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score for Optimal k")
    plt.show()

    optimal_k_silhouette = valid_k_values[np.argmax(silhouette_scores)]

    print(f"Optimal number of clusters based on Silhouette Score: {optimal_k_silhouette}")

    return optimal_k_silhouette, silhouette_scores

def silhouette_method(X, max_range):
    """
    Function to implement the Silhouette Score method for choosing K in the K-means algorithm. 
    Returns:
        - Optimal number of clusters (highest slihouette score)
        - List of silhouette scores per amount of clusters
    """
    silhouette_scores = []
    k_values = range(2, max_range + 1)  

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        if k > 1:
            silhouette_scores.append(silhouette_score(X, kmeans.labels_))

    plt.figure(figsize=(8, 5))
    plt.plot(k_values[:len(silhouette_scores)], silhouette_scores, marker='s', linestyle='-', color='g')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score for Optimal k")
    plt.show()

    optimal_k_silhouette = k_values[:len(silhouette_scores)][np.argmax(silhouette_scores)]

    print(f"Optimal number of clusters based on Silhouette Score: {optimal_k_silhouette}")
    
    return optimal_k_silhouette, silhouette_scores

def visualize_clusters_pca(X_vectors, labels, title="UMAP Visualization of K-Means Clusters", ax = None):

    reducer = umap.UMAP(n_components=2)
    X_umap = reducer.fit_transform(X_vectors)

    df_umap = pd.DataFrame(X_umap, columns=["UMAP1", "UMAP2"])
    df_umap["Cluster"] = labels

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))

    unique_clusters = np.unique(labels)

    colors = sns.color_palette("Set2", len(unique_clusters))

    for cluster, color in zip(unique_clusters, colors):
        cluster_points = df_umap[df_umap["Cluster"] == cluster]
        ax.scatter(cluster_points["UMAP1"], cluster_points["UMAP2"],
               label=f"Cluster {cluster}", color=color, alpha=0.6)

    ax.set_xlabel("UMAP Component 1")
    ax.set_ylabel("UMAP Component 2")
    ax.set_title(title)
    ax.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.show()

def visualize_dis_clusters(labels, n_clusters, ax = None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(labels, bins=np.arange(-0.5, n_clusters, 1), edgecolor='black', alpha=0.7)
    ax.set_xticks(range(n_clusters))
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Patents' Corpus")
    ax.set_title("Distribution of Patents' Corpus Across Clusters")

def top_words_tf(labels, X, vectorizer, kmeans=None, hdbscan=None, top_n=10):
    """
    Extracts the top N keywords for each cluster based on TF-IDF scores.

    Parameters:
    - labels: Cluster labels assigned to the data.
    - X: TF-IDF matrix (sparse matrix).
    - vectorizer: Fitted TF-IDF vectorizer.
    - kmeans: Trained KMeans model (if applicable).
    - hdbscan: Trained HDBSCAN model (if applicable).
    - top_n: Number of top words to extract per cluster.

    Returns:
    - Dictionary mapping each cluster to its top N keywords.
    """
    terms = vectorizer.get_feature_names_out()
    cluster_keywords = {}

    if kmeans:
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
        cluster_keywords = {i: [terms[ind] for ind in order_centroids[i, :top_n]] for i in range(kmeans.n_clusters)}
    
    else:
        df_tfidf = pd.DataFrame(X.toarray(), columns=terms)
        df_tfidf["Cluster"] = labels

        for cluster in np.unique(labels):
            if cluster == -1:
                continue
            cluster_data = df_tfidf[df_tfidf["Cluster"] == cluster].drop(columns=["Cluster"])
            top_keywords = cluster_data.mean(axis=0).nlargest(top_n).index.tolist()
            cluster_keywords[cluster] = top_keywords

    return cluster_keywords

def top_words_word2vec(labels, X, word2vec_model, kmeans, top_n=10):
    cluster_keywords = {}

    for i in range(kmeans.n_clusters):
        cluster_center = kmeans.cluster_centers_[i]
        similar_words = word2vec_model.wv.similar_by_vector(cluster_center, topn=top_n)
        top_words = [word for word, _ in similar_words]
        cluster_keywords[i] = top_words

    return cluster_keywords

def k_means(X, k, df, plus, vectorizer = None, word2vec_model=None):
    df_copy = df.copy()
    init_method = "k-means++" if plus else "random"

    kmeans = KMeans(n_clusters=k, init=init_method, random_state=40, n_init=10)
    kmeans.fit(X)

    labels = kmeans.labels_
    num_iterations = kmeans.n_iter_
    print(f"K-Means converged in {num_iterations} iterations.")

    df_clusters = pd.DataFrame({
        "patent_id": df["patent_id"],
        "Cluster": labels
    })

    distances = np.linalg.norm(X - kmeans.cluster_centers_[labels], axis=1)
    df_clusters["DistanceToCentroid"] = distances
    
    df_copy = df_copy.merge(df_clusters, on="patent_id", how="left")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    visualize_dis_clusters(labels, k, ax=axes[0])

    if vectorizer is None and word2vec_model is None:
        scatter = axes[1].scatter(X[:, 0], X[:, 1], c=labels, s=0.8, cmap='Spectral')
        axes[1].set_title("UMAP Projection of Patents' Corpus")
        axes[1].set_xlabel("UMAP Dimension 1")
        axes[1].set_ylabel("UMAP Dimension 2")
        fig.colorbar(scatter, ax=axes[1], label="Cluster")
    
    else:
        visualize_clusters_pca(X, labels, ax=axes[1])

    plt.tight_layout()
    plt.show()

    if vectorizer is not None:
        cluster_keywords = top_words_tf(labels, X, vectorizer, kmeans)

        for cluster, keywords in cluster_keywords.items():
            print(f"Cluster {cluster}: {', '.join(keywords)}")

        return df_copy, cluster_keywords
    
    elif word2vec_model is not None:
        cluster_keywords = top_words_word2vec(labels, X, word2vec_model, kmeans)

        for cluster, keywords in cluster_keywords.items():
            print(f"Cluster {cluster}: {', '.join(keywords)}")
        
        return df_copy, cluster_keywords
    
    return df_copy


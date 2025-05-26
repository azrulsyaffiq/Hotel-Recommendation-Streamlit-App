import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans

# Load data and vectorizer
df = pd.read_pickle("hotel_data_southeastasia.pkl", compression='gzip')
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Drop rows with missing critical fields
df = df.dropna(subset=['HotelName', 'cityName', 'combined_features', 'HotelRating'])

# TF-IDF transformation
tfidf_matrix = vectorizer.transform(df['combined_features'])

# Fit KMeans on all hotel TF-IDF vectors
NUM_CLUSTERS = 20  # you can tune this
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
cluster_labels = kmeans.fit_predict(tfidf_matrix)

# Add cluster info to df
df['Cluster'] = cluster_labels

def recommend_similar_hotels_priority(hotel_name, city_name, radius_km=10, top_n = 5):
    # Get selected hotel info
    selected_hotel = df[df['HotelName'] == hotel_name].iloc[0]
    selected_rating = selected_hotel['HotelRating']
    
    # Step 1: Filter by city (or radius if using lat/lon)
    city_hotels = df[df['cityName'].str.lower() == city_name.lower()]
    
    # Step 2: Prioritize same rating hotels
    same_rating_hotels = city_hotels[city_hotels['HotelRating'] == selected_rating]

    # Step 3: Fallback to nearby hotels with similar but lower/higher ratings (optional)
    if len(same_rating_hotels) < top_n:
        other_hotels = city_hotels[city_hotels['HotelRating'] != selected_rating]
        other_hotels = other_hotels.sort_values(by='RatingValue', ascending=False)
        fallback_hotels = pd.concat([same_rating_hotels, other_hotels]).head(top_n + 1)
    else:
        fallback_hotels = same_rating_hotels

    # Step 4: Compute TF-IDF similarity within these hotels
    tfidf_features = vectorizer.transform(fallback_hotels['combined_features'])
    selected_vector = vectorizer.transform([selected_hotel['combined_features']])
    cosine_sim = cosine_similarity(selected_vector, tfidf_features).flatten()

    fallback_hotels = fallback_hotels.copy()
    fallback_hotels['Similarity'] = cosine_sim
    fallback_hotels = fallback_hotels[fallback_hotels['HotelName'] != hotel_name]


    # Step 5: Sort by rating and similarity
    results = fallback_hotels.sort_values(by=['RatingValue', 'Similarity'], ascending=[False, False])

    return results.head(top_n)

def avg_similarity(hotel_name, city_name, top_n=5):
    selected_hotel = df[df['HotelName'] == hotel_name].iloc[0]
    selected_rating = selected_hotel['HotelRating']
    
    # Step 1: Filter by city
    city_hotels = df[df['cityName'].str.lower() == city_name.lower()]
    same_rating_hotels = city_hotels[city_hotels['HotelRating'] == selected_rating]

    if len(same_rating_hotels) < top_n:
        other_hotels = city_hotels[city_hotels['HotelRating'] != selected_rating]
        other_hotels = other_hotels.sort_values(by='RatingValue', ascending=False)
        fallback_hotels = pd.concat([same_rating_hotels, other_hotels]).head(top_n + 1)
    else:
        fallback_hotels = same_rating_hotels

    # Vectorize
    tfidf_features = vectorizer.transform(fallback_hotels['combined_features'])
    selected_vector = vectorizer.transform([selected_hotel['combined_features']])
    cosine_similarities = cosine_similarity(selected_vector, tfidf_features).flatten()

    # Exclude the selected hotel itself
    fallback_hotels = fallback_hotels.copy()
    fallback_hotels['Similarity'] = cosine_similarities
    fallback_hotels = fallback_hotels[fallback_hotels['HotelName'] != hotel_name]

    # Return average similarity of top-N
    top_similarities = fallback_hotels.sort_values(by='Similarity', ascending=False).head(top_n)['Similarity']
    return top_similarities.mean() if len(top_similarities) > 0 else 0


# Offline Evaluation
def evaluate_model(top_k=5, sample_size=100):
    precision_list = []
    recall_list = []
    f1_list = []
    average_precisions = []
    avg_sim_list = []
    cluster_match_ratios = [] 

    sampled_hotels = df.sample(n=min(sample_size, len(df)), random_state=42)

    for idx, row in tqdm(sampled_hotels.iterrows(), total=len(sampled_hotels)):
        hotel_name = row['HotelName']
        city = row['cityName']
        true_rating = row['HotelRating']
        selected_cluster = row['Cluster']

        try:
            recommended = recommend_similar_hotels_priority(hotel_name, city, top_k)
            avg_sim = avg_similarity(hotel_name, city, top_k) 
            avg_sim_list.append(avg_sim)
        except:
            continue

        # Relevant = other hotels in same city with same rating (excluding self)
        relevant = df[
            (df['cityName'].str.lower() == city.lower()) &
            (df['HotelRating'] == true_rating) &
            (df['HotelName'] != hotel_name)
            ]['HotelName'].values

        recommended_names = recommended['HotelName'].values
        hit_list = [1 if hotel in relevant else 0 for hotel in recommended_names]

        # Precision@k
        precision = sum(hit_list) / top_k
        precision_list.append(precision)

        # Recall@k
        total_relevant = len(relevant)
        recall = sum(hit_list) / total_relevant if total_relevant > 0 else 0
        recall_list.append(recall)

        # F1@k
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
        f1_list.append(f1)

        # Average Precision@k
        if any(hit_list):
            cum_sum = 0
            correct = 0
            for i, val in enumerate(hit_list):
                if val == 1:
                    correct += 1
                    cum_sum += correct / (i + 1)
            ap = cum_sum / correct
        else:
            ap = 0
        average_precisions.append(ap)

         # ---- Cluster consistency ----
        match_count = 0
        for hotel in recommended_names:
            rec_cluster = df[df['HotelName'] == hotel]['Cluster'].values[0]
            if rec_cluster == selected_cluster:
                match_count += 1
        cluster_match_ratio = match_count / top_k
        cluster_match_ratios.append(cluster_match_ratio)

    print(f"\n🔎 Offline Evaluation on {len(precision_list)} samples:")
    print(f"🔹 Precision@{top_k}: {np.mean(precision_list):.4f}")
    print(f"🔹 Recall@{top_k}: {np.mean(recall_list):.4f}")
    print(f"🔹 F1-score@{top_k}: {np.mean(f1_list):.4f}")
    print(f"🔹 MAP@{top_k}: {np.mean(average_precisions):.4f}")
    print(f"🔹 Avg Cosine Similarity@{top_k}: {np.mean(avg_sim_list):.4f}")
    print(f"🔹 Cluster Consistency@{top_k}: {np.mean(cluster_match_ratios):.4f}")

# Run evaluation
if __name__ == "__main__":
    evaluate_model(top_k=5, sample_size=100)

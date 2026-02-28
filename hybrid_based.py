import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def hybrid_recommendation(content_data, collab_data, item_name, user_id, top_n=10, alpha=0.5):

    # ---------- CONTENT-BASED ----------
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(content_data['Tags'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    if item_name not in content_data['Name'].values:
        print("Item not found")
        return None

    item_idx = content_data[content_data['Name'] == item_name].index[0]

    content_scores = list(enumerate(cosine_sim[item_idx]))
    content_scores = sorted(content_scores, key=lambda x: x[1], reverse=True)[1:]

    content_dict = {i: score for i, score in content_scores}

    # ---------- COLLABORATIVE ----------
    user_item_matrix = collab_data.pivot_table(
        index="user_id",
        columns="product_id",
        values="rating"
    ).fillna(0)

    if user_id not in user_item_matrix.index:
        print("User not found")
        return None

    similarity = cosine_similarity(user_item_matrix)
    similarity_df = pd.DataFrame(
        similarity,
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )

    similar_users = similarity_df[user_id].sort_values(ascending=False)[1:]

    target_user_ratings = user_item_matrix.loc[user_id]

    collab_scores = {}

    for user in similar_users.index:
        user_ratings = user_item_matrix.loc[user]
        liked_products = user_ratings[user_ratings >= 4]

        for product in liked_products.index:
            if target_user_ratings[product] == 0:
                collab_scores[product] = collab_scores.get(product, 0) + liked_products[product]

    # Normalize collaborative scores
    if collab_scores:
        max_score = max(collab_scores.values())
        collab_scores = {k: v / max_score for k, v in collab_scores.items()}

    # ---------- COMBINE ----------
    hybrid_scores = {}

    for idx in range(len(content_data)):
        content_score = content_dict.get(idx, 0)
        collab_score = collab_scores.get(idx, 0)

        hybrid_scores[idx] = alpha * content_score + (1 - alpha) * collab_score

    # ---------- TOP RESULTS ----------
    sorted_items = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)

    top_items = [i[0] for i in sorted_items[:top_n]]

    return content_data.iloc[top_items][['Name', 'Brand', 'ReviewCount']]
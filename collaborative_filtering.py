import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def recommend_products(data, target_user, top_n=5):

    # 1. Check if user exists
    if target_user not in data["user_id"].values:
        print("User does not exist.")
        return None

    # 2. Create user-item rating matrix
    user_item_matrix = data.pivot_table(
        index="user_id",
        columns="product_id",
        values="rating"
    ).fillna(0)

    # 3. Calculate similarity between users
    similarity = cosine_similarity(user_item_matrix)

    similarity_df = pd.DataFrame(
        similarity,
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )

    # 4. Find top similar users (excluding target user)
    similar_users = similarity_df[target_user].sort_values(ascending=False)[1:]

    # 5. Get products rated by similar users
    target_user_ratings = user_item_matrix.loc[target_user]

    recommended_products = pd.Series()

    for user in similar_users.index:
        user_ratings = user_item_matrix.loc[user]
        
        # 6. Select products liked by similar user (rating >=4)
        liked_products = user_ratings[user_ratings >= 4]

        for product in liked_products.index:
            # 7. Filter products not rated by target user
            if target_user_ratings[product] == 0:
                recommended_products.loc[product] = liked_products[product]

    # 8. Return top recommendations
    recommended_products = recommended_products.sort_values(ascending=False)

    return recommended_products.head(top_n)
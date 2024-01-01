
#############################################
# PROJECT: Hybrid Recommender System
#############################################

# Make a prediction using the item-based and user-based recommender methods for the user whose ID is given.
# Consider 5 suggestions from the user-based model, 5 suggestions from the item-based model, and finally make 10 suggestions from 2 models.

#############################################
# Preparation of Data
#############################################

import pandas as pd
pd.pandas.set_option('display.max_columns', None)
pd.pandas.set_option('display.width', 300)

# Step 1: Read the Movie and Rating data sets.
# movie dataset containing movie name and movie genre information
movie = pd.read_csv('datasets\movie.csv')
movie.iloc
movie.shape


# rating is a dataset containing movie name, 
# votes for the movie and time information
rating = pd.read_csv('datasets\rating.csv')
rating.head()
rating.shape
rating["userId"].nunique()


# Step 2: Add the names and genres of the movies to the rating data set using the movie movie set.
# Only the IDs of the movies voted by users in the rating are available.
# We add the movie names and genres of the ids from the movie data set.
df = movie.merge(rating, how="left", on="movieId")
df.head()
df.shape


# Step 3: Calculate how many people voted for each movie. Remove movies with less than 1000 votes from the data set.
# We calculate how many people voted for each movie.
comment_counts = pd.DataFrame(df["title"].value_counts())
comment_counts

# We keep the names of movies with less than 1000 votes in rare_movies.
# And we subtract it from the data set
rare_movies = comment_counts[comment_counts["title"] <= 1000].index
common_movies = df[~df["title"].isin(rare_movies)]
common_movies.shape


# Step 4: userIDs in the index, movie names in the columns and ratings as values.
# Create a pivot table for the dataframe.

user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
user_movie_df.head()


# Step 5: Let's functionalize all the above operations
def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()


#############################################
# Determining the Movies Watched by the User to Make a Recommendation
#############################################

# Step 1: Choose a random user ID.
random_user = 108170

# Step 2: Create a new dataframe named random_user_df consisting of observation units of the selected user.
random_user_df = user_movie_df[user_movie_df.index == random_user]
random_user_df.head()
random_user_df.shape

# Step 3: Assign the movies voted by the selected user to a list named movies_watched.
movies_watched = random_user_df.columns[random_user_df.notna().any()].to_list()
movies_watched

movie.columns[movie.notna().any()].to_list()

#############################################
# Görev 3: Accessing the Data and IDs of Other Users Watching the Same Movies
#############################################

# Step 1: Select the columns of the movies watched by the selected user from user_movie_df and create a new dataframe named movies_watched_df.
movies_watched_df = user_movie_df[movies_watched]
movies_watched_df.head()
movies_watched_df.shape

# Step 2: Create a new dataframe named user_movie_count, which contains information about how many movies each user watched that the selected user watched.
# And we create a new df.
user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]
user_movie_count.head(5)

# Step 3: We consider those who watched 60 percent or more of the movies voted by the selected user as similar users.
# Create a list named users_same_movies from the ids of these users.
perc = len(movies_watched) * 60 / 100
users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]
len(users_same_movies)



#############################################
# Determining the Users Most Similar to the User to Make a Recommendation
#############################################

# Step 1: Filter the movies_watched_df dataframe to find the IDs of users that are similar to the selected user in the user_same_movies list.
final_df = movies_watched_df[movies_watched_df.index.isin(users_same_movies)]
final_df.head()
final_df.shape

# Step 2: Create a new corr_df dataframe in which the correlations of users with each other will be found.
corr_df = final_df.T.corr().unstack().sort_values()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()

#corr_df[corr_df["user_id_1"] == random_user]



# Step 3: Create a new dataframe named top_users by filtering out users with high correlation (over 0.65) with the selected user.
top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][["user_id_2", "corr"]].reset_index(drop=True)
top_users = top_users.sort_values(by='corr', ascending=False)
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
top_users.shape
top_users.head()

# Step 4: Merge the top_users dataframe with the rating data set
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]
top_users_ratings["userId"].unique()
top_users_ratings.head()



#############################################
# Calculating the Weighted Average Recommendation Score and Finding the Top 5 Movies
#############################################

# Create a new variable called weighted_rating, which is the product of each user's corr and rating values.
top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']
top_users_ratings.head()

# Step 2: Create a new file called recommendation_df, which contains the movie id and the average value of all users' weighted ratings for each movie.
# create dataframe.
recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()
recommendation_df.head()

# Step3: Select the movies with a weighted rating greater than 3.5 in recommendation_df and sort them according to the weighted rating.
# Save the first 5 observations as movies_to_be_recommend.
recommendation_df[recommendation_df["weighted_rating"] > 3.5]
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)


# Adım 4:  Tavsiye edilen 5 filmin isimlerini getiriniz.
movies_to_be_recommend.merge(movie[["movieId", "title"]])["title"][:5]

# 0    Mystery Science Theater 3000: The Movie (1996)
# 1                               Natural, The (1984)
# 2                             Super Troopers (2001)
# 3                         Christmas Story, A (1983)
# 4                       Sonatine (Sonachine) (1993)



#############################################
# Item-Based Recommendation
#############################################

# Make an item-based suggestion based on the name of the movie the user last watched and gave the highest rating.
user = 108170

# Step 1: Read the movie,rating data sets.
movie = pd.read_csv('Modül_4_Tavsiye_Sistemleri/datasets/movie.csv')
rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')

# Step 2: Get the ID of the movie with the most current score among the movies that the user to be recommended gave 5 points.
movie_id = rating[(rating["userId"] == user) & (rating["rating"] == 5.0)].sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]

# Step 3: Filter the user_movie_df dataframe created in the User based recommendation section according to the selected movie id.
movie[movie["movieId"] == movie_id]["title"].values[0]

movie_df = user_movie_df[movie[movie["movieId"] == movie_id]["title"].values[0]]
movie_df

# Step 4: Using the filtered dataframe, find the correlation between the selected movie and other movies and rank them.
user_movie_df.corrwith(movie_df).sort_values(ascending=False).head(10)

# Function that executes the last two steps
def item_based_recommender(movie_name, user_movie_df):
    movie = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie).sort_values(ascending=False).head(10)

# Step 5: Give the first 5 movies as suggestions, apart from the selected movie itself.
movies_from_item_based = item_based_recommender(movie[movie["movieId"] == movie_id]["title"].values[0], user_movie_df)
# Through 1 to 6. Index 0 has the movie itself. We have to drop it.
movies_from_item_based[1:6].index


# 'My Science Project (1985)',
# 'Mediterraneo (1991)',
# 'Old Man and the Sea,
# The (1958)',
# 'National Lampoon's Senior Trip (1995)',
# 'Clockwatchers (1997)']




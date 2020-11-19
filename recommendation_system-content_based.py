# -*- coding: utf-8 -*-

"""# 基於內容推薦"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import paired_distances,cosine_similarity

movies = pd.read_csv('./ml-latest-small/movies.csv')
rate = pd.read_csv('./ml-latest-small/ratings.csv')

display(movies.head())
display(rate.head())

# movies留下movieId與genres(電影分類)
# rate留下userId與movieId
# 把movies與rate以movieId合併成一個df

movies.drop('title',axis=1,inplace=True)
rate.drop(['rating', 'timestamp'],axis=1,inplace=True)
df = pd.merge(rate, movies, on='movieId')
df.head()

# 建立movie的特徵矩陣
oneHot = movies["genres"].str.get_dummies("|") # One-Hot Encoding
movie_arr = pd.concat([movies, oneHot], axis=1)
movie_arr.drop("genres",axis=1,inplace=True)
movie_arr.set_index("movieId",inplace=True)
display(movie_arr.head())

# 建立user的特徵矩陣
oneHot = df["genres"].str.get_dummies("|") # One-Hot Encoding
user_arr = pd.concat([df, oneHot], axis=1)
user_arr.drop(["movieId","genres"],axis=1,inplace=True)
user_arr = user_arr.groupby('userId').mean()
display(user_arr.head())

print(movie_arr.shape)
print(user_arr.shape)

# user-movie相似度矩陣
similar_matrix = 1-cosine_similarity(user_arr.values,movie_arr.values) # 要找距離最短的
similar_matrix = pd.DataFrame(similar_matrix, index = user_arr.index, columns = movie_arr.index)
similar_matrix

# 取得與特定user最相似的前num部movie
def get_the_most_similar_movies(searchUserId, num):
  vec = similar_matrix.loc[searchUserId].values
  sorted_index = np.argsort(vec)[:num]  #找距離最短
  return list(similar_matrix.columns[sorted_index])

# 取得與特定movie最相似的前num部movie
def get_the_most_similar_users(searchMovieId, num):
  movie_vec = similar_matrix[searchMovieId].values 
  sorted_index = np.argsort(movie_vec)[:num]  #找距離最短
  return list(similar_matrix.index[sorted_index])

# sort最相似的資料
searchMovieId = 1
searchUserId = 2
num = 10

similar_movies_index = get_the_most_similar_movies(searchUserId, num)
similar_user_index = get_the_most_similar_users(searchMovieId, num)
print(similar_movies_index)
print(similar_user_index)

# 重新讀入movies.csv為了要有title
movies = pd.read_csv('./ml-latest-small/movies.csv')

# 列出推薦名單
df_recommend_movies = pd.DataFrame({f'推薦給[使用者{searchUserId}]的前{num}部電影':movies[movies.movieId.isin(similar_movies_index)].title[:num]}).reset_index()
df_recommend_movies.drop('index',axis=1,inplace=True)
df_recommend_users = pd.DataFrame({f'可能會喜歡[電影{searchMovieId}]的前{num}個使用者':rate[rate.userId.isin(similar_user_index)].userId.unique()[:num]}).reset_index()
df_recommend_users.drop('index',axis=1,inplace=True)

pd.concat([df_recommend_movies,df_recommend_users],axis=1)
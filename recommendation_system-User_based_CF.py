"""# 基於使用者的偕同過濾推薦(User-based Collaborative Filtering)"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import paired_distances,cosine_similarity

movies = pd.read_csv('./ml-latest-small/movies.csv')
rate = pd.read_csv('./ml-latest-small/ratings.csv')

display(movies.head())
display(rate.head())

movies.drop('genres',axis=1,inplace=True)
rate.drop('timestamp',axis=1,inplace=True)
df = pd.merge(rate,movies,on='movieId')
df

groups = df.groupby('userId')
pd.DataFrame(groups.size(),columns=['count']).plot();

def find_common_movies(user, other_users):
  s1 = set(user.movieId.values)
  s2 = set(other_users.movieId.values) 
  return s1.intersection(s2)

def vec2matrix_cosine_similarity(vec1,vec2):
  vec1 = np.mat(vec1)
  vec2 = np.mat(vec2)
  cos = float(vec1*vec2.T)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
  sim = 0.5 + 0.5 * cos
  return sim

def cal_cosine_similarity_from_rating(user,other_users,common_moviesId):
  user_rating = user[user.movieId.isin(common_moviesId)].sort_values(by="movieId")["rating"].values.reshape(1,len(common_moviesId))
  other_user_rating = other_users[other_users.movieId.isin(common_moviesId)].sort_values(by="movieId")["rating"].values.reshape(1,len(common_moviesId))
  sim = vec2matrix_cosine_similarity(user_rating,other_user_rating)
  return sim

def cal_each_user_similarity(userId):
  user_similarity = []
  for other_userId in df.userId.unique():
    if other_userId == userId:
      continue
    user = groups.get_group(userId)
    other_users = groups.get_group(other_userId)
    common_moviesId = find_common_movies(user,other_users)
    # 避免都無關，common_moviesId = {}
    if common_moviesId != set():
      sim = cal_cosine_similarity_from_rating(user,other_users,common_moviesId)
      user_similarity.append([other_userId,sim])
  return user_similarity

def top_num_similar_users(user_Id, num):
  user_similarity = cal_each_user_similarity(user_Id)
  user_similarity = sorted(user_similarity, key=lambda x: x[1], reverse=True)
  similar_users = [x[0] for x in user_similarity][0:num]
  return similar_users

def recommend(user_Id, num=10):
  # 找尋最相近的前num個使用者
  similar_users = top_num_similar_users(user_Id, num)
  # 欲搜尋的user_Id看過的電影
  seen_movies = df.loc[df.userId==user_Id,"movieId"].values
  # 由其他相似的使用者看過的電影來找出欲搜尋的user_Id沒看過的電影
  other_similarUsers_seen_movies = df.loc[df.userId.isin(similar_users),"movieId"].values
  not_seen_movies = set(other_similarUsers_seen_movies)-set(seen_movies)
  # 計算這些沒看過的電影的平均評分
  movie_groups = df.loc[df.movieId.isin(not_seen_movies)].groupby('movieId')
  top_num_movies = movie_groups.mean().sort_values(by='rating', ascending=False)[:num].index
  return df.loc[df.movieId.isin(top_num_movies), "title"].unique()

# sort最相似的資料
searchUserId = 3
num = 10

# 透過協同過濾法推薦給searchUserId前num個電影
recommend_top_num_movies = recommend(searchUserId, num)

# 列出推薦名單
df_recommend_movies = pd.DataFrame({f'推薦給[使用者{searchUserId}]的前{num}部電影':recommend_top_num_movies}).reset_index()
df_recommend_movies.drop('index',axis=1,inplace=True)
df_recommend_movies
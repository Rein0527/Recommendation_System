# 協同過濾
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import paired_distances,cosine_similarity

# 流程:
# 1.找到與使用者有共同喜好的其他使用者
# 2.將這些其他使用者看過的最高評分的餐廳推薦給使用者

# 導入資料
restaurant = pd.read_csv('restaurant.csv')
customer = pd.read_csv('customer.csv')
# print(restaurant.head())
# print(customer.head())

# 合併資料集
restaurant.drop('genres',axis=1,inplace=True)
customer.drop('timestamp',axis=1,inplace=True)
df = pd.merge(customer,restaurant,on='rId')
# print(df)

# 資料分布
groups = df.groupby('userId')
pd.DataFrame(groups.size(),columns=['count'])

# 找出使用者與其他使用者共同去過的餐廳id
def find_common_restaurant(user, other_users):
  s1 = set(user.rId.values)
  s2 = set(other_users.rId.values) 
  return s1.intersection(s2)

def vec2matrix_cosine_similarity(vec1,vec2):
  vec1 = np.mat(vec1)
  vec2 = np.mat(vec2)
  cos = float(vec1*vec2.T)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
  sim = 0.5 + 0.5 * cos
  return sim

# 由餐廳的評價計算客戶的餘弦相似度
def cal_cosine_similarity_from_rating(user,other_users,common_rId):
  user_rating = user[user.rId.isin(common_rId)].sort_values(by="rId")["rating"].values.reshape(1,len(common_rId))
  other_user_rating = other_users[other_users.rId.isin(common_rId)].sort_values(by="rId")["rating"].values.reshape(1,len(common_rId))
  sim = vec2matrix_cosine_similarity(user_rating,other_user_rating)
  return sim

def cal_each_user_similarity(userId):
  user_similarity = []
  for other_userId in df.userId.unique():
    if other_userId == userId:
      continue
    user = groups.get_group(userId)
    other_users = groups.get_group(other_userId)
    common_rId = find_common_restaurant(user,other_users)
    if common_rId != set():
      sim = cal_cosine_similarity_from_rating(user,other_users,common_rId)
      user_similarity.append([other_userId,sim])
  return user_similarity

# 找出前幾個相似的使用者Id
def top_num_similar_users(user_Id, num):
  user_similarity = cal_each_user_similarity(user_Id)
  user_similarity = sorted(user_similarity, key=lambda x: x[1], reverse=True)
  similar_users = [x[0] for x in user_similarity][0:num]
  return similar_users

def recommend(user_Id, num=10):
  # 找尋最相近的前幾個使用者
  similar_users = top_num_similar_users(user_Id, num)
  # 欲搜尋的user_Id去過的餐廳
  seen_restaurant = df.loc[df.userId==user_Id,"rId"].values
  # 由其他相似的使用者去過的餐廳來找出欲搜尋的user_Id沒去過的餐廳
  other_similarUsers_seen_restaurant = df.loc[df.userId.isin(similar_users),"rId"].values
  not_seen_restaurant = set(other_similarUsers_seen_restaurant)-set(seen_restaurant)
  # 計算這些沒去過的餐廳的平均評分
  restaurant_groups = df.loc[df.rId.isin(not_seen_restaurant)].groupby('rId')
  top_num_restaurant = restaurant_groups.mean().sort_values(by='rating', ascending=False)[:num].index
  return df.loc[df.rId.isin(top_num_restaurant), "title"].unique()

# sort最相似的資料
searchUserId = int(input("searchUserId:"))
#searchUserId = 10
num = 3

# 透過協同過濾法推薦給searchUserId前幾個餐廳
recommend_top_num_restaurant = recommend(searchUserId, num)

# 列出推薦名單
df_recommend_restaurant = pd.DataFrame({f'推薦給[客戶{searchUserId}]的前{num}間餐廳':recommend_top_num_restaurant}).reset_index()
df_recommend_restaurant.drop('index',axis=1,inplace=True)
print(df_recommend_restaurant)

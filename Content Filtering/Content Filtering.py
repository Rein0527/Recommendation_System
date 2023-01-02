import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import paired_distances,cosine_similarity

#內容過濾
#根據你過去喜歡的產品，去推薦給你相似的產品。
# 流程:
# 1.建立特徵矩陣，計算每個使用者過往對每一種 餐廳風格(style) 的平均評分
# 2.計算 客戶 與 餐廳 的餘弦相似度(距離)
# 3.推薦 某客戶可能喜愛的餐廳 或 某餐廳可能受喜愛的客戶

#導入資料
restaurant_data = pd.read_csv('restaurant_data.csv')
customer_data = pd.read_csv('customer_data.csv')
name = ['摩斯漢堡','麥當勞','老董便當','葡吉小廚','鐵火燒肉','黑毛屋','涓豆腐','奧特拉麵','新高軒','義塔里','上村牧場','偷飯賊','英格莉莉','原味德里','頌丹樂','日本橋濱町酒食處','韓鶴亭','一風堂','貳樓餐廳','乾杯列車','丸壽司','瓦城大心','金子半之助','扒飯眷村菜','韓虎嘯','大戶屋','點點心','銀座杏子豬排','小南門點心世界','雞三和','瓦城 EXPRESS','大河屋','漢堡王','The Soup Spoon 匙碗湯','八方雲集','GUGU廚房義式料理','港苑茶餐廳']
ID = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]
restaurant_dict={0:'摩斯漢堡 ',1:'麥當勞 ',2:'老董便當',3:'葡吉小廚',4:'鐵火燒肉',5:'黑毛屋',6:'涓豆腐',7:'奧特拉麵',8:'新高軒',9:'義塔里',10:'上村牧場',11:'偷飯賊',12:'英格莉莉',13:'原味德里',14:'頌丹樂',15:'日本橋濱町酒食處',16:'韓鶴亭',17:'一風堂',18:'貳樓餐廳',19:'乾杯列車',20:'丸壽司',21:'瓦城大心',22:'金子半之助',23:'扒飯眷村菜' ,24:'韓虎嘯' ,25:'大戶屋' ,26:'點點心' ,27:'銀座杏子豬排' ,28:'小南門點心世界' ,29:'雞三和' ,30:'瓦城 EXPRESS',31:'大河屋',32:'漢堡王' ,33:'The Soup Spoon 匙碗湯',34:'八方雲集',35:'GUGU廚房義式料理',36:'港苑茶餐廳'}
rd= {"ID":ID,'餐廳':name}  # key可視為欄名
df=pd.DataFrame(rd)
print(df)
# print(restaurant_data)
# print(customer_data)

# restaurant留下name與style
# customer留下customer與name
# 把restaurant與customer以name合併成一個df
restaurant_data.drop(['name'],axis=1,inplace=True)
customer_data.drop(['rating'],axis=1,inplace=True)
df = pd.merge(customer_data,restaurant_data, on='rID')
# print(df)

# 建立特徵矩陣
# restaurant_arr = 餐廳ID對style做完One-Hot Encoding 的矩陣，背後意義是餐廳ID對應到何種餐廳風格
# customer_arr = 客戶ID對style做完One-Hot Encoding 後客戶對每個餐廳的平均評分的矩陣，背後意義是客戶ID對應到某種餐廳風格的平均評分

#建立restaurant的特徵矩陣
oneHot = restaurant_data['style'].str.get_dummies("|") # One-Hot Encoding
restaurant_arr = pd.concat([restaurant_data, oneHot], axis=1)
restaurant_arr.drop('style',axis=1,inplace=True)
restaurant_arr.set_index('rID',inplace=True)
# print(restaurant_arr.head())

# 建立customer的特徵矩陣
oneHot = df['style'].str.get_dummies("|") # One-Hot Encoding
customer_arr = pd.concat([df, oneHot], axis=1)
customer_arr.drop(['rID','style'],axis=1,inplace=True)
customer_arr = customer_arr.groupby('userID').mean()
# print(customer_arr.head())

# customer-restaurant相似度矩陣
similar_matrix = cosine_similarity(customer_arr.values,restaurant_arr.values)
similar_matrix = pd.DataFrame(similar_matrix, index = customer_arr.index, columns = restaurant_arr.index)
#print(similar_matrix.head())

# 定義兩個函式分別取得前幾個最相似的restaurant與前幾個最相似的customer

# 取得與特定customer最相似的前num個restaurant
def get_the_most_similar_restaurant(searchuserID, num):
  vec = similar_matrix.loc[searchuserID].values
  sorted_index = np.argsort(-vec)[:num]   #找距離最短
  return list(similar_matrix.columns[sorted_index])

# 取得與特定restaurant最相似的前num個customer
def get_the_most_similar_customer(searchrID, num):
  restaurant_vec = similar_matrix[searchrID].values 
  sorted_index = np.argsort(-restaurant_vec)[:num]  #找距離最短
  return list(similar_matrix.index[sorted_index])  


# 設定欲搜尋的餐廳id和使用者id

# sort最相似的資料
searchrID = int(input('搜尋餐廳ID:'))
searchuserID = int(input('搜尋客戶ID:'))
num = int(input('顯示筆數:'))

# 開始搜尋

similar_restaurant_index = get_the_most_similar_restaurant(searchuserID, num)
similar_customer_index = get_the_most_similar_customer(searchrID, num)
#print(similar_restaurant_index)
#print(similar_customer_index)

# 列出推薦餐廳與客戶的名單

# 重新讀入restaurant.csv為了要有name
restaurant = pd.read_csv('restaurant_data.csv')

# 列出推薦名單
df_recommend_restaurant = pd.DataFrame({f'推薦給[客戶{searchuserID}]的前{num}餐廳':restaurant[restaurant.rID.isin(similar_restaurant_index)].name[:num]}).reset_index()
df_recommend_restaurant.drop('index',axis=1,inplace=True)
df_recommend_customer = pd.DataFrame({f'可能會喜歡[餐廳{searchrID}]的前{num}個客戶':customer_data[customer_data.userID.isin(similar_customer_index)].userID.unique()[:num]}).reset_index()
df_recommend_customer.drop('index',axis=1,inplace=True)

# result=pd.concat([df_recommend_restaurant,df_recommend_customer],axis=1)
print('客戶去過的餐廳',restaurant_dict[searchrID])
# print(result)
print(df_recommend_restaurant)
print(df_recommend_customer)




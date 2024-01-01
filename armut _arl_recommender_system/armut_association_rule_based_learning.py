import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

df_org = pd.read_csv(r"datasets\armut_data.csv")

df = df_org.copy()

df.head()

# Checking the Data

def check_data(dataframe,head=5):
    print(20*"-" + "Information".center(20) + 20*"-")
    print(dataframe.info())
    print("\n" + 20*"-" + "The First 5 Data".center(20) + 20*"-")
    print(dataframe.head())
    print("\n" + 20 * "-" + "The Last 5 Data".center(20) + 20 * "-")
    print(dataframe.tail())
    print("\n" + 20 * "-" + "Missing Values".center(20) + 20 * "-")
    print(dataframe.isnull().sum())
    print("\n" + 40 * "-" + "Describe the Data".center(40) + 40 * "-")
    print(dataframe.describe([0.01, 0.05, 0.10, 0.50, 0.75, 0.90, 0.95, 0.99]).T)

check_data(df)


# Step 2

df['New_Service'] = df['ServiceId'].astype(str) + "_" + df['CategoryId'].astype(str)
df.head()
# Out[15]:
#    UserId  ServiceId  CategoryId           CreateDate New_Service
# 0   25446          4           5  2017-08-06 16:11:00         4_5
# 1   22948         48           5  2017-08-06 16:12:00        48_5
# 2   10618          0           8  2017-08-06 16:13:00         0_8
# 3    7256          9           4  2017-08-06 16:14:00         9_4
# 4   25446         48           5  2017-08-06 16:16:00        48_5


# Step 3

df['Year-Month'] = pd.to_datetime(df['CreateDate'],format='%Y-%m').dt.to_period('M')
df['BasketId'] = df['UserId'].astype(str) + "_" + df['Year-Month'].astype(str)
df.info()
df.head()
# Out[19]:
#    UserId  ServiceId  CategoryId           CreateDate New_Service Year-Month       BasketId
# 0   25446          4           5  2017-08-06 16:11:00         4_5    2017-08  25446_2017-08
# 1   22948         48           5  2017-08-06 16:12:00        48_5    2017-08  22948_2017-08
# 2   10618          0           8  2017-08-06 16:13:00         0_8    2017-08  10618_2017-08
# 3    7256          9           4  2017-08-06 16:14:00         9_4    2017-08   7256_2017-08
# 4   25446         48           5  2017-08-06 16:16:00        48_5    2017-08  25446_2017-08

################################################################3
# Develop the Association Rules and Make Suggestions
#################################################################


df_serv = df.groupby(['BasketId', "New_Service"])["New_Service"].count().unstack().fillna(0).\
    applymap(lambda x: 1 if x >0 else 0)

df_serv.head()

df_serv.shape
# Out[26]: (71220, 50)


# Step 2

frequent_items = apriori(df_serv,min_support=0.01,use_colnames=True)

frequent_items.sort_values("support", ascending=False)


rules = association_rules(frequent_items,
                          metric="support",
                          min_threshold=0.01)


#

rules[(rules["support"]>0.01) & (rules["confidence"]>0.2) & (rules["lift"]>2)]. \
sort_values("confidence", ascending=False)


df.loc[df["New_Service"] == "2_0"].sort_values("CreateDate", ascending=False).head()
# Out[66]:
#         UserId  ServiceId  CategoryId           CreateDate New_Service Year-Month       BasketId
# 162519   10591          2           0  2018-08-06 14:43:00         2_0    2018-08  10591_2018-08
# 162502   11769          2           0  2018-08-06 09:30:00         2_0    2018-08  11769_2018-08
# 162497   12022          2           0  2018-08-06 08:47:00         2_0    2018-08  12022_2018-08
# 162484   11656          2           0  2018-08-06 07:17:00         2_0    2018-08  11656_2018-08
# 162469   18900          2           0  2018-08-06 04:30:00         2_0    2018-08  18900_2018-08

# UserId = 10591

rules[rules["antecedents"] == {'2_0'}].sort_values("confidence", ascending=False).head() # with confidence

rules[rules["antecedents"] == {'2_0'}].sort_values("lift", ascending=False).head() # or lift


######### Function

def arl_recommender(df_rules, product_id, rec_count=1):
    sorted_rules = df_rules.sort_values("support", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["consequents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["antecedents"])[:])
    return recommendation_list[0:rec_count]


arl_recommender(rules,'18_4',1)

arl_recommender(rules,'25_0',3)

arl_recommender(rules,'22_0',3)

arl_recommender(rules,'13_11',3)

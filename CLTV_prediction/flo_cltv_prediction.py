

# CUSTOMER LIFETIME VALUE PREDICTION

# BUSINESS PROBLEM:

# FLO,an online shoe store,wants to segment its customers and determine marketing strategies
# according to these segments. In this regard, the behavior of customers will be defined and
# groups will be formed according to the clustering in these behaviors.



# Story of Dataset

# The dataset shows the customers which last purchases from the FLO store on Omnichannel(both online and offline
# shopping store)in 2020-2021. However, these customers have consist of infomation from their past shopping behavior.


# master_id : Unique Customer Number
# order_channel : Which channel of the shopping platform is used (Android, ios, Desktop, Mobile))
# last_order_channel : The channel where the most recent purchase was made
# first_order_date : The customer's first purchase date
# last_order_date : The customer's last purchase date
# last_order_date_online :  The customer's last purchase date in online shopping platform
# last_order_date_offline : The customer's last purchase date in offline shopping platform
# order_num_total_ever_online : The customer's total purchases in online shopping platform
# order_num_total_ever_offline :  The customer's total purchases in offline shopping platform
# customer_value_total_ever_offline : The total expenditure by customer in offline shopping platform
# customer_value_total_ever_online : The total expenditure by customer in online shopping platform
# interested_in_categories_12 : List of categories the customer has shopped in the last 12 months


##############################################################################################
#                          MISSION 1 : DATA UNDERSTANDING AND PREPARATION                    #
##############################################################################################

# STEP 1 -
#
# IMPORT LIBRARIES

import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns',1000)
pd.set_option('display.width', 500)
# pd.set_option('display_max_rows',None)
pd.set_option('display.float_format',lambda x : '%.5f' % x)

# READ DATASET
df_= pd.read_csv(r"datasets\flo_data_20k.csv")
df = df_.copy()


# CHECKING THE DATA

def datacheck(dataframe):
    print("******Head******")
    print(dataframe.head(10))
    print("******Shape******")
    print(dataframe.shape)
    print("******Info********")
    print(dataframe.info())
    print("******Describe********")
    print(dataframe.describe().T)
    print("***** NAN Values********")
    print(dataframe.isnull().sum())

datacheck(df)
# ******Head******
#                               master_id order_channel last_order_channel first_order_date last_order_date last_order_date_online last_order_date_offline  order_num_total_ever_online  order_num_total_ever_offline  customer_value_total_ever_offline  customer_value_total_ever_online       interested_in_categories_12
# 0  cc294636-19f0-11eb-8d74-000d3a38a36f   Android App            Offline       2020-10-30      2021-02-26             2021-02-21              2021-02-26                      4.00000                       1.00000                          139.99000                         799.38000                           [KADIN]
# 1  f431bd5a-ab7b-11e9-a2fc-000d3a38a36f   Android App             Mobile       2017-02-08      2021-02-16             2021-02-16              2020-01-10                     19.00000                       2.00000                          159.97000                        1853.58000  [ERKEK, COCUK, KADIN, AKTIFSPOR]
# 2  69b69676-1a40-11ea-941b-000d3a38a36f   Android App        Android App       2019-11-27      2020-11-27             2020-11-27              2019-12-01                      3.00000                       2.00000                          189.97000                         395.35000                    [ERKEK, KADIN]
# 3  1854e56c-491f-11eb-806e-000d3a38a36f   Android App        Android App       2021-01-06      2021-01-17             2021-01-17              2021-01-06                      1.00000                       1.00000                           39.99000                          81.98000               [AKTIFCOCUK, COCUK]
# 4  d6ea1074-f1f5-11e9-9346-000d3a38a36f       Desktop            Desktop       2019-08-03      2021-03-07             2021-03-07              2019-08-03                      1.00000                       1.00000                           49.99000                         159.99000                       [AKTIFSPOR]
# 5  e585280e-aae1-11e9-a2fc-000d3a38a36f       Desktop            Offline       2018-11-18      2021-03-13             2018-11-18              2021-03-13                      1.00000                       2.00000                          150.87000                          49.99000                           [KADIN]
# 6  c445e4ee-6242-11ea-9d1a-000d3a38a36f   Android App        Android App       2020-03-04      2020-10-18             2020-10-18              2020-03-04                      3.00000                       1.00000                           59.99000                         315.94000                       [AKTIFSPOR]
# 7  3f1b4dc8-8a7d-11ea-8ec0-000d3a38a36f        Mobile            Offline       2020-05-15      2020-08-12             2020-05-15              2020-08-12                      1.00000                       1.00000                           49.99000                         113.64000                           [COCUK]
# 8  cfbda69e-5b4f-11ea-aca7-000d3a38a36f   Android App        Android App       2020-01-23      2021-03-07             2021-03-07              2020-01-25                      3.00000                       2.00000                          120.48000                         934.21000             [ERKEK, COCUK, KADIN]
# 9  1143f032-440d-11ea-8b43-000d3a38a36f        Mobile             Mobile       2019-07-30      2020-10-04             2020-10-04              2019-07-30                      1.00000                       1.00000                           69.98000                          95.98000                [KADIN, AKTIFSPOR]

# ******Shape******
# (19945, 12)

# ******Info********
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 19945 entries, 0 to 19944
# Data columns (total 12 columns):
#  #   Column                             Non-Null Count  Dtype
# ---  ------                             --------------  -----
#  0   master_id                          19945 non-null  object
#  1   order_channel                      19945 non-null  object
#  2   last_order_channel                 19945 non-null  object
#  3   first_order_date                   19945 non-null  object
#  4   last_order_date                    19945 non-null  object
#  5   last_order_date_online             19945 non-null  object
#  6   last_order_date_offline            19945 non-null  object
#  7   order_num_total_ever_online        19945 non-null  float64
#  8   order_num_total_ever_offline       19945 non-null  float64
#  9   customer_value_total_ever_offline  19945 non-null  float64
#  10  customer_value_total_ever_online   19945 non-null  float64
#  11  interested_in_categories_12        19945 non-null  object
# dtypes: float64(4), object(8)
# memory usage: 1.8+ MB
# None

# ******Describe********
#                                         count      mean       std      min       25%       50%       75%         max
# order_num_total_ever_online       19945.00000   3.11085   4.22565  1.00000   1.00000   2.00000   4.00000   200.00000
# order_num_total_ever_offline      19945.00000   1.91391   2.06288  1.00000   1.00000   1.00000   2.00000   109.00000
# customer_value_total_ever_offline 19945.00000 253.92260 301.53285 10.00000  99.99000 179.98000 319.97000 18119.14000
# customer_value_total_ever_online  19945.00000 497.32169 832.60189 12.99000 149.98000 286.46000 578.44000 45220.13000

# ***** NAN Values********
# master_id                            0
# order_channel                        0
# last_order_channel                   0
# first_order_date                     0
# last_order_date                      0
# last_order_date_online               0
# last_order_date_offline              0
# order_num_total_ever_online          0
# order_num_total_ever_offline         0
# customer_value_total_ever_offline    0
# customer_value_total_ever_online     0
# interested_in_categories_12          0
# dtype: int64


# STEP 2 Define the function for outlier threshold and replace with threshold values

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = round(quartile3 + 1.5 * interquantile_range)
    low_limit = round(quartile1 - 1.5 * interquantile_range)
    return low_limit, up_limit



def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit



# STEP 3
replace_with_thresholds(df, "order_num_total_ever_online")
replace_with_thresholds(df, "order_num_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_online")

df.describe().T

df.describe(percentiles=[.50,.60,.70,.80,.90]).T


# STEP 4

# Total purchases for omnichannel customers
df["total_purchases_number"] = df["order_num_total_ever_online"] +  df["order_num_total_ever_offline"]

# Total expense for omnichannel customers
df["total_customer_expense"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

df.head()

# STEP 5

convert =["first_order_date","last_order_date","last_order_date_online","last_order_date_offline"]
df[convert] = df[convert].apply(pd.to_datetime)
df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 19945 entries, 0 to 19944
# Data columns (total 14 columns):
#  #   Column                             Non-Null Count  Dtype
# ---  ------                             --------------  -----
#  0   master_id                          19945 non-null  object
#  1   order_channel                      19945 non-null  object
#  2   last_order_channel                 19945 non-null  object
#  3   first_order_date                   19945 non-null  datetime64[ns]
#  4   last_order_date                    19945 non-null  datetime64[ns]
#  5   last_order_date_online             19945 non-null  datetime64[ns]
#  6   last_order_date_offline            19945 non-null  datetime64[ns]
#  7   order_num_total_ever_online        19945 non-null  float64
#  8   order_num_total_ever_offline       19945 non-null  float64
#  9   customer_value_total_ever_offline  19945 non-null  float64
#  10  customer_value_total_ever_online   19945 non-null  float64
#  11  interested_in_categories_12        19945 non-null  object
#  12  total_purchases_number             19945 non-null  float64
#  13  total_customer_expense             19945 non-null  float64
# dtypes: datetime64[ns](4), float64(6), object(4)
# memory usage: 2.1+ MB


##############################################################################################
#                          MISSION 2 : CREATING CLTV DATA STRUCTURE                          #
##############################################################################################

df["last_order_date"].max()
#  '2021-05-30'
last_date = dt.datetime(2021,5,30)
type(last_date)
#  datetime.datetime


today_date = dt.datetime(2021, 6, 2)
type(today_date)


cltv_df = pd.DataFrame({"customer_id": df["master_id"],
             "recency_cltv_weekly": ((df["last_order_date"] - df["first_order_date"]).dt.days)/7,
             "T_weekly": ((today_date - df["first_order_date"]).dt.days)/7,
             "frequency": df["total_purchases_number"],
             "monetary_cltv_avg": df["total_customer_expense"] / df["total_purchases_number"]})


cltv_df.head()
#                             customer_id  recency_cltv_weekly  T_weekly  frequency  monetary_cltv_avg
# 0  cc294636-19f0-11eb-8d74-000d3a38a36f             17.00000  30.71429    5.00000          187.87400
# 1  f431bd5a-ab7b-11e9-a2fc-000d3a38a36f            209.85714 225.00000   10.00000          138.09700
# 2  69b69676-1a40-11ea-941b-000d3a38a36f             52.28571  79.00000    5.00000          117.06400
# 3  1854e56c-491f-11eb-806e-000d3a38a36f              1.57143  21.00000    2.00000           60.98500
# 4  d6ea1074-f1f5-11e9-9346-000d3a38a36f             83.14286  95.57143    2.00000          104.99000

##############################################################################################
#            MISSION 3 : CREATING BG/NBD,GAMMA GAMMA MODELS AND CALCULATING CLTV             #
##############################################################################################

# STEP 1 : Fit the BG/NBD Model

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
       cltv_df['recency_cltv_weekly'],
       cltv_df['T_weekly'])

bgf.summary


# Out[60]:
#           coef  se(coef)  lower 95% bound  upper 95% bound
# r      3.48182   0.04946          3.38488          3.57877
# alpha 71.77502   1.12544         69.56916         73.98088
# a      0.00004   0.00003         -0.00002          0.00009
# b      0.26044   0.11470          0.03562          0.48526



#  Transaction Process (Buy)     ///    Till You Die (BırakmaSüreci)
# BG/NBD modeli için model parametreleri R, alfa, a ve b'dir.
# R: Satın alma işleminin gama  dağılımının şekil parametresi.   Buy
# Alpha: Satın alma işleminin gama dağılımının ölçek parametresi.Müşteri sadakatini temsil etmektedir. Buy
# Bir müşteri hayatta olduğu sürece belirli bir zaman periyodunda bu müşteri tarafından gerçekleştirecek işlem sayısı transaction rate(işlem oranı) ile poisson dağılır.
# Transaction rate her bir müşteriye göre değişir ve tüm kütle için gama(r,α) dağılır.


# a: Bırakma işleminin beta dağılımının şekil parametresi. Till You Die
# b :Bırakma işleminin beta dağılımının şekil parametresi. Till You Die
# Her bir müşterinin p olasılığı ile Dropout Rate’i vardır.
# Bu Dropout Rate’ler tüm kütle üzerinden her bir müşteriye özel olarak beta (a, b) dağılır.

# BG/ NBD modeli ile Customer Life Time Value Prediction hesabının
# Expected Number of Transaction adımını olasılıksal olarak modelleyecek.



from lifetimes.plotting import plot_frequency_recency_matrix

plot_frequency_recency_matrix(bgf)
plt.show(block=True)



from lifetimes.plotting import plot_probability_alive_matrix
fig = plt.figure(figsize=(12,8))
plot_probability_alive_matrix(bgf)
plt.show(block=True)



cltv_df["exp_sales_3_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4*3,
                                                        cltv_df['frequency'],
                                                        cltv_df['recency_cltv_weekly'],
                                                        cltv_df['T_weekly'])

cltv_df.sort_values(by='exp_sales_6_month', ascending=False).head()

cltv_df["exp_sales_6_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4*6,
                                                        cltv_df['frequency'],
                                                        cltv_df['recency_cltv_weekly'],
                                                        cltv_df['T_weekly'])
cltv_df.head()

# Out[93]:
#                             customer_id  recency_cltv_weekly  T_weekly  frequency  monetary_cltv_avg  exp_sales_3_month  exp_sales_6_month
# 0  cc294636-19f0-11eb-8d74-000d3a38a36f             17.00000  30.71429    5.00000          187.87400            0.99307            1.98613
# 1  f431bd5a-ab7b-11e9-a2fc-000d3a38a36f            209.85714 225.00000   21.00000           95.88333            0.98991            1.97981
# 2  69b69676-1a40-11ea-941b-000d3a38a36f             52.28571  79.00000    5.00000          117.06400            0.67503            1.35005
# 3  1854e56c-491f-11eb-806e-000d3a38a36f              1.57143  21.00000    2.00000           60.98500            0.70897            1.41794
# 4  d6ea1074-f1f5-11e9-9346-000d3a38a36f             83.14286  95.57143    2.00000          104.99000            0.39307            0.78614



# bgf.expected_number_of_purchases_up_to_time(4*3)



# Alternative

cltv_df["exp_sales_6_month"] = bgf.predict(4*6,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'])



cltv_df['probability_alive'] = bgf.conditional_probability_alive(cltv_df['frequency'], cltv_df['recency_cltv_weekly'], cltv_df['T_weekly'])

cltv_df.head(10)
# Out[97]:
#                             customer_id  recency_cltv_weekly  T_weekly  frequency  monetary_cltv_avg  exp_sales_3_month  exp_sales_6_month  probability_alive
# 0  cc294636-19f0-11eb-8d74-000d3a38a36f             17.00000  30.71429    5.00000          187.87400            0.99307            1.98613            0.99997
# 1  f431bd5a-ab7b-11e9-a2fc-000d3a38a36f            209.85714 225.00000   21.00000           95.88333            0.98991            1.97981            0.99999
# 2  69b69676-1a40-11ea-941b-000d3a38a36f             52.28571  79.00000    5.00000          117.06400            0.67503            1.35005            0.99996
# 3  1854e56c-491f-11eb-806e-000d3a38a36f              1.57143  21.00000    2.00000           60.98500            0.70897            1.41794            0.99990
# 4  d6ea1074-f1f5-11e9-9346-000d3a38a36f             83.14286  95.57143    2.00000          104.99000            0.39307            0.78614            0.99996
# 5  e585280e-aae1-11e9-a2fc-000d3a38a36f            120.85714 132.42857    3.00000           66.95333            0.38089            0.76179            0.99998
# 6  c445e4ee-6242-11ea-9d1a-000d3a38a36f             32.57143  65.00000    4.00000           93.98250            0.65636            1.31273            0.99992
# 7  3f1b4dc8-8a7d-11ea-8ec0-000d3a38a36f             12.71429  54.71429    2.00000           81.81500            0.51992            1.03984            0.99975
# 8  cfbda69e-5b4f-11ea-aca7-000d3a38a36f             58.42857  70.85714    5.00000          210.93800            0.71358            1.42716            0.99998
# 9  1143f032-440d-11ea-8b43-000d3a38a36f             61.71429  96.14286    2.00000           82.98000            0.39171            0.78342            0.99990


# Bonus

bgf.conditional_expected_number_of_purchases_up_to_time(
    4*3, cltv_df["frequency"], cltv_df["recency_cltv_weekly"], cltv_df["T_weekly"]
).sort_values(ascending=False).head(10)


cltv_df["exp_sales_3_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4*3,
                                                        cltv_df['frequency'],
                                                        cltv_df['recency_cltv_weekly'],
                                                        cltv_df['T_weekly'])

cltv_df.head()
#
cltv1= cltv_df.drop(cltv_df.loc[:,'recency_cltv_weekly':'monetary_cltv_avg'].columns, axis=1)

cltv1.sort_values(by='exp_sales_3_month', ascending=False).head()
# Out[104]:
#                                 customer_id  exp_sales_3_month  exp_sales_6_month  probability_alive
# 7330   a4d534a2-5b1b-11eb-8dbd-000d3a38a36f            6.33443           12.66883            0.99999
# 6322   329968c6-a0e2-11e9-a2fc-000d3a38a36f            5.41538           10.83074            0.99998
# 11150  5d1c466a-9cfd-11e9-9897-000d3a38a36f            5.24088           10.48175            1.00000
# 9347   44d032ee-a0d4-11e9-a2fc-000d3a38a36f            5.02614           10.05227            0.99953
# 14402  03f502d4-a559-11e9-a2fc-000d3a38a36f            4.05869            8.11736            0.99992


# BG-NBD modeli, müşteri davranışını analiz ederek ve modelin hesapladığı parametreleri kullanarak bu tahmini hesaplar.
#
# BG-NBD modeli, müşteri davranışını iki önemli faktöre dayandırır: frekans (müşterinin geçmişte kaç kez satın alma yaptığı) ve yenileme süresi (müşterinin satın alma işlemleri arasındaki zaman aralığı).
# Ayrıca, müşterinin gözlem süresi (müşterinin ilk satın alma tarihinden son satın alma tarihine kadar geçen süre) de dikkate alınır.
#
# Model, müşterinin geçmiş frekansı, yenileme süresi ve gözlem süresi gibi bilgilere dayanarak müşterinin gelecekteki satın alma davranışını tahmin eder. Bu tahmin, müşterinin gelecekte belirli bir zaman aralığında kaç satın alma yapacağını ifade eder.




plot_period_transactions(bgf, max_frequency=10)
plt.show(block=True)

from lifetimes.plotting import plot_probability_alive_matrix

plot_probability_alive_matrix(bgf)
plt.show(block=True)

from lifetimes.plotting import plot_period_transactions
plot_period_transactions(bgf,10)
plt.show(block=True)



# STEP 2 : Fit the Gamma Gamma Model


ggf = GammaGammaFitter(penalizer_coef=0.001)


ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])

ggf.summary

# Out[110]:
#       coef  se(coef)  lower 95% bound  upper 95% bound
# p 13.18529   0.10543         12.97865         13.39194
# q  1.69729   0.01565          1.66661          1.72797
# v 12.95046   0.10533         12.74402         13.15690


# p değeri: Bu değer, Gamma-Gamma modelinin bir parametresidir ve müşteri değerini (ortalama satın alma değeri) tahmin etmek için kullanılır
# q değeri: Bu değer, Gamma-Gamma modelinin bir başka parametresidir ve müşteri değerinin varyansını tahmin etmek için kullanılır.
# v değeri: Bu değer, müşteri değerinin (ortalama satın alma değeri) müşterinin frekansına (geçmişteki satın alma sayısı) bağlı olarak nasıl değiştiğini gösteren bir parametredir.

cltv_df["exp_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])

cltv_df.head()

cltv_six = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'],
                                       cltv_df['monetary_cltv_avg'],
                                       time=6,  # 6 aylık
                                       freq="W",  # T'nin frekans bilgisi.
                                       discount_rate=0.01)


cltv_year = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'],
                                       cltv_df['monetary_cltv_avg'],
                                       time=12,  # 12 aylık
                                       freq="W",  # T'nin frekans bilgisi.
                                       discount_rate=0.01)

cltv_df["cltv_prediction"] = cltv_year

cltv_df.sort_values("cltv_prediction",ascending=False)[:10]


# Out[131]:
#                                 customer_id  recency_cltv_weekly  T_weekly  frequency  monetary_cltv_avg  exp_sales_3_month  exp_sales_6_month  probability_alive  exp_average_profit       cltv
# 13880  7137a5c0-7aad-11ea-8f20-000d3a38a36f              6.14286  13.28571   11.00000          758.08545            2.58110            5.16220            1.00000           767.36060 8071.62542
# 9055   47a642fe-975b-11eb-8c2a-000d3a38a36f              2.85714   8.00000    4.00000         1401.80000            1.34665            2.69331            1.00000          1449.06047 7952.41597
# 6666   53fe00d4-7b7a-11eb-960b-000d3a38a36f              9.71429  13.14286   17.00000          259.86529            3.74575            7.49150            1.00000           262.07291 4000.53190
# 7330   a4d534a2-5b1b-11eb-8dbd-000d3a38a36f             62.71429  67.42857   52.00000          166.22462            5.60671           11.21342            1.00000           166.71225 3809.18890
# 8868   9ce6e520-89b0-11ea-a6e7-000d3a38a36f              3.42857  34.57143    8.00000          601.22625            1.49285            2.98571            1.00000           611.49262 3720.19401
# 12438  625f40a2-5bd2-11ea-98b0-000d3a38a36f             74.28571  74.71429   16.00000          501.87375            1.78445            3.56889            1.00000           506.16667 3680.89926
# 19538  55d54d9e-8ac7-11ea-8ec0-000d3a38a36f             52.57143  58.85714   31.00000          228.53000            3.71576            7.43152            1.00000           229.60695 3476.88076
# 17323  f59053e2-a503-11e9-a2fc-000d3a38a36f             51.71429 101.14286    7.00000         1106.46714            0.75099            1.50198            1.00000          1127.61153 3451.05637
# 6402   851de3b4-8f0c-11eb-8cb8-000d3a38a36f              8.28571   9.57143    2.00000          862.69000            0.90073            1.80146            1.00000           923.67997 3390.56492
# 10876  ae149d98-9b6a-11eb-9c47-000d3a38a36f              6.14286   7.28571    9.00000          317.48444            2.42983            4.85966            1.00000           322.51156 3193.58376




##############################################################################################
#            MISSION 4 : CREATING SEGMENT ACCORDING CLTV VALUES                              #
##############################################################################################

cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv_prediction"], 4, labels=["D","C", "B", "A"])

cltv_df.head()
# Out[133]:
#                             customer_id  recency_cltv_weekly  T_weekly  frequency  monetary_cltv_avg  exp_sales_3_month  exp_sales_6_month  probability_alive  exp_average_profit      cltv cltv_segment
# 0  cc294636-19f0-11eb-8d74-000d3a38a36f             17.00000  30.71429    5.00000          187.87400            1.11345            2.22690            1.00000           193.63268 878.63201            A
# 1  f431bd5a-ab7b-11e9-a2fc-000d3a38a36f            209.85714 225.00000   21.00000           95.88333            1.02466            2.04933            1.00000            96.66505 403.65224            B
# 2  69b69676-1a40-11ea-941b-000d3a38a36f             52.28571  79.00000    5.00000          117.06400            0.69341            1.38681            1.00000           120.96762 341.83234            B
# 3  1854e56c-491f-11eb-806e-000d3a38a36f              1.57143  21.00000    2.00000           60.98500            0.75366            1.50732            1.00000            67.32015 206.76506            D
# 4  d6ea1074-f1f5-11e9-9346-000d3a38a36f             83.14286  95.57143    2.00000          104.99000            0.36490            0.72980            1.00000           114.32511 170.00938            D



stats=["mean","sum","count"]

cltv_df.columns

cltv_df.groupby("cltv_segment").agg({"recency_cltv_weekly":stats,"frequency":stats,"exp_sales_3_month":stats,"exp_sales_6_month":stats,"exp_average_profit":stats,"cltv_prediction":stats})

#              recency_cltv_weekly                    frequency                   exp_sales_3_month                  exp_sales_6_month                  exp_average_profit                          cltv
#                             mean          sum count      mean         sum count              mean        sum count              mean        sum count               mean           sum count      mean           sum count
# cltv_segment
# D                      140.79412 702140.28571  4987   3.68999 18402.00000  4987           0.38005 1895.29454  4987           0.76009 3790.58908  4987          101.37834  505573.78160  4987 147.58126  735987.73892  4987
# C                       92.51484 461279.00000  4986   4.32331 21556.00000  4986           0.51221 2553.89420  4986           1.02443 5107.78840  4986          134.44674  670351.45625  4986 263.47476 1313685.13802  4986
# B                       81.14163 404572.14286  4986   5.10329 25445.00000  4986           0.61119 3047.41044  4986           1.22239 6094.82087  4986          168.20611  838675.68495  4986 389.58473 1942469.46993  4986
# A                       66.59426 332039.00000  4986   6.79623 33886.00000  4986           0.84371 4206.75429  4986           1.68743 8413.50857  4986          232.93452 1161411.49184  4986 745.16778 3715406.53870  4986



# Write a csv file to a new folder

from pathlib import Path
filepath = Path('D:/12thTerm_DS_Bootcamp/3Week_CRM_Analytics/cltv.csv')
filepath.parent.mkdir(parents=True, exist_ok=True)
cltv_df.to_csv(filepath)


###################################
# BONUS#
###################################

cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 3, labels=["C", "B", "A"])
cltv_df.groupby("cltv_segment").agg({"count"})

cltv_df.drop("customer_id",axis=1,inplace=True)

cltv_seg=cltv_df.loc[:,"monetary_cltv_avg":"cltv_segment"]

cltv_df.groupby("cltv_segment").agg({"mean","sum"})


#              recency_cltv_weekly                      T_weekly                   frequency               monetary_cltv_avg                 exp_sales_3_month               exp_sales_6_month               exp_average_profit                          cltv
#                              sum count      mean           sum count      mean         sum count    mean               sum count      mean               sum count    mean               sum count    mean                sum count      mean           sum count      mean
# cltv_segment
# C                   871926.28571  6649 131.13645 1021306.85714  6649 153.60308 25557.00000  6649 3.84374      657080.55705  6649  98.82397        2794.00051  6649 0.42021        5588.00102  6649 0.84043       694216.94221  6649 104.40923  589898.38896  6649  88.71987
# B                   605395.14286  6648  91.06425  734336.28571  6648 110.45973 30474.00000  6648 4.58394      943090.46254  6648 141.86078        3464.69286  6648 0.52116        6929.38572  6648 1.04233       987314.57628  6648 148.51302 1040469.56079  6648 156.50866
# A                   422709.00000  6648  63.58439  530346.85714  6648  79.77540 32835.00000  6648 4.93908     1374276.23531  6648 206.72025        4144.64987  6648 0.62344        8289.29974  6648 1.24689      1434879.87111  6648 215.83632 1797364.24284  6648 270.36165



#

df_= pd.read_csv("D:/12thTerm_DS_Bootcamp/3Week_CRM_Analytics/flo_rfm_project/flo_data_20k.csv")
df = df_.copy()


# FUNCTION
def cltv_function(dataframe):

    dataframe["total_purchases_number"] = dataframe["order_num_total_ever_online"] +  dataframe["order_num_total_ever_offline"]
    dataframe["total_customer_expense"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]

    # convert datetime
    convert =["first_order_date","last_order_date","last_order_date_online","last_order_date_offline"]
    dataframe[convert] = dataframe[convert].apply(pd.to_datetime)
    last_date = dt.datetime(2021,5,30)
    today_date = dt.datetime(2021, 6, 6)


    # Create CLTV Dataframe
    cltv_df = pd.DataFrame({"master_id": dataframe["master_id"],
                 "recency_cltv_weekly": ((dataframe["last_order_date"] - dataframe["first_order_date"]).dt.days)/7,
                 "T_weekly": ((today_date - dataframe["first_order_date"]).dt.days)/7,
                "frequency": dataframe["total_purchases_number"],
                "monetary_cltv_avg": dataframe["total_customer_expense"] / dataframe["total_purchases_number"]})


    # BG/NBD
    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(cltv_df['frequency'],
            cltv_df['recency_cltv_weekly'],
            cltv_df['T_weekly'])


    cltv_df["exp_sales_3_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4*3,
                                                            cltv_df['frequency'],
                                                            cltv_df['recency_cltv_weekly'],
                                                            cltv_df['T_weekly'])

    cltv_df["exp_sales_6_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4*6,
                                                            cltv_df['frequency'],
                                                            cltv_df['recency_cltv_weekly'],
                                                            cltv_df['T_weekly'])


    # Gamma-Gamma Sub Model
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])

    cltv_df["exp_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
    cltv = ggf.customer_lifetime_value(bgf,
                                           cltv_df['frequency'],
                                           cltv_df['recency_cltv_weekly'],
                                           cltv_df['T_weekly'],
                                           cltv_df['monetary_cltv_avg'],
                                           time=12,  # 12 aylık
                                           freq="W",  # T'nin frekans bilgisi.
                                           discount_rate=0.01)


    cltv_df["cltv"] = cltv
    cltv_df.sort_values("cltv",ascending=False)[:10]

    # Create Segmentation by CLTV
    cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D","C", "B", "A"])
    stats=["mean","sum","count"]
    cltv_df.groupby("cltv_segment").agg(stats)

    return cltv_df


cltv_df = cltv_function(df)

cltv_df.head()

from pathlib import Path
filepath = Path('D:/13th Data Science Bootcamp/3Week_CRM_Analytics/cltv.csv')
filepath.parent.mkdir(parents=True, exist_ok=True)
cltv_df.to_csv(filepath)


stats=["mean","sum","count"]
cltv_df.groupby("cltv_segment").agg(stats)



# Bonus CLTV & RFM Segmentation


rfm = pd.read_csv("D:/13th Data Science Bootcamp/3Week_CRM_Analytics/rfm_copy.csv",index_col=0)
rfm.head()



cltv_rfm = pd.merge(rfm, cltv_df, on=["master_id"])


len(cltv_rfm.loc[(cltv_rfm["segment"] == "champions" ) & (cltv_rfm["cltv_segment"] == "B")])

cltv_rfm.loc[(cltv_rfm["segment"] == "hibernating" ) & (cltv_rfm["cltv_segment"] == "A")]["master_id"]


cltv_rfm.columns




import seaborn as sns


def visual_bar_plot(dataframe,cat_col1,cat_col2,cat_col3,cat_col4,target):
    fig, qaxis = plt.subplots(1, 3, figsize=(14, 12))
    sns.barplot(x=cat_col1, y=target, hue=cat_col2, data=dataframe, ax = qaxis[0])
    qaxis[0].set_title(f"{cat_col1} vs {target} comparison by {cat_col2}")

    sns.barplot(x=cat_col1, y=target, hue=cat_col3, data=dataframe, ax = qaxis[1])
    qaxis[1].set_title(f"{cat_col1} and {target} comparison by {cat_col3}")

    sns.barplot(x=cat_col1, y=target, hue=cat_col4, data=dataframe, ax=qaxis[2])
    qaxis[2].set_title(f"{cat_col1} and {target} comparison by {cat_col4}")
    plt.show(block=True)


visual_bar_plot(cltv_rfm,"segment","cltv_segment","segment","cltv_segment","segment")



def visual_box_plot(dataframe,cat_col,num_col1,num_col2,target):
    fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(14, 12))
    sns.boxplot(x=cat_col, y=num_col1, hue=target, data=dataframe, ax=axis1)
    axis1.set_title(f"{cat_col} vs {num_col1} comparison by {target}")

    sns.boxplot(x=cat_col, y=num_col2, hue=target, data=dataframe, ax=axis2)
    axis2.set_title(f"{cat_col} and {num_col2} comparison by {target}")
    plt.show(block=True)

visual_box_plot(cltv_rfm,"segment","cltv","cltv","segment")



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.2f" % x)
pd.set_option("display.width", 500)
pd.set_option('display.expand_frame_repr', False)


#########################################################################
################# Exploratory Data Analysis ########################

df_ = pd.read_csv(r'datasets\Telco-Customer-Churn.csv')
df = df_.copy()

df.head()
df.info()
df.describe().T

# df.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T

df["Churn"] = df["Churn"].map({'No':0,'Yes':1})

def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=10, car_th=20)


non_numeric_mask = pd.to_numeric(df['TotalCharges'], errors='coerce').isna()
df[non_numeric_mask]


df[df['tenure']==0]


df['TotalCharges'] = df['TotalCharges'].replace(' ', 0).astype('float64')

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=10, car_th=20)

len(cat_cols)
len(num_cols)

# Observe the distribution of numerical and categorical variables in the data.

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


for col in cat_cols:
    cat_summary(df, col, plot=True)


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in num_cols:
    num_summary(df, col, plot=True)


plt.figure(figsize=(10, 6))
sns.violinplot(x='Churn', y='tenure', data=df)
plt.title('Relationship between Tenure and Churn')
plt.xlabel('Churn')
plt.ylabel('Tenure (months)')
plt.show()


plt.figure(figsize=(14, 6))
sns.countplot(x=pd.cut(df['tenure'], bins=24), hue='Churn', data=df, palette='Set1')
plt.title('Distribution of Tenure by Churn')
plt.xlabel('Tenure (months)')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.legend(title='Churn', labels=['No', 'Yes'])
plt.show()


# Examine the target variable with categorical variables.


def churn_sum_with_cat(df, cat_col, plot=False):
    print(pd.DataFrame(df.groupby([cat_col, 'Churn']).agg({'customerID':'nunique'})))
    if plot:
        sns.countplot(x=cat_col, hue='Churn', data=df)
        plt.show()


for col in cat_cols:
    churn_sum_with_cat(df, col, plot=True)


def churn_sum_with_num(df, num_col, plot=False):
    print(pd.DataFrame(df.groupby('Churn').agg({num_col:'mean'})))
    if plot:
        sns.boxplot(x='Churn', y=num_col, data=df)
        plt.show()


for col in num_cols:
    churn_sum_with_num(df, col, plot=True)


# Examine the outliers


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        print("{}:".format(col), "has outliers")
    else:
        print("{}:".format(col), "no outlier")

for col in num_cols:
    check_outlier(df, col, 0.25, 0.75)


#######################################################
#---- Contract | PaperlessBilling | Payment Method----#

contract = df[df['Churn'] == 1]['Contract'].value_counts()
contract = [contract[0] / sum(contract) * 100, contract[1] / sum(contract) * 100, contract[2] / sum(contract) * 100] # Month-to-month - One year - Two year

paperlessbilling = df[df['Churn'] == 1]['PaperlessBilling'].value_counts()
paperlessbilling = [paperlessbilling[0] / sum(paperlessbilling) * 100,paperlessbilling[1] / sum(paperlessbilling) * 100] # No - Yes

paymentmethod = df[df['Churn'] == 1]['PaymentMethod'].value_counts()
paymentmethod = [paymentmethod[0] / sum(paymentmethod) * 100, paymentmethod[1] / sum(paymentmethod) * 100,
            paymentmethod[2] / sum(paymentmethod) * 100, paymentmethod[3] / sum(paymentmethod) * 100]


# Plot Graphics #
colors = ['#E94B3C','#2D2926']
ax,fig = plt.subplots(nrows = 1,ncols = 3,figsize = (12,12))

plt.subplot(1,3,1)
plt.pie(contract,labels = ['Month-to-month','One year','Two year'],autopct='%1.1f%%',startangle = 90,explode = (0.1,0.1,0.1),colors = colors,
       wedgeprops = {'edgecolor' : 'black','linewidth': 1,'antialiased' : True})
plt.title('Contract');

plt.subplot(1,3,2)
plt.pie(paperlessbilling,labels = ['No', 'Yes'],autopct='%1.1f%%',startangle = 90,explode = (0.1,0),colors = colors,
       wedgeprops = {'edgecolor' : 'black','linewidth': 1,'antialiased' : True})
plt.title('PaperlessBilling');

plt.subplot(1,3,3)
plt.pie(paymentmethod,labels = ['Bank Transfer (automatic)','Credit Card (automatic)','Electronic check','Mailed check'],autopct='%1.1f%%',startangle = 90,explode = (0.1,0,0.1,0),colors = colors,
       wedgeprops = {'edgecolor' : 'black','linewidth': 1,'antialiased' : True})
plt.title('PaymentMethod');

plt.show()


# Tenure vs Contract | PaperlessBilling | Payment Method ###

l3 = ['Contract','PaperlessBilling','PaymentMethod'] # Payment Information
fig = plt.subplots(nrows = 1,ncols = 3,figsize = (25,7))
for i in range(len(l3)):
    plt.subplot(1,3,i + 1)
    ax = sns.boxplot(x = l3[i],y = 'tenure',data = df,hue = 'Churn',palette = colors);
    plt.title('tenure vs ' + l3[i]);

plt.show()


# Examine if there are any missing observations.

df.isna().any()

df[(df['PhoneService']=='No')&(df['MultipleLines']=='Yes')]

# internet service no olup online security yes ...

df[(df['InternetService'] == 'No') &
   (df['OnlineSecurity'] == 'Yes')]

df[(df['InternetService'] == 'No') &
   (df['DeviceProtection'] == 'Yes')]


#########################################################################
################## Feature Engineering ########################

# Creating new variables

# Create Family Status variable
df['FamilyStatus'] = df.apply(lambda row: 'Single' if row['Partner'] == 'No' and row['Dependents'] == 'No' else 'Couple'
if row['Partner'] == 'Yes' and row['Dependents'] == 'No' else 'Family', axis=1)

# Let's look at the churn and the tenure variable
plt.figure(figsize=(14, 6))
sns.countplot(x=pd.cut(df['tenure'], bins=24), hue='Churn', data=df, palette='Set1')
plt.title('Distribution of Tenure by Churn')
plt.xlabel('Tenure (months)')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Churn', labels=['No', 'Yes'])
plt.show()

# In the chart, the churn rate between 0-3 months is very high, there is a decrease between 3 and 24, and 24 and above seems to be more stable
# I added a new variable to divide it into 3 groups like this:

df['new_tenure'] = pd.cut(x=df['tenure'], bins=[-1, 3, 24, 73], labels=[0, 1, 2])

# Create bins for monthly charges (adjust bin edges as needed)
bins = [df['MonthlyCharges'].min(), 30, 60, 90, df['MonthlyCharges'].max()]
labels = ['Low', 'Medium', 'High', 'Very High']
df['MonthlyChargeLevel'] = pd.cut(df['MonthlyCharges'], bins=bins, labels=labels, right=False)

# Create 'IsAutomaticPayment' variable includes automatic payment methods
automatic_payment_methods = ['Bank transfer (automatic)', 'Credit card (automatic)']
df['IsAutomaticPayment'] = (df['PaymentMethod'].isin(automatic_payment_methods)).astype(int)

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=10, car_th=20)



# Encoding

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, categorical_cols=[x for x in cat_cols if x != 'Churn'], drop_first=True)


#### Feature Selection ####
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,mutual_info_classif

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=10, car_th=20)

features = df.loc[:,cat_cols]
target = df.loc[:,'Churn']

best_features = SelectKBest(score_func = chi2,k = 'all')
fit = best_features.fit(features,target)

featureScores = pd.DataFrame(data = fit.scores_,index = list(features.columns),columns = ['Chi Squared Score'])

plt.subplots(figsize = (12,7))
sns.heatmap(featureScores.sort_values(ascending = False,by = 'Chi Squared Score'),annot = True,cmap = colors,linewidths = 0.4,linecolor = 'black',fmt = '.2f');
plt.title('Selection of Categorical Features')
plt.tight_layout()

plt.show()


# Standardize numerical variables

rs = RobustScaler()
df[num_cols] = rs.fit_transform(df[num_cols])


#########################################################################
######################## Modeling ############################

# Build models with classification algorithms and examine accuracy scores.
# Choose the 4 best models.

X = df.drop(['Churn', 'customerID'], axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)


######## LOGISTIC REGRESSION ###########

log_model = LogisticRegression().fit(X, y)

y_pred = log_model.predict(X)


def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()


plot_confusion_matrix(y, y_pred)

print(classification_report(y, y_pred))
# accuracy : 0.81
# precision 0.67
# recall 0.55
# f1 0.60


y_prob = log_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)
# 0.85

# HOLDOUT

log_model = LogisticRegression().fit(X_train, y_train)

y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
# accuracy 0.81
# precision 0.69
# recall 0.52
# f1 0.59

RocCurveDisplay.from_estimator(log_model, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()

# AUC
roc_auc_score(y_test, y_prob)
#0.84

''' Since there is imbalance in the dataset, SMOTE method can be applied.
I got an error and could not proceed.

import imblearn
#from collections import Counter
#from imblearn.over_sampling import SMOTE
#from imblearn.under_sampling import RandomUnderSampler
#from imblearn.pipeline import Pipeline

#over = SMOTE(sampling_strategy = 1)

#f1 = df.iloc[:,:13].values
#t1 = df.iloc[:,13].values

#f1, t1 = over.fit_resample(f1, t1)
#Counter(t1)
'''


# CROSS VAL
log_model = LogisticRegression().fit(X, y)

cv_results = cross_validate(log_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()  # 0.81
cv_results['test_precision'].mean()  # 0.67
cv_results['test_recall'].mean() # 0.54
cv_results['test_f1'].mean() # 0.60
cv_results['test_roc_auc'].mean() # 0.85


#################################################
###############        KNN       ################

X = df.drop(['Churn', 'customerID'], axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)


knn_model = KNeighborsClassifier().fit(X, y)

# For Confusion matrix, y_pred:
y_pred = knn_model.predict(X)

# For AUC, y_prob:
y_prob = knn_model.predict_proba(X)[:, 1]

print(classification_report(y, y_pred))
# accuracy 0.84
# precision 0.72
# recall 0.62
# f1 0.67

roc_auc_score(y, y_prob)
# 0.90

# HOLDOUT

knn_model = KNeighborsClassifier().fit(X_train, y_train)

y_pred = knn_model.predict(X_test)
y_prob = knn_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
# accuracy 0.77
# precision 0.58
# recall 0.50
# f1 0.54

RocCurveDisplay.from_estimator(knn_model, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()

# AUC
roc_auc_score(y_test, y_prob)
#0.77

# CROSS VALIDATION
cv_results = cross_validate(knn_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean() # 0.77
cv_results['test_f1'].mean() # 0.54
cv_results['test_roc_auc'].mean() # 0.78



# Perform hyperparameter optimization with the models you choose and
# rebuild the model with the hyperparameters you found


knn_model = KNeighborsClassifier()
knn_model.get_params()

knn_params = {"n_neighbors": range(2, 50)}

knn_gs_best = GridSearchCV(knn_model,
                           knn_params,
                           cv=5,
                           n_jobs=-1,
                           verbose=1).fit(X, y)

knn_gs_best.best_params_


knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X, y)

cv_results = cross_validate(knn_final,
                            X,
                            y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean() # 0.80
cv_results['test_f1'].mean() # 0.59
cv_results['test_roc_auc'].mean() # 0.83

## Logistic Regression

log_model.get_params()

log_param = {
    'C': [0.01, 0.1, 1, 10],  # Regularization strength
    'penalty': ['l1', 'l2']  # Penalty type
}

log_best = GridSearchCV(log_model,
                        log_param,
                        cv=5,
                        n_jobs=-1,
                        verbose=1).fit(X, y)

log_best.best_params_


log_final = log_model.set_params(**log_best.best_params_).fit(X, y)

cv_results = cross_validate(log_final,
                            X,
                            y,
                            cv=5,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

cv_results['test_accuracy'].mean() # 0.81
cv_results['test_precision'].mean() # 0.67
cv_results['test_recall'].mean() # 0.54
cv_results['test_f1'].mean() # 0.59
cv_results['test_roc_auc'].mean() # 0.85

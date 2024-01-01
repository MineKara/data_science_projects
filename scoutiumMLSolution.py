
################################################################
# TALENT HUNT CLASSIFICATION WITH ARTIFICIAL LEARNING
################################################################

################################################################
# Business Problem
################################################################
# Predicting which class (average, highlighted) players are based on the points given to the characteristics of the football players watched by the scouts.


#####################################
# Dataset Story:
#####################################

# The data set consists of information containing the characteristics and scores of the football players evaluated by the scouts according to the characteristics of the football players observed in the matches from Scoutium.
# attributes: Contains the points that users who evaluate players give to the characteristics of each player they watch and evaluate in a match. (Independent variables)
#potential_labels: Contains the potential tags of users who evaluate players, which contain their final opinions about the players in each match. (target variable)
#9 Variable, 10730 Observations, 0.65 mb


#####################################
# Variables:
#####################################

# task_response_id: The set of a scout's evaluations of all players on a team's roster in a match.

# match_id: The id of the relevant match.

# evaluator_id: Evaluator's id.

# player_id: The id of the relevant player.

# position_id: The id of the position played by the relevant player in that match.

#1- Goalkeeper
#2- Center back
#3- Right back
#4- Left back
#5- Defensive midfielder
#6- Central midfielder
#7- Right wing
#8- Left wing
#9- Attacking midfielder
#10- Striker


# analysis_id: Set containing a scout's attribute evaluations of a player in a match.

# attribute_id: The id of each attribute on which players are evaluated.

# attribute_value: The value (point) given by a scout to an attribute of a player.

# potential_label: Label that indicates a scout's final decision on a player in a match. (target variable)




import pandas as pd
import numpy as np
from sklearn.model_selection import *
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_predict
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss
from sklearn.metrics import accuracy_score

pd.set_option('display.max_columns', None)

################################################################
# Preparation of Dataset
################################################################

################################################################
# Reading two csv files
################################################################

df = pd.read_csv("datasets\scoutium_attributes.csv", sep=";")

df2 = pd.read_csv("datasets\scoutium_attributes.csv", sep=";")

################################################################
# Combine the csv files using the merge function. (Perform the combination using 4 variables: "task_response_id", 'match_id', 'evaluator_id' "player_id".)
################################################################

dff = pd.merge(df, df2, how='left', on=["task_response_id", 'match_id', 'evaluator_id', "player_id"])



################################################################
# Görev 3: Remove the Goalkeeper (1) class in position_id from the dataset.
################################################################

dff = dff[dff["position_id"] != 1]


################################################################
# Remove the below_average class in the potential_label from the dataset. (below_average class constitutes 1% of the entire dataset)
################################################################

dff["potential_label"].value_counts()

dff = dff[dff["potential_label"] != "below_average"]

dff["attribute_value"].value_counts()
dff["attribute_id"].value_counts()
dff["attribute_id"].nunique()


dff["player_id"].value_counts()
dff["player_id"].nunique()

################################################################
# Create a table using the “pivot_table” function from the dataset you created. Manipulate this pivot table so that each row has one player.
################################################################

################################################################
# In each column, proceed to include the player's "position_id", "potential_label" and all "attribute_ids" of each player, respectively.
################################################################

pt = pd.pivot_table(dff, values="attribute_value", columns="attribute_id", index=["player_id","position_id","potential_label"])

################################################################
# Get rid of the index error by using the “reset_index” function and convert the names of the “attribute_id” columns to string. (df.columns.map(str))
################################################################

pt = pt.reset_index(drop=False)
pt.columns = pt.columns.map(str)
pt.head()

##################################
# EXPLORATORY DATA ANALYSIS
##################################


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(pt)


##################################
# Examine numerical and categorical variables.
##################################

##################################
# ANALYSIS OF CATEGORICAL VARIABLES
##################################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in ["position_id","potential_label"]:
    cat_summary(pt, col)

pt.head()
pt["position_id"].value_counts()


##################################
# ANALYSIS OF NUMERICAL VARIABLES
##################################

num_cols = pt.columns[3:]


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(pt, col, plot=True)


##################################
# Examine the target variable with numerical variables.
##################################

##################################
# ANALYSIS OF NUMERICAL VARIABLES BY TARGET
##################################

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(pt, "potential_label", col)


##################################
# Correlation
##################################

pt[num_cols].corr()

# Korelasyon Matrisi
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(pt[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show(block=True)


##################################
# Apply Feature Extraction
##################################

pt["min"] = pt[num_cols].min(axis=1)
pt["max"] = pt[num_cols].max(axis=1)
pt["sum"] = pt[num_cols].sum(axis=1)
pt["mean"] = pt[num_cols].mean(axis=1)
pt["median"] = pt[num_cols].median(axis=1)

pt.head()
pt.tail()


pt["mentality"] = pt["position_id"].apply(lambda x: "defender" if (x == 2) | (x == 5) | (x == 3) | (x == 4) else "attacker")



################################################################
# Encode the “potential_label” categories (average, highlighted) numerically using the Label Encoder function.
################################################################

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


labelEncoderCols = ["potential_label"]


binary_cols = [col for col in pt.columns if pt[col].dtypes == "O" and len(pt[col].unique()) == 2]

for col in binary_cols:
    label_encoder(pt, col)

for col in labelEncoderCols:
    pt = label_encoder(pt, col)




################################################################
# Apply standardScaler to scale the data in all “num_cols” variables
################################################################

num_cols = pt.columns[3:]

lst = ["min","max","sum","mean","median"]
num_cols = list(num_cols)

for i in lst:
    num_cols.append(i)



scaler = StandardScaler()
pt[num_cols] = scaler.fit_transform(pt[num_cols])
pt.head()
pt[num_cols]
pt.head()


################################################################
# Develop a machine learning model that predicts the potential tags of football players with minimum error based on the data set we have.
################################################################


y = pt["potential_label"]
X = pt.drop(["potential_label", "player_id"], axis=1)


models = [('LR', LogisticRegression(random_state=42)),
                   ('KNN', KNeighborsClassifier()),
                   #("SVC", SVC()),
                   ("CART", DecisionTreeClassifier(random_state=42)),
                   ("RF", RandomForestClassifier(random_state=42)),
                   #('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier(random_state=42)),
                   ('XGBoost', XGBClassifier( random_state=42, eval_metric='logloss')),
                   #('CatBoost', CatBoostClassifier(verbose=False)),
              ("LightGBM", LGBMClassifier(random_state=42,verbosity=-1))]



for name, model in models:
    print(name)
    for score in ["roc_auc", "f1", "precision", "recall", "accuracy"]:
        cvs = cross_val_score(model, X, y, scoring=score, cv=10).mean()
        print(score+" score:"+str(cvs))



################################################################
# Receiver Operation Characteristic(ROC) Curve - Visualization
################################################################
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,test_size = 0.2,
                                                 random_state = 42)

def roc_curve_parameter(model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    [fpr, tpr, thr] = roc_curve(y_test, y_pred_proba)
    print('Train/Test split results:')
    print(model.__class__.__name__ + " accuracy is %2.3f" % accuracy_score(y_test, y_pred))
    print(model.__class__.__name__ + " log_loss is %2.3f" % log_loss(y_test, y_pred_proba))
    print(model.__class__.__name__ + " auc is %2.3f" % auc(fpr, tpr))

    idx = np.min(np.where(tpr > 0.95))  # index of the first threshold for which the sensibility > 0.95

    plt.figure(figsize=(15, 8))
    plt.plot(fpr, tpr, color='coral', label='ROC curve (area = %0.3f)' % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot([0, fpr[idx]], [tpr[idx], tpr[idx]], 'k--', color='blue')
    plt.plot([fpr[idx], fpr[idx]], [0, tpr[idx]], 'k--', color='blue')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - specificity)', fontsize=14)
    plt.ylabel('True Positive Rate (recall)', fontsize=14)
    plt.title('{} Receiver operating characteristic (ROC) curve'.format(model))
    plt.legend(loc="lower right")
    plt.show(block=True)

    print("Using a threshold of %.3f " % thr[idx] + "guarantees a sensitivity of %.3f " % tpr[idx] +
          "and a specificity of %.3f" % (1 - fpr[idx]) +
          ", i.e. a false positive rate of %.2f%%." % (np.array(fpr[idx]) * 100))

for name, model in models:
    roc_curve_parameter(model)


################################################################
# Plot the order of the features using the feature_importance function, which indicates the importance level of the variables.
################################################################

tree_models = [("CART", DecisionTreeClassifier(random_state=42)),
                   ("RF", RandomForestClassifier(random_state=42)),
                   ('GBM', GradientBoostingClassifier(random_state=42)),
                   ('XGBoost', XGBClassifier( random_state=42, eval_metric='logloss')),
              ("LightGBM", LGBMClassifier(random_state=42,verbosity=-1))]



# feature importance
def plot_importance_model(model, features, num=len(X), save=False):
    model.fit(X,y)
    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("{} Features".format(model))
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig("importances.png")


for name, model in tree_models:
    plot_importance_model(model, X,10)


####################################################################33,


################################################################
# Perform Hyperparameter Optimization
################################################################

################################################################
# LightGBM Hyperparameter
################################################################

lgbm_model = LGBMClassifier(random_state=46,verbosity=-1)

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [100,300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=10, n_jobs=-1, verbose=True).fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


###### Hyperparameter new values

lgbm_params = {"learning_rate": [0.01, 0.02, 0.05, 0.1],
               "n_estimators": [200, 300, 350, 400],
               "colsample_bytree": [0.9, 0.8, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()



################################################################
# RandomForestClassifier Hyperparameter
################################################################

rf_model = RandomForestClassifier(random_state=17)

rf_random_params = {"max_depth": np.random.randint(5, 50, 10),
                    "max_features": [3, 5, 7, "auto", "sqrt"],
                    "min_samples_split": np.random.randint(2, 50, 20),
                    "n_estimators": [int(x) for x in np.linspace(start=200, stop=1500, num=10)]}

rf_random = RandomizedSearchCV(estimator=rf_model,
                               param_distributions=rf_random_params,
                               n_iter=100,  # denenecek parametre sayısı
                               cv=3,
                               verbose=True,
                               random_state=42,
                               n_jobs=-1)

rf_random_final = rf_model.set_params(**rf_random.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(rf_random_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


################################################################
# GradientBoosting Hyperparameter
################################################################

xgboost_model = XGBClassifier(random_state=17, use_label_encoder=False)


xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 500, 1000],
                  "colsample_bytree": [0.7, 1]}

xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(xgboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()



################################################################
# FEATURE IMPORTANCE FOR HYPERPARAMETER MODELS
################################################################

def plot_importance(model, features, num=len(X), save=False):

    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig("importances.png")


plot_importance(rf_random_final, X)
plot_importance(lgbm_final, X)
plot_importance(xgboost_final, X)



################################################################
# BONUS : PCA(PRINCIPAL COMPONENT ANALYSIS)
################################################################

X.shape
# Out[89]: (271, 41)

################################################################
# Adım 1: PCA - Dimension Reduction
################################################################
# First, we want to perform dimensionality reduction with PCA. At this point, we calculate the variance ratios of our relevant independent variable X
# We analyze with PCA.

from sklearn.decomposition import PCA
pca = PCA().fit(X)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Bileşen Sayısını")
plt.ylabel("Kümülatif Varyans Oranı")
plt.show(block=True)


pca = PCA(n_components=6)
pca_fit = pca.fit_transform(X)
pca.explained_variance_ratio_
np.cumsum(pca.explained_variance_ratio_)
# array([0.66367736, 0.8391152 , 0.86304915, 0.87987436, 0.89243661,
#        0.90321065])
# Çıkan sonuca göre 6 yeni değişken ile %90 oranında bir varyansı yakaladığımızı tespit edebiliyoruz.



################################################################
# We combine the new 6 independent variables we obtained with our dependent variable and create a new data set called "pca_df".
################################################################

pca_df = pd.concat([pd.DataFrame(pca_fit, columns=["PC1","PC2","PC3","PC4","PC5","PC6"]),
                      pt["potential_label"]], axis=1)


pca_df.head()
# Out[92]:
#         PC1       PC2       PC3       PC4       PC5       PC6  potential_label
# 0  5.223161  1.775925 -1.075765  0.810953 -0.412815  0.085568                0
# 1 -4.775504  3.147241  0.748675 -0.022634 -0.241394 -0.876595                0
# 2 -4.502561 -2.694821 -0.451921 -0.620796 -0.331664  0.146943                0
# 3 -4.147895 -2.207057  0.447023  1.241783  0.901357 -0.238358                0
# 4 -5.087387  3.669200 -0.461094 -0.262896 -0.614225  0.878350                0


################################################################
# We develop a machine learning model that predicts the potential labels of football players with minimum error on the "pca_df" data set.
# We observe our model success evaluation metrics on each machine learning model.
################################################################

y = pca_df["potential_label"]
X = pca_df.drop(["potential_label"], axis=1)


models = [('LR', LogisticRegression(random_state=42)),
                   ('KNN', KNeighborsClassifier()),
                   #("SVC", SVC()),
                   ("CART", DecisionTreeClassifier(random_state=42)),
                   ("RF", RandomForestClassifier(random_state=42)),
                   #('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier(random_state=42)),
                   ('XGBoost', XGBClassifier( random_state=42, eval_metric='logloss')),
                   #('CatBoost', CatBoostClassifier(verbose=False)),
              ("LightGBM", LGBMClassifier(random_state=42,verbosity=-1))]


for name, model in models:
    print(name)
    for score in ["roc_auc", "f1", "precision", "recall", "accuracy"]:
        cvs = cross_val_score(model, X, y, scoring=score, cv=10).mean()
        print(score+" score:"+str(cvs))



################################################################
# Observe the effect of each variable on the model through the feature_importance function of the new models created.
################################################################


def plot_importance_model(model, features, num=len(X), save=False):
    model.fit(X,y)
    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("{} Features".format(model))
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig("importances.png")


for name, model in tree_models:
    plot_importance_model(model, X,10)
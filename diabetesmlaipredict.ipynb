#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.chdir('YourPath')    # Set working directory
#importing dataset
df = pd.read_csv('diabetes.csv')
df.head()
df.describe()
df.info()
df.shape
#replace NaN
df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
df['Glucose'].fillna(df['Glucose'].mean(), inplace=True)
df['BloodPressure'].fillna(df['BloodPressure'].mean(), inplace=True)
df['SkinThickness'].fillna(df['SkinThickness'].mean(), inplace=True)
df['Insulin'].fillna(df['Insulin'].mean(), inplace=True)
df['BMI'].fillna(df['BMI'].mean(), inplace=True)
#correlation heatmap
plt.figure(figsize=(16, 6))
# define the mask to set the values in the upper triangle to True
mask = np.triu(np.ones_like(df.corr(), dtype=np.bool))
heatmap = sns.heatmap(df.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Triangle Correlation Heatmap', fontdict={'fontsize':18}, pad=16);
X=df.drop('Outcome',axis=1)
y=df['Outcome']
plt.figure(figsize= (10,6))
fig = y.value_counts(normalize = True).plot.pie(autopct='%1.2f%%')
plt.title("Pie-chart showing Outcome", fontdict={'fontsize': 20, 'fontweight' : 5, 'color' : 'Green'})
fig.legend(title="Outcome",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))
plt.show()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=1)
from sklearn.preprocessing import StandardScaler
scaling_x=StandardScaler()
X_train=scaling_x.fit_transform(X_train)
X_test=scaling_x.transform(X_test)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
model_rfc=rfc.fit(X_train, y_train)
y_pred_rfc=rfc.predict(X_test)
rfc.score(X_test, y_test)
from sklearn.metrics import classification_report, confusion_matrix
cf_matrix=confusion_matrix(y_test, y_pred_rfc)
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
y_pred=y_pred_rfc
print ('Accuracy: ', accuracy_score(y_test, y_pred))
print ('F1 score: ', f1_score(y_test, y_pred))
print ('Recall: ', recall_score(y_test, y_pred))
print ('Precision: ', precision_score(y_test, y_pred))
print ('\n clasification report:\n', classification_report(y_test,y_pred))
print ('\n confussion matrix:\n',confusion_matrix(y_test, y_pred))
sns.distplot(df.BMI)
plt.show()
# BMI vs Glucose

plt.figure(figsize= [10,6])
plt.scatter(df["BMI"], df["Glucose"], alpha = 0.5)
plt.title("Scatter plot analysing BMI vs Glucose\n", fontdict={'fontsize': 20, 'fontweight' : 5, 'color' : 'Green'})
plt.xlabel("BMI", fontdict={'fontsize': 12, 'fontweight' : 5, 'color' : 'Black'})
plt.ylabel("Glucose", fontdict={'fontsize': 12, 'fontweight' : 5, 'color' : 'Black'} )
plt.show()
from mlxtend.plotting import plot_decision_regions
def classify_with_rfc(X,Y):
    x = df[[X,Y]].values
    y = df['Outcome'].astype(int).values
    rfc = RandomForestClassifier()
    rfc.fit(x,y)
    # Plotting decision region
    plot_decision_regions(x, y, clf=rfc, legend=2)
    # Adding axes annotations
    plt.xlabel(X)
    plt.ylabel(Y)
    plt.show()
    
feat = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']
size = len(feat)
for i in range(0,size):
    for j in range(i+1,size):
        classify_with_rfc(feat[i],feat[j])
import scikitplot as skplt
skplt.estimators.plot_learning_curve(RandomForestClassifier(), X_test, y_pred_rfc,
                                     cv=7, shuffle=True, scoring="accuracy",
                                     n_jobs=-1, figsize=(6,4), title_fontsize="large", text_fontsize="large",
                                     title="RandomForestClassifier Learning Curve");
from sklearn.metrics import roc_curve
y_pred_proba = rfc.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot([0,1],[0,1],'k-')
plt.plot(fpr,tpr, label='Knn')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('RFC ROC curve')
plt.show()
from sklearn import svm
clf = svm.SVC()
clf.fit(X_train, y_train)
clf.predict(X_test)
clf.score(X_test, y_test)
from sklearn.linear_model import LogisticRegression
lreg = LogisticRegression()
lreg.fit(X_train, y_train)
lreg.predict(X_test)
lreg.score(X_test, y_test)
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=0)
dtc.fit(X_train, y_train)

y_pred_dtc = dtc.predict(X_test)
dtc.score(X_test, y_test)
from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
xgb.predict(X_test)
xgb.score(X_test, y_test)


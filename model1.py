import sagemaker
import boto3
from sagemaker import get_execution_role

region = boto3.Session().region_name

session = sagemaker.Session()
bucket = session.default_bucket()
prefix = 'sagemaker/autopilot-dm'

role = get_execution_role()

sm = boto3.Session().client(service_name='sagemaker',region_name=region)
containers = {'us-west-2': '433757028032.dkr.ecr.us-west-2.amazonaws.com/xgboost:latest',
              'us-east-1': '811284229777.dkr.ecr.us-east-1.amazonaws.com/xgboost:latest',
              'us-east-2': '825641698319.dkr.ecr.us-east-2.amazonaws.com/xgboost:latest',
              'eu-west-1': '685385470294.dkr.ecr.eu-west-1.amazonaws.com/xgboost:latest'}
xgboost = sagemaker.estimator.Estimator(containers[region],role, train_instance_count=1, train_instance_type='ml.m4.xlarge',output_path='s3://{}/{}/output'.format(bucket, prefix),sagemaker_session=session)

!pip install imblearn
!pip install xgboost
import numpy as np
import pandas as pd

#Machine Learning Packages
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


#Model Selection/Assessment Packages
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.metrics import classification_report, precision_recall_curve, confusion_matrix, auc
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

#Other Packages
from imblearn.datasets import fetch_datasets
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows',500)

s3_input_train = sagemaker.s3_input(s3_data='s3://{}/{}/clean_input_onlynum'.format(bucket, prefix), content_type='csv')

train_data, test_data = np.split(train.sample(frac=1, random_state=1729), [int(0.7 * len(train))])
print(train_data.shape, test_data.shape)

train['Current Loan Amount'] = ((train['Current Loan Amount'] - train['Current Loan Amount'].mean())/(train['Current Loan Amount'].std()))  
    
train['Credit Score'] = ((train['Credit Score'] - train['Credit Score'].mean())/(train['Credit Score'].std()))      
train['Annual Income'] = ((train['Annual Income'] - train['Annual Income'].mean())/(train['Annual Income'].std()))   
train['Monthly Debt'] = ((train['Monthly Debt'] - train['Monthly Debt'].mean())/(train['Monthly Debt'].std()))
train['Years of Credit History'] = ((train['Years of Credit History'] - train['Years of Credit History'].mean())/(train['Years of Credit History'].std()))
train['Months since last delinquent'] = ((train['Months since last delinquent'] - train['Months since last delinquent'].mean())/(train['Months since last delinquent'].std()))
train['Number of Open Accounts'] = ((train['Number of Open Accounts'] - train['Number of Open Accounts'].mean())/(train['Number of Open Accounts'].std()))
train['Number of Credit Problems'] = ((train['Number of Credit Problems'] - train['Number of Credit Problems'].mean())/(train['Number of Credit Problems'].std()))
train['Current Credit Balance'] = ((train['Current Credit Balance'] - train['Current Credit Balance'].mean())/(train['Current Credit Balance'].std()))
train['Maximum Open Credit'] = ((train['Maximum Open Credit'] - train['Maximum Open Credit'].mean())/(train['Maximum Open Credit'].std()))      
train['Bankruptcies'] = ((train['Bankruptcies'] - train['Bankruptcies'].mean())/(train['Bankruptcies'].std()))      
train['Tax Liens'] = ((train['Tax Liens'] - train['Tax Liens'].mean())/(train['Tax Liens'].std()))   


    test_data['Current Loan Amount'] = ((test_data['Current Loan Amount'] - test_data['Current Loan Amount'].mean())/(test_data['Current Loan Amount'].std()))  
    
test_data['Credit Score'] = ((test_data['Credit Score'] - test_data['Credit Score'].mean())/(test_data['Credit Score'].std()))      
test_data['Annual Income'] = ((test_data['Annual Income'] - test_data['Annual Income'].mean())/(test_data['Annual Income'].std()))   
test_data['Monthly Debt'] = ((test_data['Monthly Debt'] - test_data['Monthly Debt'].mean())/(test_data['Monthly Debt'].std()))
test_data['Years of Credit History'] = ((test_data['Years of Credit History'] - test_data['Years of Credit History'].mean())/(test_data['Years of Credit History'].std()))
test_data['Months since last delinquent'] = ((test_data['Months since last delinquent'] - test_data['Months since last delinquent'].mean())/(test_data['Months since last delinquent'].std()))
test_data['Number of Open Accounts'] = ((test_data['Number of Open Accounts'] - test_data['Number of Open Accounts'].mean())/(test_data['Number of Open Accounts'].std()))
test_data['Number of Credit Problems'] = ((test_data['Number of Credit Problems'] - test_data['Number of Credit Problems'].mean())/(test_data['Number of Credit Problems'].std()))
test_data['Current Credit Balance'] = ((test_data['Current Credit Balance'] - test_data['Current Credit Balance'].mean())/(test_data['Current Credit Balance'].std()))
test_data['Maximum Open Credit'] = ((test_data['Maximum Open Credit'] - test_data['Maximum Open Credit'].mean())/(test_data['Maximum Open Credit'].std()))      
test_data['Bankruptcies'] = ((test_data['Bankruptcies'] - test_data['Bankruptcies'].mean())/(test_data['Bankruptcies'].std()))      
test_data['Tax Liens'] = ((test_data['Tax Liens'] - test_data['Tax Liens'].mean())/(test_data['Tax Liens'].std()))   


from imblearn.over_sampling import SMOTE

y_train = train['Loan Status']
y_test = test_data['Loan Status']
X_train = train.drop(['Loan Status'], axis=1)
X_test = test_data.drop(['Loan Status'], axis=1)


os = SMOTE(random_state=0)

columns = X_train.columns

os_train_data_X, os_train_data_y = os.fit_sample(X_train, y_train)
os_train_data_X = pd.DataFrame(data=os_train_data_X, columns=columns )
os_train_data_y = pd.DataFrame(data=os_train_data_y, columns=['Loan Status'])

# classifiers = {
#     "Logisitic Regression Classifier": LogisticRegression(n_jobs=3,random_state=0),
#     "K-Nearest-Neighbors Classifier": KNeighborsClassifier(n_jobs=3),
#     "Decision Tree Classifier": DecisionTreeClassifier(),
#     "Random Forest Classifier": RandomForestClassifier(n_estimators=750, n_jobs=3, random_state=0),
#     "Neural Network Classifier": MLPClassifier(random_state=0),
#     "XGBoost Classifier": XGBClassifier(n_jobs=3),
#     "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=750, random_state=0),
#     "AdaBoost Classifier": AdaBoostClassifier(n_estimators=100, base_estimator=RandomForestClassifier(),learning_rate=1)
# }
classifiers = {
    "XGBoost Classifier": XGBClassifier(n_jobs=3),
}

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

   for key, classifier in classifiers.items():
    classifier.fit(os_train_data_X, os_train_data_y)
    y_score = classifier.predict(X_test)
    print("="*125)
    print(color.BOLD + color.UNDERLINE + key + color.END, "\n")
    print(color.BOLD + "Confusion Matrix: \n" + color.END)
    print(confusion_matrix(y_test, y_score), "\n" )
    print(classification_report(y_test, y_score), "\n")
    acc = round(accuracy_score(y_test, y_score), 3)
    rec = round(recall_score(y_test, y_score), 3)
    prec = round(precision_score(y_test, y_score), 3)
    print(color.BOLD + f'Accuracy: {acc}')
    print(f'Recall: {rec}')
    print(f'Precision: {prec}' + color.END)
    print("="*125, '\n\n')

    model = XGBClassifier()
model.fit(os_train_data_X, os_train_data_y)


print(model)

y_pred = model.predict(os_train_data_X)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(os_train_data_y, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
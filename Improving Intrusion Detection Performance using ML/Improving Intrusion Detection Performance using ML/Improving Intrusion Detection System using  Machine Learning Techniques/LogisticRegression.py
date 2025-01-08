import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')

# Reading CSV file
data_train = pd.read_csv("C:\\Users\\srira\\Downloads\\Major Project (1)\\Major Project\\Major Project\\NSS-KDD.txt")

# Define columns to use
selected_features = [
    'duration', 'src_bytes', 'dst_bytes', 'hot', 'num_failed_logins', 'logged_in', 
    'num_compromised', 'root_shell', 'su_attempted', 'num_file_creations', 
    'num_shells', 'num_access_files', 'count', 'srv_count', 'serror_rate', 
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'dst_host_count', 
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'outcome', 'level'
]
columns = (['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot'
,'num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations'
,'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate'
,'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count'
,'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate'
,'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','outcome','level'])

data_train.columns = columns
data_train = data_train[selected_features]

# Preprocessing - Scaling
def Scaling(df_num, cols):
    std_scaler = RobustScaler()
    std_scaler_temp = std_scaler.fit_transform(df_num)
    std_df = pd.DataFrame(std_scaler_temp, columns=cols)
    return std_df

cat_cols = ['outcome', 'logged_in', 'root_shell', 'su_attempted']  # Adjust according to selected features
def preprocess(dataframe):
    df_num = dataframe.drop(cat_cols, axis=1)
    num_cols = df_num.columns
    scaled_df = Scaling(df_num, num_cols)
    
    dataframe.drop(labels=num_cols, axis="columns", inplace=True)
    dataframe[num_cols] = scaled_df[num_cols]
    
    outcome_mapping = {"normal": 0, "dos": 1, "probe": 2, "r2l": 3, "u2r": 4}
    dataframe['outcome'] = dataframe['outcome'].map(outcome_mapping).astype(int)
    
    return dataframe

scaled_train = preprocess(data_train)

# Split data
x = scaled_train.drop(['outcome', 'level'], axis=1).values
y = scaled_train['outcome'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Evaluate classification model
kernal_evals = dict()
def evaluate_classification(model, name, X_train, X_test, y_train, y_test):
    train_accuracy = metrics.accuracy_score(y_train, model.predict(X_train))
    test_accuracy = metrics.accuracy_score(y_test, model.predict(X_test))
    train_precision = metrics.precision_score(y_train, model.predict(X_train), average='macro')
    test_precision = metrics.precision_score(y_test, model.predict(X_test), average='macro')
    train_recall = metrics.recall_score(y_train, model.predict(X_train), average='macro')
    test_recall = metrics.recall_score(y_test, model.predict(X_test), average='macro')
    train_f1_score=metrics.f1_score(y_train,model.predict(X_train),average='macro')
    test_f1_score=metrics.f1_score(y_test,model.predict(X_test),average='macro')
    kernal_evals[str(name)] = [train_accuracy, test_accuracy, train_precision, test_precision, train_recall, test_recall]
    print(f"Training Accuracy {name}: {train_accuracy * 100:.2f}%  |  Test Accuracy {name}: {test_accuracy * 100:.2f}%")
    print(f"Training Precision {name}: {train_precision * 100:.2f}%  |  Test Precision {name}: {test_precision * 100:.2f}%")
    print(f"Training Recall {name}: {train_recall * 100:.2f}%  |  Test Recall {name}: {test_recall * 100:.2f}%")
    print(f"Training F1-Score {name}: {train_f1_score * 100:.2f}%  |  Test F1-Score {name}: {test_f1_score * 100:.2f}%")
    
    actual = y_test
    predicted = model.predict(X_test)
    confusion_matrix = metrics.confusion_matrix(actual, predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[f'Class {i}' for i in range(5)])

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.grid(False)
    cm_display.plot(ax=ax)
    plt.title(f'Confusion Matrix for {name}')
    plt.show()

def f_importances(coef, names, top=-1):
    imp = coef
    imp, names = zip(*sorted(list(zip(imp, names))))

    # Show all features
    if top == -1:
        top = len(names)
    
    plt.figure(figsize=(10,10))
    plt.barh(range(top), imp[::-1][0:top], align='center')
    plt.yticks(range(top), names[::-1][0:top])
    plt.title('feature importances for Decision Tree')
    plt.show()

features_names = data_train.drop(['outcome', 'level'] , axis = 1)

lr = LogisticRegression().fit(x_train, y_train)
evaluate_classification(lr, "Logistic Regression", x_train, x_test, y_train, y_test)
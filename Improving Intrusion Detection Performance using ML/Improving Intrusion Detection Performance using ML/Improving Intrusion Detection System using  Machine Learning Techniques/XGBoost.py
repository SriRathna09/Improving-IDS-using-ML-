import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

# Display all columns
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
columns = (['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot',
            'num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations',
            'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate',
            'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count',
            'dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate',
            'dst_host_srv_rerror_rate','outcome','level'])

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
y_reg = scaled_train['level'].values

# PCA for dimensionality reduction
pca = PCA(n_components=20)
pca = pca.fit(x)
x_reduced = pca.transform(x)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train_reduced, x_test_reduced, y_train_reduced, y_test_reduced = train_test_split(x_reduced, y, test_size=0.2, random_state=42)
x_train_reg, x_test_reg, y_train_reg, y_test_reg = train_test_split(x, y_reg, test_size=0.2, random_state=42)

# XGBoost Regressor Model
xg_r = xgb.XGBRegressor(objective='reg:linear', n_estimators=20).fit(x_train_reg, y_train_reg)

# Calculate regression performance metrics
train_rmse = np.sqrt(metrics.mean_squared_error(y_train_reg, xg_r.predict(x_train_reg)))
test_rmse = np.sqrt(metrics.mean_squared_error(y_test_reg, xg_r.predict(x_test_reg)))
train_r2 = metrics.r2_score(y_train_reg, xg_r.predict(x_train_reg))
test_r2 = metrics.r2_score(y_test_reg, xg_r.predict(x_test_reg))

print(f"Training RMSE XGBOOST: {train_rmse}")
print(f"Test RMSE XGBOOST: {test_rmse}")
print(f"Training R^2 XGBOOST: {train_r2}")
print(f"Test R^2 XGBOOST: {test_r2}")

# Plot predictions vs actual values for visualization
y_pred = xg_r.predict(x_test_reg)
df = pd.DataFrame({"Y_test": y_test_reg, "Y_pred": y_pred})

plt.figure(figsize=(16, 8))
plt.plot(df[:80], marker='o')  # Display the first 80 predictions vs actual values
plt.legend(['Actual', 'Predicted'])
plt.title('Actual vs Predicted Values (First 80 samples)')
plt.xlabel('Sample index')
plt.ylabel('Level')
plt.show()

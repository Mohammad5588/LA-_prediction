import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the training data into a Pandas dataframe
df_train = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")

# Load the testing data into a Pandas dataframe
df_test = pd.read_csv("test_Y3wMUE5_7gLdaTN.csv")

# Exploratory data analysis (EDA)
sns.pairplot(df_train)
plt.show()

# Prepare the data for model training
# Split the training data into input features (X) and output target (y)
X_train = df_train.drop("LoanAmount", axis=1)
y_train = df_train["LoanAmount"]

# Split the testing data into input features (X) and output target (y)
X_test = df_test.drop("LoanAmount", axis=1)
y_test = df_test["LoanAmount"]

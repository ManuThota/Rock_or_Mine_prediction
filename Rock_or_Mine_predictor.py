import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv('Copy of sonar data.csv', header = None)
# To display some of the data
# print(data.head())

print(f"The shape of the Dataset : {data.shape}")

# To describe the Dataset
# print(data.describe())

# Splitting the Data
x = data.drop(columns=60, axis=1)
y = data[60]

# Splitting the data into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1,stratify=y, random_state=42)

# Model
model = LogisticRegression()

# Fitting the training data into the Model
model.fit(x_train, y_train)

# Checking Accuracy of Model over Training Data
y_train_prediction = model.predict(x_train)
training_accuracy_score = accuracy_score(y_train, y_train_prediction)
print(f"Training Data Accuracy Score : {training_accuracy_score}")

# Checking Accuracy of Model over Testing Data
y_test_prediction = model.predict(x_test)
testing_accuracy_score = accuracy_score(y_test, y_test_prediction)
print(f"Testing Data Accuracy Score : {testing_accuracy_score}")

# Predictive System

input_data = (0.0291,0.0400,0.0771,0.0809,0.0521,0.1051,0.0145,0.0674,0.1294,0.1146,0.0942,0.0794,0.0252,0.1191,0.1045,0.2050,0.1556,0.2690,0.3784,0.4024,0.3470,0.1395,0.1208,0.2827,0.1500,0.2626,0.4468,0.7520,0.9036,0.7812,0.4766,0.2483,0.5372,0.6279,0.3647,0.4572,0.6359,0.6474,0.5520,0.3253,0.2292,0.0653,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0056,0.0237,0.0204,0.0050,0.0137,0.0164,0.0081,0.0139,0.0111)

# Changing the input data into numpy array
input_data_into_np_array = np.asarray(input_data)

# Reshaping the array for predicting one instance
input_data_reshaped = input_data_into_np_array.reshape(1, -1)

# Prediction
prediction = model.predict(input_data_reshaped)
if prediction[0] == 'R':
    print("The Detected object is ROCK")
else:
    print("The Detected object is MINE")

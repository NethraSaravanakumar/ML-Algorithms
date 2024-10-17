# Import necessary libraries for data manipulation, encoding, model training, and evaluation
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

# Load dataset and handle missing values (replace '?' with 'unknown')
obj_df = (pd.read_csv('data.csv')
        .replace({'?': 'unknown'}))

# Drop any columns containing 'Unnamed' (usually unnecessary or index columns)
obj_df.drop(obj_df.columns[obj_df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

# Split data into features (X) and target (y)
X = obj_df.drop(columns='provstate')  # Features (independent variables)
y = obj_df['provstate'].copy()  # Target (dependent variable)

# Split the dataset into training and testing sets (50% test size)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Encode categorical features in the training set using MultiColumnLabelEncoder (custom class to handle multiple columns)
le = MultiColumnLabelEncoder()
X_train_le = le.fit_transform(X_train)  # Transform the training set
X_train_le.head()  # Display the first few rows of the encoded training data

# Encode categorical features in the testing set
le = MultiColumnLabelEncoder()
X_test_le = le.fit_transform(X_test)  # Transform the test set
X_test_le.head()  # Display the first few rows of the encoded testing data

# Encode target labels (y) in the training set
le = LabelEncoder()
y_train_le = le.fit_transform(y_train)  # Transform the training labels
y_train_le  # Display the encoded training labels

# Encode target labels (y) in the testing set
le = LabelEncoder()
y_test_le = le.fit_transform(y_test)  # Transform the test labels
y_test_le  # Display the encoded test labels

# Initialize the XGBoost classifier
classifier = XGBClassifier()

# Train the model using the encoded training set (X and y)
classifier.fit(X_train_le, y_train_le)

# Predict the target labels on the test set
y_pred = classifier.predict(X_test_le)

# Evaluate the model using cross-validation (10 folds) on the training set
accuracies = cross_val_score(estimator = classifier, X = X_train_le, y = y_train_le, cv = 10)

# Print the average accuracy and the standard deviation of the model
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

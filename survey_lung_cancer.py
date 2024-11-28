import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import jaccard_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import itertools

# Load data and preprocess
data = pd.read_csv('survey lung cancer.csv')
label_encoder = LabelEncoder()
for col in ['LUNG_CANCER', 'GENDER']:
    data[col] = label_encoder.fit_transform(data[col])

# Feature Selection (Example using all features for now)
features = ["GENDER", "AGE", "SMOKING", "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE",
            "CHRONIC DISEASE", "WHEEZING", "ALCOHOL CONSUMING", "COUGHING",
            "SHORTNESS OF BREATH", "SWALLOWING DIFFICULTY", "CHEST PAIN", "ALLERGY ", "FATIGUE "]
X = data[features]
y = data["LUNG_CANCER"]

# Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter Tuning
param_grid = {'n_neighbors': list(range(1, 31)), 'weights': ['uniform', 'distance']}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_knn = grid_search.best_estimator_

# Prediction
y_pred = best_knn.predict(X_test)

# Evaluation
print(classification_report(y_test, y_pred))
print("Jaccard Score:", jaccard_score(y_test, y_pred))

# Confusion Matrix
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


cnf_matrix = confusion_matrix(y_test, y_pred)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['DISEASE=0', 'DISEASE=1'])
plt.show()

print("Best parameters:", grid_search.best_params_)
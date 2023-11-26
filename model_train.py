import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import data.modelData as data

class PredictShape:
    def __init__(self, featureVector) -> None:
        self._new_feature = featureVector

    def train_model(self):
        """
        Training a DTC for prediction of facial shape 
        """
        # Get model data 
        X  = data.X
        y = data.y
        
        #Prepare the Data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Select DTC
        model = DecisionTreeClassifier()

        # Train the Model
        model.fit(X_train, y_train)

        # This is where the face values to be predicted will go 
        new_feature_vector = self._new_feature

        # Make Prediction
        predictions = model.predict([new_feature_vector])
        y_test_predictions = model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_test_predictions)

        print(f"Accuracy: {accuracy * 100:.2f}%")
        print("Prediction:", predictions)

test = PredictShape([0.7689243027888446, 0.701195219123506, 0.6593625498007968, 0.36254980079681276, 1.3005181347150259, 56.740016844595104])
test.train_model()

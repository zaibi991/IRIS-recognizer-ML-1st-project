from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import accuracy_score,classification_report,recall_score,confusion_matrix,f1_score
from sklearn.preprocessing import LabelEncoder
iris_data = pd.read_csv(r'C:\Users\gulza\OneDrive\Desktop\IRIS.csv')
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X = iris_data[features]
Y = iris_data['species'].astype('category')

#encoding the species column to computer understandable language
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)

#splitting data
X_train,X_test,Y_train,Y_test = train_test_split (X,Y_encoded,test_size = 0.3,random_state = 42)

#random forest model
model = RandomForestClassifier(random_state = 42)
model.fit(X_train,Y_train)
Y_prediction = model.predict (X_test)
accuracy = accuracy_score (Y_test,Y_prediction)

#accuracy and all the tests to check
print ("Accuracy: ",accuracy)
print ("classification report ", classification_report(Y_test,Y_prediction))
f1 = f1_score(Y_test, Y_prediction, average='weighted')
print("F1-score:", f1)
recall = recall_score(Y_test, Y_prediction, average='weighted')
print("Recall:", recall)
conf_matrix = confusion_matrix(Y_test, Y_prediction)
print("Confusion Matrix:")
print(conf_matrix)

#decoding the output into human understandable language
Y_prediction_decoded = label_encoder.inverse_transform(Y_prediction)
print (Y_prediction_decoded)
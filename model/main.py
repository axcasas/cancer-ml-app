import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle5

# function to clean data 
def clean_data():
    data = pd.read_csv('data/data.csv') # load data
    data = data.drop(['Unnamed: 32','id'], axis=1) # remove unnecesary columns 
    data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0}) # convert object data to numerical data
    return data # returns cleaned data

# function for the model 
def create_model(data):
    # 1. Set X, y
    X = data.drop(['diagnosis'], axis=1) # our predictors are all the columns except the diagnosis
    y = data['diagnosis']

    # 2. Scale the data 
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 3. Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Train model 
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 5. Test model
    y_pred = model.predict(X_test)
    print('Accuracy: ', accuracy_score(y_test, y_pred))
    print('Classification report: ', accuracy_score(y_test, y_pred))

    return model, scaler

# main function
def main():

    data = clean_data()

    model, scaler = create_model(data)

    with open('model/model.pkl','wb') as f:
      pickle5.dump(model,f)
    
    with open('model/scaler.pkl','wb') as f:
      pickle5.dump(scaler, f)

if __name__ == '__main__':
    main()
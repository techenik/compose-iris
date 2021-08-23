import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

# define the class encodings and reverse encodings
classes = {1: "Kama", 2: "Rosa", 3: "Canadian"}
r_classes = {y: x for x, y in classes.items()}

# function to train and load the model during startup
def init_model():
    if not os.path.isfile("models/seeds_nb.pkl"):
        clf = GaussianNB()
        df = pd.read_csv('./data/seeds_dataset.txt', sep= '\t', header= None,
                names=['area','perimeter','compactness','lengthOfKernel','widthOfKernel','asymmetryCoefficient',
                      'lengthOfKernelGroove','seedType'])
        X = df.drop('seedType', axis = 1)
        y = df['seedType']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3)
        clf.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, clf.predict(X_test))
        print("Model trained with Accuracy: " , accuracy)
        pickle.dump(clf, open("models/seeds_nb.pkl", "wb"))


# function to train and save the model as part of the feedback loop
def train_model(data):
    # load the model
    clf = pickle.load(open("models/seeds_nb.pkl", "rb"))

    # pull out the relevant X and y from the FeedbackIn object
    X = [list(d.dict().values())[:-1] for d in data]
    y = [r_classes[d.seedType] for d in data]

    # fit the classifier again based on the new data obtained
    clf.fit(X, y)

    # save the model
    pickle.dump(clf, open("models/seeds_nb.pkl", "wb"))

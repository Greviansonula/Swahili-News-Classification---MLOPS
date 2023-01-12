import pickle
from sklearn.ensemble import RandomForestClassifier

with open('x_train.pkl', 'rb') as file:
    x_train = pickle.load(file)

with open('y_train.pkl', 'rb') as file:
    y_train = pickle.load(file)

classifier = RandomForestClassifier(n_estimators=100 ,criterion='entropy' , random_state=0).fit(x_train, y_train)
with open('model.pkl', 'wb') as file:
    pickle.dump(classifier)
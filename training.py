import pickle
from sklearn.ensemble import RandomForestClassifier

with open('data_splits.pkl', 'rb') as file:
    data_splits = pickle.load(file)

x_train, y_train = data_splits['x_train'], data_splits['y_train']

classifier = RandomForestClassifier(n_estimators=100 ,criterion='entropy' , random_state=0).fit(x_train, y_train)
with open('model.pkl', 'wb') as file:
    pickle.dump(classifier, file)
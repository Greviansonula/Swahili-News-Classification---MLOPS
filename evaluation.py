import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer, roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support as score


with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('x_test.pkl', 'rb') as file:
    x_test = pickle.load(file)
with open('y_test.pkl', 'rb') as file:
    y_test = pickle.load(file)

y_pred = model.predict(x_test)

# Performance metrics

accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)

# Get precision, recall, f1 scores

precision, recall, f1score, support = score(y_test, y_pred, average='micro')
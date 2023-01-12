import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer, roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support as score


with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('data_splits.pkl', 'rb') as file:
    data_splits = pickle.load(file)

x_test, y_test = data_splits['x_test'], data_splits['y_test']

y_pred = model.predict(x_test)

# Performance metrics

accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)

# Get precision, recall, f1 scores

precision, recall, f1score, support = score(y_test, y_pred, average='micro')
model_name = "Random Forest"
print(f'Test Accuracy Score of Basic {model_name}: % {accuracy}')

print(f'Precision : {precision}')
print(f'Recall : {recall}')

print(f'F1-score : {f1score}')
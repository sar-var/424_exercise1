import sklearn
from sklearn import metrics
from sklearn.naive_bayes import BernouliNB

BernNB = BernolliNB()
BernNB.fit(X_train, Y_train)
Y_pred = BernNB.predict(X_test)
print metrics.accuracy_score(Y_test, Y_pred)
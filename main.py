import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from sklearn.model_selection import train_test_split

#Generate linearly separable data (D1)
x1, y1 = make_blobs(n_samples=200, n_features=2, centers=2, random_state=77)

#Split into train and test sets
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.1, random_state=77)

#Generate linearly non-separable data (D2)
x2, y2 = make_circles(n_samples=200, factor=0.5, noise=0.1, random_state=77)

#Split into train and test sets
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.2, random_state=77)

#Plot D1
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(x1[:, 0], x1[:, 1], c=y1)
plt.title("Linearly Separable Data (D1)")

#Plot D2
plt.subplot(1, 2, 2)
plt.scatter(x2[:, 0], x2[:, 1], c=y2)
plt.title("Linearly Non-separable Data (D2)")
plt.show()

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#Hard-margin SVM for D1
svm_hard = SVC(kernel='linear', C=1e8)  # Use a very large C for hard margin. C is regularization parameter.
svm_hard.fit(x1_train, y1_train)

y1_pred = svm_hard.predict(x1_test)
accuracy_T1 = accuracy_score(y1_test, y1_pred)
print(f"Hard-margin SVM accuracy on T1: {accuracy_T1:.2f}")

def plot_decision_boundary(clf, X, y, plot_index):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.subplot(1, 2, plot_index)
    plt.contourf(xx, yy, Z, alpha=0.3)    
    plt.scatter(X[:, 0], X[:, 1], c=y)

plt.figure(figsize=(12, 6))

plot_decision_boundary(svm_hard, x1_train, y1_train, 1)

#Soft-margin SVM for D2
svm_soft = SVC(kernel='linear', C=1.0)  # Use a smaller C for soft margin. C is regularization parameter.
svm_soft.fit(x2_train, y2_train)

y2_pred = svm_soft.predict(x2_test)
accuracy_T2 = accuracy_score(y2_test, y2_pred)
print(f"Soft-margin SVM accuracy on T2: {accuracy_T2:.2f}")

plot_decision_boundary(svm_soft, x2_train, y2_train, 2)

plt.show()

from sklearn.neural_network import MLPClassifier

#Two-layer MLP for D1
mlp_D1 = MLPClassifier(hidden_layer_sizes=(10,), max_iter=5_000, random_state=11)
mlp_D1.fit(x1_train, y1_train)
y1_mlp_pred = mlp_D1.predict(x1_test)
accuracy_mlp_T1 = accuracy_score(y1_test, y1_mlp_pred)
print(f"MLP accuracy on T1: {accuracy_mlp_T1:.2f}")

#Two-layer MLP for D2
mlp_D2 = MLPClassifier(hidden_layer_sizes=(10,), max_iter=5_000, random_state=11)
mlp_D2.fit(x2_train, y2_train)
y2_mlp_pred = mlp_D2.predict(x2_test)
accuracy_mlp_T2 = accuracy_score(y2_test, y2_mlp_pred)

print(f"MLP accuracy on T2: {accuracy_mlp_T2:.2f}")

print("\nComparison of SVM and MLP Results:")
print(f"Hard-margin SVM accuracy on T1: {accuracy_T1:.2f}")
print(f"MLP accuracy on T1: {accuracy_mlp_T1:.2f}")
print(f"Soft-margin SVM accuracy on T2: {accuracy_T2:.2f}")
print(f"MLP accuracy on T2: {accuracy_mlp_T2:.2f}")

if accuracy_T1 > accuracy_mlp_T1:
    print("Hard-margin SVM performs better than MLP on linearly separable data (D1).")
else:
    print("MLP performs better than Hard-margin SVM on linearly separable data (D1).")

if accuracy_T2 > accuracy_mlp_T2:
    print("Soft-margin SVM performs better than MLP on linearly non-separable data (D2).")
else:
    print("MLP performs better than Soft-margin SVM on linearly non-separable data (D2).")

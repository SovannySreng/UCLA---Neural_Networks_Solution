
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

def evaluate_model(model, Xtest, ytest):
    ypred = model.predict(Xtest)
    cm = confusion_matrix(ytest, ypred)
    accuracy = accuracy_score(ytest, ypred)
    
    return cm, accuracy

def plot_loss_curve(model):
    loss_values = model.loss_curve_
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values, label='Loss', color='blue')
    plt.title('Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
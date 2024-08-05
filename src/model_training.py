
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier

def train_model(x, y):
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=123)
    
    scaler = MinMaxScaler()
    scaler.fit(xtrain)
    Xtrain = scaler.transform(xtrain)
    Xtest = scaler.transform(xtest)
    
    mlp = MLPClassifier(hidden_layer_sizes=(3), batch_size=50, max_iter=100, random_state=123)
    mlp.fit(Xtrain, ytrain)
    
    return mlp, Xtrain, Xtest, ytrain, ytest
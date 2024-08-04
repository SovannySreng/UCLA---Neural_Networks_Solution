from sklearn.neural_network import MLPClassifier

def train_model(x_train, y_train):
    model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000)
    model.fit(x_train, y_train)
    return model
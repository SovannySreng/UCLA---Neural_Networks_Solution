import warnings
warnings.filterwarnings("ignore")

from src.data_preprocessing import load_data, preprocess_data
from src.eda import perform_eda
from src.feature_engineering import create_features, split_data
from src.model_training import train_model
from src.evaluation import evaluate_model, plot_loss_curve

def main():
    data_path = 'H:/My Drive/BISI II/Data Science/Term Assignments/UCLA - Neural_Networks_Solution/data/Admission.csv'
    
    # Load and preprocess data
    data = load_data(data_path)
    data = preprocess_data(data)
    
    # Perform EDA
    perform_eda(data)
    
    # Feature Engineering
    data = create_features(data)
    x, y = split_data(data)
    
    # Train Model
    model, Xtrain, Xtest, ytrain, ytest = train_model(x, y)
    
    # Evaluate Model
    cm, accuracy = evaluate_model(model, Xtest, ytest)
    print('Confusion Matrix:')
    print(cm)
    print('Accuracy:', accuracy)
    
    # Plot Loss Curve
    plot_loss_curve(model)

if __name__ == "__main__":
    main()
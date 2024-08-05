
import seaborn as sns
import matplotlib.pyplot as plt

def perform_eda(data):
    plt.figure(figsize=(15, 8))
    sns.scatterplot(data=data, x='GRE_Score', y='TOEFL_Score', hue='Admit_Chance')
    plt.show()
import pandas as pd
from training import Training
from graph import Graph
from testing import Testing
from sklearn.model_selection import train_test_split


# conda install -c conda-forge scikit-learn pandas




if __name__ == '__main__':
    epocas_max=100
    tasa_minima=97
    nu=0.0001
    
    df = pd.read_csv('data/OR.csv')
    df = df.sample(frac=1).reset_index(drop=True)
    df.columns = ['x1', 'x2', 'yd']
    
    train, test = train_test_split(df, test_size=0.2)
    
    tr = Training(train, epocas_max, tasa_minima, nu)
    tr.train()
    print(tr)
    gr = Graph(df, train, tr.get_pesos())
    gr.view()
    
    ts = Testing(test, tr.get_pesos(), nu)
    print(ts)
    

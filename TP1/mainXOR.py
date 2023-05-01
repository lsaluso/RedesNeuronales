import pandas as pd
from training import Training
from graph import Graph
from testing import Testing
from sklearn.model_selection import train_test_split


# conda install -c conda-forge scikit-learn pandas




if __name__ == '__main__':
    epocas_max=100
    tasa_minima=99
    nu=0.0001
    

    df = pd.read_csv('data/XOR.csv')
    df = df.sample(frac=1).reset_index(drop=True)
    df.columns = ['x1','x2', 'yd']

    # Partición del df en 3 partes iguales para implementar
    # validación cruzada con 3 particiones
    p1, p2 = train_test_split(df, test_size=0.66)
    p2, p3 = train_test_split(p2, test_size=0.5)

    tasa_aciertos = 0    
    for i in range(1, 4):        
        if i ==1 :
            train = pd.concat([p2,p3])
            test = p1
        else:
            if i == 2:
                train = pd.concat([p1,p3])
                test = p2
            else:
                train = pd.concat([p1,p2])
                test = p3
   
        tr = Training(train, epocas_max, tasa_minima, nu)
        tr.train()
        
        ts = Testing(test, tr.get_pesos(), nu)
        print(ts)

        tasa_aciertos = tasa_aciertos + ts.get_tasa_aciertos()
        print("Iteración ", i)
        print(tr)
        gr = Graph(df, train, tr.get_pesos())
        gr.view()
    
    print("Tasa de Aciertos promedio=",str(tasa_aciertos/3))

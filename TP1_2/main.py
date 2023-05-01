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
    

    df = pd.read_csv('data/spheres1d10.csv')
    df = df.sample(frac=1).reset_index(drop=True)
    df.columns = ['x1','x2','x3', 'yd']

    # Partición del df en 5 partes iguales para implementar
    # validación cruzada con 5 particiones
    p1, p2 = train_test_split(df, test_size=0.8)
    p2, p3 = train_test_split(p2, test_size=0.75)
    p3, p4 = train_test_split(p3, test_size=0.666)
    p4, p5 = train_test_split(p4, test_size=0.5)
    
    
    tasa_aciertos = 0    
    for i in range(1, 6):
        if i ==1 :
            train = pd.concat([p2,p3,p4,p5])
            test = p1
        else:
            if i == 2:
                train = pd.concat([p1,p3,p4,p5])
                test = p2
            else:
                if i == 3:
                    train = pd.concat([p1,p2,p4,p5])
                    test = p3
                else:
                    if i == 4:
                        train = pd.concat([p1,p2,p3,p5])
                        test = p4
                    else:
                        train = pd.concat([p1,p2,p3,p4])
                        test = p5                   
   
        tr = Training(train, epocas_max, tasa_minima, nu)
        tr.train()
        print("Iteración ", i)
        print(tr)
        ts = Testing(test, tr.get_pesos(), nu)
        print(ts)
        tasa_aciertos = tasa_aciertos + ts.get_tasa_aciertos()
        #gr = Graph(df, train, tr.get_pesos())
        #gr.view()
    
    print("Tasa de Aciertos promedio=",str(tasa_aciertos/5))

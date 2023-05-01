import numpy as np

class Testing:
    
    def __init__(self, test, pesos, nu=0.01):

        aciertos = 0
        x0= -1
        num_patrones = test.count()[0]+1
        i= 0
        for index, row in test.iterrows():
            y = np.sign(pesos[i]['w0']*x0 +  pesos[i]['w1']*row['x1'] + pesos[i]['w2']*row['x2'])
            i = i + 1
    
            if (y==row['yd']):
                aciertos = aciertos + 1
        self._tasa_aciertos = (aciertos/num_patrones) * 100
        
    def get_tasa_aciertos(self):
        return self._tasa_aciertos

    def __str__(self):
        return "Tasa de aciertos calculada en Test: "+ str(self._tasa_aciertos)

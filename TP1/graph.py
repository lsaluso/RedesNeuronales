import matplotlib.pyplot as plt
import seaborn as sns



class Graph:
    
    def __init__(self, df, train, pesos):
        self._df = df
        self._train = train
        self._pesos = pesos
        sns.set()
        plt.xlim([-1.1, 1.1])
        plt.ylim([-1.1, 1.1])
        plt.scatter(self._df.x1, self._df.x2, s=0.01)
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')

    
    def view(self):
        #print (str(self._train.shape[0]), str(self._train.shape[0]/10))
        for i in range(0,self._train.shape[0],int(self._train.shape[0]/10)): #range(10):      
            # y1 = w0/w2 - w1/w2 * x1
            # y2 = w0/w2 - w1/w2 * x2
            x1 = -1 # self._train.iloc[0]['x1']
            x2 = 1 # self._train.iloc[1]['x1']
            y1 = self._pesos[i]['w0'] / self._pesos[i]['w2'] - self._pesos[i]['w1'] / self._pesos[i]['w2'] * x1
            y2 = self._pesos[i]['w0'] / self._pesos[i]['w2'] - self._pesos[i]['w1'] / self._pesos[i]['w2'] * x2
            x_values = [x1, x2]
            y_values = [y1, y2]
            alpha = round(i/self._train.shape[0],2)
            plt.plot(x_values, y_values, 'ro', linestyle="--", alpha=alpha)
            plt.text(x2+0.05, y2-0.01,  str(i), alpha=alpha)
            plt.text(x1-0.05, y1-0.01,  str(i), alpha=alpha)
        plt.show()
        
    
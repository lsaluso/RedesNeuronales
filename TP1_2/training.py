from perceptron import Perceptron


class Training:

    def __init__(self, train, epocas_max=100, tasa_minima=80, nu=0.1):
        self._epocas = 1
        self._epocas_max = epocas_max
        self._tasa_aciertos = 0
        self._tasa_minima = tasa_minima
        self._nu = nu
        self._num_patrones = train.count()[0] + 1
        self._pesos = {}
        self._perceptron = Perceptron(self._nu)
        self._train = train

    def train(self):
        while (self._epocas < self._epocas_max) and (self._tasa_aciertos <= self._tasa_minima):
            aciertos = 0
            i = 0
            for index, row in self._train.iterrows():
                self._pesos[i] = {"w0": self._perceptron.get_w0(),
                                  "w1": self._perceptron.get_w1(),
                                  "w2": self._perceptron.get_w2()}
                i = i + 1
                self._perceptron.calculate(row)
                if self._perceptron.get_y() == row['yd']:
                    aciertos = aciertos + 1

            self._tasa_aciertos = (aciertos / self._num_patrones) * 100
            self._epocas = self._epocas + 1

    def get_pesos(self):
        return self._pesos
        
    def __str__(self):
        cad =  "w0=" + str(self._perceptron.get_w0()) + "\n"
        cad += "w1=" + str(self._perceptron.get_w1()) + "\n"
        cad += "w2=" + str(self._perceptron.get_w2()) + "\n"
        cad += "Tasa de acierto minima: " + str(self._tasa_minima) + "\n"
        cad += "Tasa de aciertos calculada:" + str(self._tasa_aciertos) + "\n"
        cad += "Epocas máximas:" + str(self._epocas_max) + "\n"
        cad += "Epocas calculadas:" + str(self._epocas) + "\n"
    
        return cad

    def get_tasa_aciertos(self):
        return self._tasa_aciertos
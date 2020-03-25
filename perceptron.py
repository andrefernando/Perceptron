import numpy as np
import csv as csv
import matplotlib.pyplot as pyplot
import matplotlib.patches as mpatches
from random import randrange
from sklearn.model_selection import KFold

 
class Perceptron(object):

    def __init__(self, input_size, lr=0.1, epochs=500,bias=1):
        self.W = np.zeros(input_size+1)  # um campo é adicionado para o bias
        self.epochs = epochs # número de épocas de treinamento
        self.lr = lr # taxa de aprendizado
        self.predictions = [] #lista para armazenar as predições para a função de acurácia
        self.values = [] #lista para armazenar as classes para a função de acurácia
        self.bias = bias # bias
        
    # Normalizando os dados
    def normalize(self,dataset):
        minmax = list()
        minmax = [[min(column), max(column)] for column in zip(*dataset)]
        for row in dataset:
            for i in range(len(row)-1):
                row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

    def activation_fn(self, x): # função de ativação
        return 1 if x >= 0 else 0
 
    def predict(self, x): # função de predição
        z = self.W.T.dot(x) #  dot realiza multiplicação, nesse caso de arrays
        activation = self.activation_fn(z)
        return activation
 
    def fit(self, x_data, y_data): # função de treinamento
        
        for epoch in range(self.epochs): # processando uma geração

            sum_error = 0

            for i in range(x_data.shape[0]): # percorrendo todos os dados de uma das linhas
                x = np.insert(x_data[i], 0, self.bias) # inserindo o valor de bias
                y = self.predict(x) # realizando a predição
                error = y_data[i] - y # calculando se tivemos erro
                
                if error != 0: # caso a predição tenha falhado
                    sum_error+=1
                    self.W = self.W + self.lr * error * x # atualiza-se os pesos
            
            print("Época: {}".format(epoch+1) + " / Sum Erro: {}".format(sum_error))
            
            if sum_error == 0: # se não econtramos mais erros, paramos
                break

    def test(self,x_data,y_data):
        for i in range(x_data.shape[0]):
            x = np.insert(x_data[i], 0, self.bias) #Inserindo o valor de bias
            activation = self.predict(x)
            self.values.append(y_data[i])
            self.predictions.append(activation)

        correct = 0
        for i in range(len(self.predictions)):
            if self.predictions[i] == self.values[i]:
                correct += 1
        accuracy = int(correct / float(len(self.predictions)) * 100.0)
        print("Acurácia: {}%".format(accuracy))

#Fim da classe

# Utilizando a classe Perceptron
def open_file(path):
    with open(path) as dataset:
        data = np.array(list(csv.reader(dataset)))
        x_data = np.zeros((len(data)-1,2))
        y_data = np.empty(len(data)-1)

        for x in range(0,len(data)-1):
            x_data[x][0] = data[x][2]
            x_data[x][1] = data[x][3]
            y_data[x] = data[x][-1]
            
    return x_data,y_data

x_data,y_data = open_file("iris.csv")

perceptron = Perceptron(len(x_data[0]))
perceptron.normalize(x_data)
perceptron.fit(x_data,y_data)
print(perceptron.W)

# Realizando os testes de acurácia utilizando K-Fold
kf = KFold(n_splits=10,shuffle=True,random_state=12)
for train_indices, test_indices in kf.split(x_data,y_data):
    perceptron.test(x_data[test_indices],y_data[test_indices])

#Plotando o dados

for x in range(0,len(x_data)-1):
    if (y_data[x] == 0):    
        pyplot.plot(x_data[x][0],x_data[x][1],'bo',color='green',label='Setosa' if x == 0 else "")
    else:
        pyplot.plot(x_data[x][0],x_data[x][1],'bo',color='red',label='Virginica' if x == 90 else "")

pyplot.xlabel('Petal Length')
pyplot.ylabel('Petal Width')
pyplot.legend(loc='best')

# Calculando o slope e o intercept com os dados do treinamento
for i in np.linspace(np.amin(x_data[:,:1]),np.amax(x_data[:,:1])):
    slope = -(perceptron.W[0]/perceptron.W[2])/(perceptron.W[0]/perceptron.W[1])  
    intercept = -perceptron.W[0]/perceptron.W[2]

    # Desenhando a fronteira de decisão
    y = (slope*i) + intercept
    pyplot.plot(i, y,'ko')

pyplot.show()



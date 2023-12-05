import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from time import time
# load the dataset, split into input (X) and output (y) variables
dataset = np.loadtxt('h:/Documents/Romaric/CY Tech/ING3/Semestre 5/HHPC Tools/Rendu2/CNNaPralleliser/pima-indians-diabetes.data.csv', delimiter=',')
X = dataset[:,0:8]
y = dataset[:,8]

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# define the model
class PimaClassifier(nn.Module):
    def __init__(self):
        super(PimaClassifier, self).__init__()

        # Couche d'entrée
        self.input_layer = nn.Linear(8, 12)
        self.input_activation = nn.ReLU()

        # Première couche cachée
        self.hidden1 = nn.Linear(12, 16)
        self.activation1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)

        # Deuxième couche cachée
        self.hidden2 = nn.Linear(16, 8)
        self.activation2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)

        # Couche de sortie
        self.output_layer = nn.Linear(8, 1)
        self.output_activation = nn.Sigmoid()

        # Initialisation des poids
        nn.init.xavier_normal_(self.input_layer.weight)
        nn.init.xavier_normal_(self.hidden1.weight)
        nn.init.xavier_normal_(self.hidden2.weight)
        nn.init.xavier_normal_(self.output_layer.weight)

    def forward(self, x):
        x = self.input_activation(self.input_layer(x))

        x = self.activation1(self.hidden1(x))
        x = self.dropout1(x)

        x = self.activation2(self.hidden2(x))
        x = self.dropout2(x)

        x = self.output_activation(self.output_layer(x))

        return x

model = PimaClassifier()
print(model)

# train the model
temps_init = time()
loss_fn   = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 100
batch_size = 2

for epoch in range(n_epochs):
    for i in range(0, len(X), batch_size):
        Xbatch = X[i:i+batch_size]
        y_pred = model(Xbatch)
        ybatch = y[i:i+batch_size]
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
#    print(f'finished{epoch}, latest loss {loss}')
temp_fin = time()
# compute accuracy
y_pred = model(X)
accuracy = (y_pred.round() == y).float().mean()
print(f"Accuracy {accuracy}")
print("temps training =", temp_fin - temps_init)
# make class predictions with the model
predictions = (model(X) > 0.5).int()
for i in range(5):
    print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))
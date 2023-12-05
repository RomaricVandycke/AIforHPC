import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from time import time

# Function to initiate distributed training
def setup(rank, world_size):
    # Set up the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

# Function to cleanup processes
def cleanup():
    dist.destroy_process_group()

# Load the dataset, split into input (X) and output (y) variables
dataset = np.loadtxt('h:/Documents/Romaric/CY Tech/ING3/Semestre 5/HHPC Tools/Rendu2/CNNaPralleliser/pima-indians-diabetes.data.csv', delimiter=',')
X = dataset[:, 0:8]
y = dataset[:, 8]

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# Define the model
class PimaClassifier(nn.Module):
    def __init__(self):
        super(PimaClassifier, self).__init__()

        # Input layer
        self.input_layer = nn.Linear(8, 12)
        self.input_activation = nn.ReLU()

        # First hidden layer
        self.hidden1 = nn.Linear(12, 16)
        self.activation1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)

        # Second hidden layer
        self.hidden2 = nn.Linear(16, 8)
        self.activation2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)

        # Output layer
        self.output_layer = nn.Linear(8, 1)
        self.output_activation = nn.Sigmoid()

        # Weight initialization
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

# Training function
def train_with_args(rank, world_size, args):
    # Setup process group
    setup(rank, world_size)

    # Set device for this process
    torch.cuda.set_device(rank)

    # Create model, wrap it in DistributedDataParallel
    model = PimaClassifier().cuda(rank)
    ddp_model = DDP(model, device_ids=[rank])

    print(ddp_model)

    # Train the model
    loss_fn = nn.BCELoss()  # Binary cross entropy
    optimizer = optim.Adam(ddp_model.parameters(), lr=args.lr)

    n_epochs = args.epochs
    batch_size = args.batch_size

    for epoch in range(n_epochs):
        for i in range(0, len(X), batch_size):
            Xbatch = X[i:i + batch_size].cuda(rank)
            y_pred = ddp_model(Xbatch)
            ybatch = y[i:i + batch_size].cuda(rank)
            loss = loss_fn(y_pred, ybatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            dist.barrier()  # synchronize all processes after each batch

        # Uncomment below to print loss per epoch
        # print(f'Finished epoch {epoch}, latest loss {loss.item()}')

    cleanup()

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Pima Indians Diabetes Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    world_size = 2  # Vous pouvez définir le nombre de processus ici
    mp.spawn(train_with_args, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import SAGEConv,GATConv,GATv2Conv,CuGraphSAGEConv,GCNConv
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import MoleculeNet

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
dataset = MoleculeNet(root='.', name='ESOL')
data_size = len(dataset)
BATCH = 64
hidden_channels = 64
dropout = 0.5
num_heads = 4
num_layers = 2

# define the GCN model
class ESOLNet(torch.nn.Module):
    def __init__(self):
        # Initialization
        super().__init__()
        torch.manual_seed(42)
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # self.conv3 = GATConv(hidden_channels, hidden_channels)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        # depends on cat pooling number
        self.out = Linear(hidden_channels, 1)

    def forward(self, x , edge_index, batch_index):
        # 1st Conv layer
        hidden = self.conv1(x, edge_index)
        hidden = torch.relu(hidden)
        hidden = self.dropout_layer(hidden)
        # Other layers
        hidden = self.conv2(hidden, edge_index)
        hidden = torch.relu(hidden)
        hidden = self.dropout_layer(hidden)
        # hidden = self.conv3(hidden, edge_index)
        # hidden = torch.relu(hidden)
        # hidden = self.dropout_layer(hidden)
        # Global pooling
        hidden = gmp(hidden, batch_index)
        # hidden = torch.cat((gmp(hidden, batch_index), gap(hidden, batch_index)), dim=1)
        # hidden = torch.cat((gmp(hidden, batch_index), gap(hidden, batch_index), gap(hidden, batch_index)), dim=1)
        out = self.out(hidden)
        return out

    def train_model(self, loss_fn, optimizer, loader, num_epochs):
        self.train()
        device = next(self.parameters()).device
        losses = []
        for epoch in range (num_epochs):
            total_loss = 0
            for data in loader:
                data = data.to(device)
                optimizer.zero_grad()
                pred = self(data.x.float(), data.edge_index, data.batch)
                loss = loss_fn(pred, data.y.view(-1, 1).float())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            epoch_loss = total_loss / len(loader)
            losses.append(epoch_loss)
            if epoch % 50 == 0:
                print(f"Epoch {epoch}, Loss: {epoch_loss}")
        return losses

model = ESOLNet()
print(model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.0005,weight_decay=5e-4)

loader = DataLoader(dataset[:int(data_size * 0.8)],
                    batch_size=BATCH, shuffle=True)
test_loader = DataLoader(dataset[int(data_size * 0.8):],
                         batch_size=BATCH, shuffle=True)

losses = model.train_model(loss_fn, optimizer, loader, 1000)

# Visualize learning (training loss)
losses_float = [float(loss) for loss in losses]
loss_indices = [i for i, _ in enumerate(losses_float)]
sns.lineplot(x=loss_indices, y=losses_float)
plt.show()

test_batch = next(iter(test_loader))
with torch.no_grad():
    test_batch.to(device)
    pred = model(test_batch.x.float(), test_batch.edge_index, test_batch.batch)
    df = pd.DataFrame()
    df["y_real"] = test_batch.y.tolist()
    df["y_pred"] = pred.tolist()
df["y_real"] = df["y_real"].apply(lambda row: row[0])
df["y_pred"] = df["y_pred"].apply(lambda row: row[0])
print(df)
df.to_csv('predication.csv', index=False)

# Visualize predication (real)
COMPARE = sns.scatterplot(data=df, x="y_real", y="y_pred", color='blue', label='Comparison')
COMPARE.set(xlim=(-7, 2))
COMPARE.set(ylim=(-7, 2))
plt.show()
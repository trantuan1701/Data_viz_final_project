import torch
import torch.nn as nn

class EntityEmbeddingModel(nn.Module):
    def __init__(self, emb_dims, n_cont, dropout_rate=0.2):
        super(EntityEmbeddingModel, self).__init__()
        
        # Embedding Layers
        self.embeddings = nn.ModuleList([nn.Embedding(n, d) for n, d in emb_dims])
        
        # Calculate total dimension after concatenation
        n_emb_out = sum([d for n, d in emb_dims])
        self.n_emb_out = n_emb_out
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(n_emb_out + n_cont, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(512, 1) # Predict Log Sales
        
        self.relu = nn.ReLU()
        
    def forward(self, x_cat, x_cont):
        # Process Embeddings
        x = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(x, 1)
        
        # Concatenate with Continuous vars
        x = torch.cat([x, x_cont], 1)
        
        # Dense Layers
        x = self.dropout1(self.bn1(self.relu(self.fc1(x))))
        x = self.dropout2(self.bn2(self.relu(self.fc2(x))))
        x = self.fc3(x)
        
        return x
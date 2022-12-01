import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import average_precision_score

def run(model, data_list, train_split, criterion, optimizer, epoch, batch_size):
    """
    Run the experiment.
    - parameters:
        - model: The PyG GNN model object.
        
        - data_list: The list of data objects to train and test. 
        
        - train_split: A float representing the split of train dataset.
        
        - criterion: Loss function to use. 
        
        - optimizer: Pytorch Optimizer to use. 
        
        - epoch[int]: The number of epoch to train. 
        
        - batch_size: The size of each batch. 
    """
    
    print("Split data list to train and val datasets")
    train_set, val_set = torch.utils.data.random_split(data_list, [int(len(data_list)*train_split), len(data_list) - int(len(data_list)*train_split)])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    print(f"""Training settings:
    Dataset size: {len(data_list)} 
    Train set size: {len(train_set)}  
    Val set size: {len(val_set)} 
    Epoch setting: {epoch} 
    Batch size: {batch_size} """)
    
    print("<<<Training Started!!>>>")
    
    def train():
        model.train()

        for data in train_loader:
            # Iterate in batches over the training dataset.
            out = model(data.x.float(), data.edge_index, data.batch)  # Perform a single forward pass.
            loss = criterion(out, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.

    def test(loader):
        model.eval()
        loss_total = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            out = model(data.x.float(), data.edge_index, data.batch)  
            loss = average_precision_score(data.y.numpy()[0], out.detach().numpy()[0])
            loss_total += loss
        return loss_total / len(loader.dataset)  # Derive ratio of correct predictions.
    
    
    for epoch in range(1, epoch):
        train()
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        print(f'Epoch: {epoch:03d}, Train AP: {train_acc:.4f}, Val AP: {test_acc:.4f}\n')







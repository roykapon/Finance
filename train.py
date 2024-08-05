import torch
import torch.nn as nn
from load_data import StockDataset, stock_collate
from torch.utils.data import DataLoader
from model import TransformerModel
import torch.optim as optim


def train_model(model: TransformerModel, dataloader: DataLoader, num_epochs=10, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets, dates in dataloader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            
            print(f"Loss: {loss.item()}")

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs[0].size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    print("Training complete")


if __name__ == "__main__":
    data_dir = "./data"
    company_name = "AAPL"  # Replace with actual company name
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001

    # Create dataset and dataloader
    dataset = StockDataset(f"{data_dir}\\{company_name}")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=stock_collate)
    
    # Create model
    model = TransformerModel(dataset.input_sizes)

    # Train model
    train_model(model, dataloader, num_epochs=num_epochs, learning_rate=learning_rate)

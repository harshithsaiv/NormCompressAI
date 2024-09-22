import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import matplotlib.pyplot as plt

# Define the autoencoder architecture by inheriting from the nn.Module 
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()  # Calling the parent class constructor to initialize the module correctly
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

#loss infinity function that is calculated using the formula max | x - y | 
def l_infinity_loss(output, target):
    return torch.max(torch.abs(output - target))

if __name__ == "__main__":
    # Parse command line arguments for selecting norm loss
    parser = argparse.ArgumentParser(description="Train autoencoder with specified norm loss")
    parser.add_argument('--norm', type=str, choices=['L2', 'L1', 'Linf'], default='L2', help='Select norm type (L2, L1, or Linf)')
    args = parser.parse_args()

    # Load and preprocess the CIFAR-10 dataset
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    # Initialize the model
    model = Autoencoder()
    
    # Choose loss function based on user input
    if args.norm == 'L2':
        criterion = nn.MSELoss()  # L2 norm (Mean Squared Error)
        print("Using L2 Norm Loss (MSE)")
    elif args.norm == 'L1':
        criterion = nn.L1Loss()  # L1 norm (Mean Absolute Error)
        print("Using L1 Norm Loss (MAE)")
    else:
        criterion = l_infinity_loss  # L-infinity norm (Maximum Absolute Error)
        print("Using L-infinity Norm Loss")
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters())

    # Training loop
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_values = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for data in trainloader:
            img, _ = data
            img = img.to(device)

            # Forward pass
            output = model(img)
            loss = criterion(output, img)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(trainloader)
        loss_values.append(avg_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    print("Training finished!")

    # Plot the loss over epochs
    plt.plot(range(1, num_epochs+1), loss_values, label=f'{args.norm} Norm Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss using {args.norm} Norm')
    plt.legend()
    plt.show()

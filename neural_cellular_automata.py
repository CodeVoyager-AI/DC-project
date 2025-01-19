import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load Image and Preprocess
img = Image.open('your_image.png').convert('RGB')
img = np.array(img) / 255.0  # Normalize
img = torch.tensor(img).float().unsqueeze(0)  # Add batch dim

# Define Neural Network Architecture
class NeuralCA(nn.Module):
    def __init__(self):
        super(NeuralCA, self).__init__()
        
        # First convolution: 3 channels to 16 channels
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        
        # Sobel filters + Identity filter
        self.sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).float().unsqueeze(0).unsqueeze(0)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0)
        self.identity = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]]).float().unsqueeze(0).unsqueeze(0)

        self.conv2 = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        
        # Perception vector and dense layers
        self.fc1 = nn.Linear(3 * img.shape[2] * img.shape[3], 128)  # Flattened image
        self.fc2 = nn.Linear(128, 16)

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # Conv1
        # Apply Sobel filters and identity filter
        sobel_x_out = torch.conv2d(x, self.sobel_x)
        sobel_y_out = torch.conv2d(x, self.sobel_y)
        identity_out = torch.conv2d(x, self.identity)
        
        # Combine outputs
        x = sobel_x_out + sobel_y_out + identity_out
        
        x = x.view(x.size(0), -1)  # Flatten for dense layers
        x = torch.relu(self.fc1(x))  # FC1
        x = self.fc2(x)  # FC2
        return x

# Initialize Model, Optimizer
model = NeuralCA()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Simulate growth
def simulate_growth(model, img, steps=100):
    for step in range(steps):
        optimizer.zero_grad()
        output = model(img)
        loss = loss_fn(output, img)
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            plt.imshow(output.squeeze().detach().numpy().transpose(1, 2, 0))  # Show image
            plt.title(f"Step {step}")
            plt.show()

# Run simulation
simulate_growth(model, img)

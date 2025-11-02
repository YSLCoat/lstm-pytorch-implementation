import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from model import SingleLayerLSTM


INPUT_DIM = 1
HIDDEM_DIM = 16
OUTPUT_DIM = 1
LEARNING_RATE = 1e-3

SEQUENCE_LENGTH = 50
NUM_SEQUENCES = 100

EPOCHS = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


total_points = SEQUENCE_LENGTH + NUM_SEQUENCES
data = np.sin(np.linspace(0, 10 * np.pi, total_points))

X = []
y = []

for i in range(NUM_SEQUENCES):
    X.append(data[i : i + SEQUENCE_LENGTH])
    y.append(data[i + SEQUENCE_LENGTH])


X_tensor = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1).to(device)
y_tensor = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1).to(device)


model = SingleLayerLSTM(INPUT_DIM, HIDDEM_DIM, OUTPUT_DIM).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


true_values = data[SEQUENCE_LENGTH:]
plt.figure(figsize=(14, 6))
plt.ion()
plt.show()


for epoch in range(EPOCHS):
    model.train()
    
    predictions = model(X_tensor)
    
    loss = criterion(predictions, y_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}')
        
        model.eval()
        with torch.no_grad():
            test_predictions = model(X_tensor).cpu().numpy()
        model.train()

        plt.clf()
        plt.plot(true_values, label='Ground Truth (sin(x))', linestyle='--')
        plt.plot(test_predictions.squeeze(), label=f'Model Predictions (Epoch {epoch+1})')
        plt.title('Sine Wave Estimation') 
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        plt.pause(0.1)

plt.ioff()
plt.title('Sine Wave Estimation (Final Result)')
plt.show()
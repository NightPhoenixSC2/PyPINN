import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PINN().to(device)
model.load_state_dict(torch.load("pinn_model_task_1_1.pth", map_location=device))  # Load the trained model
model.eval()

def predict_single(x_value):
    x_tensor = torch.tensor([x_value], dtype=torch.float32, device=device).view(-1, 1)
    with torch.no_grad():
        return model(x_tensor).cpu().numpy()[0, 0]

x_new = np.linspace(0, 1, 50)
x_new_sorted_indices = np.argsort(x_new)
x_new = x_new[x_new_sorted_indices]

x_new_t = torch.tensor(x_new, dtype=torch.float32, device=device).view(-1, 1)

with torch.no_grad():
    u_pred_new = model(x_new_t).cpu().numpy()
    u_pred_new = u_pred_new[x_new_sorted_indices]

plt.figure(figsize=(8, 5))
plt.plot(x_new, u_pred_new, label="Predicted u(x)", color='red', linewidth=2)
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("Predicted Values using PINN Model")
plt.legend()
plt.grid()
plt.show()
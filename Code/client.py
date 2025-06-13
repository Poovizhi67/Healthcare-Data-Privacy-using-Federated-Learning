import sys
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tenseal as ts
import flwr as fl
import pickle
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from opacus import PrivacyEngine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import time
# from mpl_toolkits.mplot3d import Axes3D
# ✅ 1️⃣ Initialize CKKS Encryption Context


# ✅ Load CKKS Context from Server
def load_ckks_context():
    if not os.path.exists("ckks_context.tenseal"):
        print("❌ CKKS context file not found! Ensure the server has shared it.")
        sys.exit(1)
    with open("ckks_context.tenseal", "rb") as f:
        context = ts.context_from(f.read())
    return context
# ✅ Load Encryption Context at Client
client_context = load_ckks_context()
print("🔐 CKKS Context Loaded at Client!")

print("🔒 CKKS Homomorphic Encryption Context Initialized!")

# ✅ 2️⃣ Find available client datasets
train_feature_files = sorted(glob.glob("client_*_train_features.npy"))
client_ids = [int(f.split("_")[1]) for f in train_feature_files]

if len(sys.argv) != 2:
    print(f"Usage: python client.py <client_index>\nAvailable clients: {client_ids}")
    sys.exit(1)

client_idx = int(sys.argv[1])

if client_idx not in client_ids:
    print(f"Invalid client index: {client_idx}. Available clients: {client_ids}")
    sys.exit(1)

# ✅ 3️⃣ Load client's dataset
train_features = np.load(f"client_{client_idx}_train_features.npy")
train_labels = np.load(f"client_{client_idx}_train_labels.npy")
test_features = np.load(f"client_{client_idx}_test_features.npy")
test_labels = np.load(f"client_{client_idx}_test_labels.npy")


# Define a list of possible noise values
sigma_values = [0.1,0.5,1.0,2.0]

# Assign noise level based on client index
selected_sigma = sigma_values[client_idx % len(sigma_values)]

# Apply noise to features
train_noise = np.random.normal(loc=0, scale=selected_sigma, size=train_features.shape)
test_noise = np.random.normal(loc=0, scale=selected_sigma, size=test_features.shape)

train_features += train_noise
test_features += test_noise

print(f"📢 Client {client_idx} - Selected Noise Level (σ): {selected_sigma}")
# print(f"🔹 Train Noise Mean: {train_noise.mean():.4f}, Std Dev: {train_noise.std():.4f}")
# print(f"🔹 Test Noise Mean: {test_noise.mean():.4f}, Std Dev: {test_noise.std():.4f}")

# 🔹 Feature Scaling (Normalization) to Improve Convergence
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

# 🔹 Feature Augmentation using SMOTE (Only on training data)
smote = SMOTE(random_state=42)
train_features, train_labels = smote.fit_resample(train_features, train_labels)

print("✅ Feature Scaling & Augmentation Applied!")
print("Train Features Shape After Augmentation:", train_features.shape)

# ✅ 4️⃣ Apply PCA for Dimensionality Reduction
pca = PCA(n_components=50)
train_features = pca.fit_transform(train_features)
test_features = pca.transform(test_features)

# ✅ 5️⃣ Define MLP Model
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# ✅ 6️⃣ Secure Weight Encryption
def encrypt_weights(weights, context):
    """
    Encrypts model weights using CKKS and serializes them.
    """
    encrypted_weights = []
    for w in weights:
        # Convert weights to float64 to avoid precision errors
        w = w.astype(np.float64).flatten().tolist()
        encrypted_weights.append(ts.ckks_vector(context, w).serialize())  
    return encrypted_weights  # List of serialized encrypted vectors (byte strings)


def decrypt_weights(encrypted_weights, context):
    """
    Decrypts received encrypted weights.
    """
    decrypted_weights = []
   
    if isinstance(encrypted_weights, list):
        print(f"🔎 Number of Parameters: {len(encrypted_weights)}")
        # print(f"🔎 Sample Parameter Type: {type(encrypted_weights[0])}")

    for ew in encrypted_weights:
        try:
            # Ensure ew is bytes before deserializing
            if not isinstance(ew, bytes):
                continue  # Skip invalid types

            # Deserialize and decrypt
            enc_vector = ts.ckks_vector_from(context, ew)
            decrypted_weights.append(torch.tensor(enc_vector.decrypt(), dtype=torch.float32))
        except Exception as e:
            return None  # Return None to prevent further errors

    if not decrypted_weights:
        return None

    return decrypted_weights



# ✅ 7️⃣ Federated Learning Client with Gradient Encryption
class FLClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_features, test_labels, context):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_features = test_features
        self.test_labels = test_labels
        self.context = context
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.002)

        self.noise_levels = []
        self.accuracies = []
        self.privacy_budgets = []
        self.training_times = []

        # ✅ Apply Differential Privacy (DP)
        self.privacy_engine = PrivacyEngine(secure_mode=False)
        self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            target_epsilon=5.0,
            target_delta=1e-5,
            epochs=5,
            max_grad_norm=2.0
        )

    def get_parameters(self, config):
        print(f"🔐 Client {client_idx} encrypting model parameters...")
        weights = [val.cpu().detach().numpy() for val in self.model.parameters()]
        return encrypt_weights(weights, self.context)  # Now it's a list of serialized encrypted vectors

    def set_parameters(self, parameters):
        decrypted_params = decrypt_weights(parameters, self.context)

        if decrypted_params is None:
            return  # Prevents the crash

        for param, new_param in zip(self.model.parameters(), decrypted_params):
            param.data = new_param.view(param.shape).to(param.device)


    # ✅ Modify fit function to Encrypt and Decrypt Gradients
    def fit(self, parameters, config):
        print(f"🚀 Client {client_idx} received encrypted parameters, starting training...")

        self.set_parameters(parameters)  # Decrypt and load weights
        self.model.train()
        total_loss = 0
        correct = 0
        total_samples = 0
        round_loss = 0.0  # ✅ Initialize round_loss before using it
        start_time = time.time()  
        for epoch in range(30):
            running_loss = 0.0
            for batch_features, batch_labels in self.train_loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = self.criterion(outputs, batch_labels)

                loss.backward()

                # ✅ DO NOT overwrite param.grad.data with encrypted gradients
                # ✅ Instead, store them in a separate list if needed
                encrypted_grads = []
                for param in self.model.parameters():
                    if param.grad is not None:
                        flattened_grad = param.grad.data.cpu().numpy().flatten()
                        encrypted_grads.append(ts.ckks_vector(self.context, flattened_grad).serialize())

                self.optimizer.step()  # Update model normally

                running_loss += loss.item()
                pred = outputs.argmax(dim=1, keepdim=True)
                correct += pred.eq(batch_labels.view_as(pred)).sum().item()
                total_samples += batch_labels.size(0)
            end_time = time.time()
            epoch_loss = running_loss / len(self.train_loader)
            round_loss += epoch_loss    
            avg_loss = running_loss / len(self.train_loader)
            accuracy = 100.0 * correct / total_samples
            print(f"📉 Epoch {epoch+1}: Loss = {avg_loss:.4f}")

        print(f"✅ Client {client_idx} training completed, encrypting updated model weights...")

        self.noise_levels.append(selected_sigma)
        self.accuracies.append(accuracy)
        training_time = end_time - start_time
        self.training_times.append(training_time)  # Store computation time
       
        # ✅ Encrypt updated weights
        new_weights = [val.cpu().detach().numpy() for val in self.model.parameters()]
        return encrypt_weights(new_weights, self.context), len(self.train_loader.dataset), {"loss":avg_loss,"accuracy":accuracy}


    def evaluate(self, parameters, config):
        print(f"🔎 Client {client_idx} evaluating model...")
        self.set_parameters(parameters)
        self.model.eval()
        with torch.no_grad():
            test_features = torch.tensor(self.test_features, dtype=torch.float32).to(device)
            test_labels = torch.tensor(self.test_labels, dtype=torch.long).to(device)
            outputs = self.model(test_features)
            y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
            accuracy = accuracy_score(test_labels.cpu().numpy(), y_pred)
            loss = self.criterion(outputs, test_labels).item()

            print(f"✅ Client {client_idx} Accuracy: {accuracy:.4f}")
        return float(loss), len(test_labels), {"accuracy": accuracy}
    def plot_noise_vs_accuracy(self):
        plt.figure(figsize=(8, 6))
        plt.scatter(self.noise_levels, self.accuracies, color='b', label="Accuracy per Noise Level")
        plt.xlabel("Noise Level (σ)")
        plt.ylabel("Accuracy (%)")
        plt.title(f"Client {client_idx} - Noise vs. Accuracy")
        plt.legend()
        plt.grid(True)
        plt.show()
    def plot_epsilon_delta_accuracy(self):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')  # No need to import Axes3D separately

        epsilons = [5.0] * len(self.accuracies)  # Modify if you have varying values
        deltas = [1e-5] * len(self.accuracies)  # Modify if needed

        ax.scatter(epsilons, deltas, self.accuracies, c='b', marker='o', label="Accuracy per Epsilon-Delta")

        ax.set_xlabel("Epsilon")
        ax.set_ylabel("Delta")
        ax.set_zlabel("Accuracy (%)")
        ax.set_title(f"Client {client_idx} - Epsilon vs Delta vs Accuracy")
        ax.legend()

        plt.show()

    def plot_computational_overhead(self):
        rounds = list(range(1, len(self.training_times) + 1))
        plt.figure(figsize=(8, 5))
        plt.plot(rounds, self.training_times, 'r-o', label="With HE")
        plt.xlabel("Number of Training Rounds")
        plt.ylabel("Computation Time (seconds)")
        plt.title("Computational Overhead of Homomorphic Encryption")
        plt.legend()
        plt.grid(True)
        # plt.savefig("computational_overhead_HE.png")
        plt.show()
       
# ✅ 8️⃣ Run the Federated Client
if __name__ == "__main__":
    train_loader = DataLoader(TensorDataset(
        torch.tensor(train_features, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.long)
    ), batch_size=32, shuffle=True)

    model = MLPClassifier(input_dim=50, num_classes=len(set(train_labels)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🚀 Starting Client {client_idx}")
    client = FLClient(model, train_loader, test_features, test_labels, client_context)

    fl.client.start_numpy_client(server_address="127.0.0.1:9090", client=client)
    client.plot_noise_vs_accuracy()
    client.plot_epsilon_delta_accuracy()
    client.plot_computational_overhead()

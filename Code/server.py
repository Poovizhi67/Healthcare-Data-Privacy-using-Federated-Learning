import flwr as fl
import tenseal as ts
import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np

# ✅ 1️⃣ Initialize CKKS Encryption Context at the Server
def create_ckks_context():
    context = ts.context(
        scheme=ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=16384,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2**40
    context.generate_galois_keys()
    return context

# ✅ Generate CKKS Context at Server
server_context = create_ckks_context()

# ✅ Serialize CKKS Context and Save (to share with clients)
with open("ckks_context.tenseal", "wb") as f:
    f.write(server_context.serialize(save_secret_key=False))  

print("🔒 CKKS Encryption Context Initialized at Server!")

# ✅ 3️⃣ Store Loss & Accuracy for Visualization
loss_per_round = []
accuracy_per_round = []
train_loss_per_round = []
train_accuracy_per_round = []
test_loss_per_round = []
test_accuracy_per_round = []

# ✅ 4️⃣ Secure Aggregation Function
def aggregate_fit(rnd, results, failures):
    global server_context  
    print(f"📡 Aggregating round {rnd}...")

    serialized_weights_list = [fit_res.parameters for _, fit_res in results]

    if not serialized_weights_list:
        return None, {}

    first_params = serialized_weights_list[0]

    if not isinstance(first_params, fl.common.Parameters):
        return None, {}

    weights = first_params.tensors  
    sample_weights = []
   
    for i, w in enumerate(weights):
        if isinstance(w, (list, np.ndarray)):  
            w = pickle.dumps(w)

        if not isinstance(w, bytes):
            continue

        try:
            enc_vector = ts.ckks_vector_from(server_context, w)
            decrypted_tensor = torch.tensor(enc_vector.decrypt(), dtype=torch.float32)
            sample_weights.append(decrypted_tensor)
        except Exception as e:
            continue  

    if not sample_weights:
        return None, {}

    aggregated_weights = [torch.mean(torch.stack(sample_weights), dim=0)]
    encrypted_weights = [ts.ckks_vector(server_context, w.tolist()).serialize() for w in aggregated_weights]

    return fl.common.ndarrays_to_parameters(encrypted_weights), {}

# ✅ 5️⃣ Aggregate Evaluation Results (Global Model Performance)
def aggregate_evaluate(rnd, results, failures):
    global loss_per_round, accuracy_per_round, train_loss_per_round, train_accuracy_per_round, test_loss_per_round, test_accuracy_per_round
    total_loss, total_accuracy, total_train_loss, total_train_accuracy, total_test_loss, total_test_accuracy, total_samples = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0

    print(f"\n📊 Evaluating Global Model Performance for Round {rnd}...")

    for client_idx, (_, eval_res) in enumerate(results):
        if eval_res.status.code == fl.common.Code.OK:
            loss = eval_res.loss
            accuracy = eval_res.metrics["accuracy"]
            num_samples = eval_res.num_examples

            total_loss += loss * num_samples
            total_accuracy += accuracy * num_samples
            total_samples += num_samples

            avg_train_loss = total_train_loss / total_samples
            avg_train_accuracy = total_train_accuracy / total_samples
            avg_test_loss = total_test_loss / total_samples
            avg_test_accuracy = total_test_accuracy / total_samples

            train_loss_per_round.append(avg_train_loss)
            train_accuracy_per_round.append(avg_train_accuracy)
            test_loss_per_round.append(avg_test_loss)
            test_accuracy_per_round.append(avg_test_accuracy)
            print(f"📌 Client {client_idx + 1}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")

    if total_samples > 0:
        avg_loss = total_loss / total_samples
        avg_accuracy = total_accuracy / total_samples

        loss_per_round.append(avg_loss)
        accuracy_per_round.append(avg_accuracy)

        avg_train_loss = total_train_loss / total_samples
        avg_train_accuracy = total_train_accuracy / total_samples
        avg_test_loss = total_test_loss / total_samples
        avg_test_accuracy = total_test_accuracy / total_samples

        train_loss_per_round.append(avg_train_loss)
        train_accuracy_per_round.append(avg_train_accuracy)
        test_loss_per_round.append(avg_test_loss)
        test_accuracy_per_round.append(avg_test_accuracy)
       
        print(f"\n📢 Global Model Performance in Round {rnd}:")
        print(f"✅ Average Loss: {avg_loss:.4f}")
        print(f"✅ Average Accuracy: {avg_accuracy:.4f}")
       
        return avg_loss, {"accuracy": avg_accuracy}
   
    return 0.0, {"accuracy": 0.0}

# ✅ 6️⃣ Custom Strategy for Secure FL with CKKS
class SecureFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        return aggregate_fit(rnd, results, failures)
   
    def aggregate_evaluate(self, rnd, results, failures):
        return aggregate_evaluate(rnd, results, failures)

# ✅ 7️⃣ Start the Secure Flower Server
num_rounds = 1
fl.server.start_server(
    server_address="127.0.0.1:9090",
    config=fl.server.ServerConfig(num_rounds=num_rounds),
    strategy=SecureFedAvg(
        fraction_fit=1.0,  
        on_fit_config_fn=lambda rnd: {"lr": 0.001},
        on_evaluate_config_fn=lambda rnd: {"val": True},
    )
)

# ✅ 8️⃣ Fix List Length Before Plotting
def trim_list(lst, length):
    return lst[:length] if len(lst) > length else lst

train_loss_per_round = trim_list(train_loss_per_round, num_rounds)
train_accuracy_per_round = trim_list(train_accuracy_per_round, num_rounds)
test_loss_per_round = trim_list(test_loss_per_round, num_rounds)
test_accuracy_per_round = trim_list(test_accuracy_per_round, num_rounds)

# ✅ 9️⃣ Plot Loss and Accuracy Per Round
plt.figure(figsize=(10, 5))

# 🔹 Loss Graph
plt.subplot(1, 2, 1)
plt.plot(range(1, len(loss_per_round) + 1), loss_per_round, marker='o', linestyle='-', color='red', label="Loss")
plt.xlabel("Round")
plt.ylabel("Loss")
plt.title("Federated Learning Loss per Round")
plt.legend()

# 🔹 Accuracy Graph
plt.subplot(1, 2, 2)
plt.plot(range(1, len(accuracy_per_round) + 1), accuracy_per_round, marker='o', linestyle='-', color='blue', label="Accuracy")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.title("Federated Learning Accuracy per Round")
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 10))

# 🔹 Training Loss Graph
plt.subplot(2, 2, 1)
plt.plot(range(1, len(train_loss_per_round) + 1), train_loss_per_round, marker='o', linestyle='-', color='blue', label="Training Loss")
plt.xlabel("Loss")  # Swapped with ylabel
plt.ylabel("Rounds")  # Swapped with xlabel
plt.title("Training Loss per Round")
plt.legend()

# 🔹 Training Accuracy Graph
plt.subplot(2, 2, 2)
plt.plot(range(1, len(train_accuracy_per_round) + 1), train_accuracy_per_round, marker='o', linestyle='-', color='red', label="Training Accuracy")
plt.xlabel("Accuracy")  # Swapped with ylabel
plt.ylabel("Rounds")  # Swapped with xlabel
plt.title("Training Accuracy per Round")
plt.legend()
# 🔹 Testing Loss Graph
plt.subplot(2, 2, 3)
plt.plot(range(1, len(test_loss_per_round) + 1), test_loss_per_round, marker='o', linestyle='-', color='green', label="Test Loss")
plt.xlabel("Loss")  # Swapped with ylabel
plt.ylabel("Rounds")  # Swapped with xlabel
plt.title("Testing Loss per Round")
plt.legend()

# 🔹 Testing Accuracy Graph
plt.subplot(2, 2, 4)
plt.plot(range(1, len(test_accuracy_per_round) + 1), test_accuracy_per_round, marker='o', linestyle='-', color='purple', label="Test Accuracy")
plt.xlabel("Accuracy")  # Swapped with ylabel
plt.ylabel("Rounds")  # Swapped with xlabel
plt.title("Testing Accuracy per Round")
plt.legend()

plt.tight_layout()
plt.show()

plt.tight_layout()
plt.show()

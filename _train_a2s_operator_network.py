import numpy as np
import torch
import json
import a2s
import matplotlib.pyplot as plt
import os
import joblib
torch.manual_seed(0)


experiment_name = "activation-to-shape-operator-network"
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
run_name, custom_path = a2s.get_next_run_path(script_dir, experiment_name) # Create new training directory

params = {
    'activation': "tanh",
    'learning_rate': 0.001,  # Set a constant learning rate
    'learning_rate_type': "exponential_decay",  # constant or exponential_decay
    'epochs': 30,
    'batch_size': 32,  # Number of samples processed in one batch, default is 32 
    'weight_init': "xavier",
    'optimizer': "adam",
    'metrics': ["l2 relative error"],
    'display_every': 1,  # Display training progress every xx epochs
}

# === Load and preprocess data ===
base_path = "CHOOSE-YOUR-DATA-PATH"
gamma, r, z = a2s.load_data(base_path)

(X_train, Y_train, Z_train), (X_valid, Y_valid, Z_valid), (X_test, Y_test, Z_test), scalers = a2s.preprocess_data(X=gamma, Y=r, z=z)
a2s.plot_input_output_distributions(X_train, X_test, Y_train, Y_test)


train_loader = a2s.create_dataloader(X=X_train, Y=Y_train, Z=Z_train, shuffle=True, batch_size=params["batch_size"])
valid_loader = a2s.create_dataloader(X=X_valid, Y=Y_valid, Z=Z_valid, shuffle=False, batch_size=params["batch_size"])
test_loader = a2s.create_dataloader(X=X_test, Y=Y_test, Z=Z_test, shuffle=False, batch_size=params["batch_size"])
data_loaders = {"train": train_loader, "valid": valid_loader, "test": test_loader}

# === Learning ===
p = 64 # influences the dimension of the last layers of the branch and trunk network
dim = r.shape[-1] # should be 3
params['layers_branch'] = [dim, p, p, p, p*dim]
params['layers_trunk'] = [1, p, p, p, p*dim]
model = a2s.DeepONet(params=params)
model, history = a2s.train_don_model(model, z, data_loaders, custom_path)

# === Test the model ===
Y_test, Y_pred, test_losses = a2s.test_don_model(model, test_loader, scalers)

# === Visualization ===
a2s.plot_pytorch_training(history, save_path=custom_path)
a2s.plot_prediction(Y_test, Y_pred, save_path=custom_path, type="DON")
a2s.plot_prediction(Y_test, Y_pred, sample_idx=12, save_path=custom_path, type="DON", nz=z.shape[1])
a2s.plot_prediction(Y_test, Y_pred, sample_idx=89, save_path=custom_path, type="DON", nz=z.shape[1])

# === Logging ===
# Save Metadata first
params["train_samples"] = X_train.shape[0]
params["valid_samples"] = X_valid.shape[0]
with open(str(custom_path + "/metadata.json"), "w") as f:
    json.dump(params, f, indent=4)
with open(str(custom_path + "/history.json"), "w") as f:
    json.dump(history, f, indent=4)
with open(str(custom_path + "/test_losses.json"), "w") as f:
    json.dump(test_losses, f, indent=4)

joblib.dump(scalers['X'], str(custom_path +'/scalerX.save')) 
joblib.dump(scalers['Y'], str(custom_path +'/scalerY.save')) 
joblib.dump(scalers['Z'], str(custom_path +'/scalerZ.save')) 

# Optional
mlf_track = False
if mlf_track:
    import mlflow
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    mlflow.set_experiment(experiment_name)
    print("MLflow tracking URI set to http://127.0.0.1:8080")

    with mlflow.start_run(run_name=run_name):
        if params!=None:
            mlflow.log_params(params)
        # Log training and validation losses
        best_idx = history["epochs"].index(history["best_step"])        
        mlflow.log_metric("best_train_loss", history["train_losses"][best_idx])
        mlflow.log_metric("best_val_loss", history["valid_losses"][best_idx])

        # Log artifacts
        mlflow.log_artifact(custom_path + "/metadata.json")
        mlflow.log_artifact(custom_path + "/training_curve.png")
        mlflow.log_artifact(custom_path + "/test_results.png")
        mlflow.log_artifact(custom_path + "/best_model")

        # Log test performance
        mlflow.log_metric("test_loss_mse", test_losses["MSE"])
        mlflow.log_metric("test_loss_l2", test_losses["L2"])

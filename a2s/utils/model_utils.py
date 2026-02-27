# maybe put that nn stuff in here?
import torch
import numpy as np

def train_don_model(model, z, data_loaders, save_path):
    """Train the model and save the best version."""
    device = 'cpu'
    if model.params["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=model.params["learning_rate"])
    elif model.params["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=model.params["learning_rate"])

    # Add a learning rate scheduler if specified
    if model.params["learning_rate_type"] == "exponential_decay":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    else:
        scheduler = None  # No scheduler for constant learning rate
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1) # This makes jumps
    loss_fn = torch.nn.MSELoss()  # Mean Squared Error loss

    history = {"train_losses": [], "valid_losses": [], "metrics_history": [], "learning_rates": [], "epochs": [], "best_step": None}
    best_val_loss = float("inf")
    best_model_state = None
    
    for epoch in range(model.params["epochs"]):
        # Training phase
        model.train()
        train_loss = 0.0

        for X_batch, Y_batch, Z_batch in data_loaders["train"]:
            X_batch, Y_batch, Z_batch = X_batch.to(device), Y_batch.to(device), Z_batch.to(device)

            # Forward pass
            Y_pred = model(X_batch, Z_batch) # TODO correct?
            loss = loss_fn(Y_pred, Y_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_loss /= len(data_loaders["train"])

        # Store the current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        history["learning_rates"].append(current_lr)

        # validate and print every xx epochs
        if epoch % model.params["display_every"] == 0 or epoch == model.params["epochs"] - 1:
            # Perform validation
            model.eval()
            val_loss = 0.0
            metrics = []
            with torch.no_grad():
                for X_batch, Y_batch, Z_batch in data_loaders["valid"]:
                    X_batch, Y_batch, Z_batch = X_batch.to(device), Y_batch.to(device), Z_batch.to(device)
                    Y_pred = model(X_batch, Z_batch) # TODO Need to pass z here?
                    loss = loss_fn(Y_pred, Y_batch)
                    val_loss += loss.item()
                    if "l2 relative error" in model.params["metrics"]:
                        l2_error = torch.norm(Y_batch - Y_pred) / torch.norm(Y_batch)
                        metrics.append(l2_error.item())

            val_loss /= len(data_loaders["valid"])
            history["epochs"].append(epoch)
            history["train_losses"].append(train_loss)
            history["valid_losses"].append(val_loss)
            if metrics:
                history["metrics_history"].append(sum(metrics) / len(metrics))

            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                history["best_step"] = epoch
                best_model_state = model.state_dict()

            if metrics:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4e}, Val Loss = {val_loss:.4e}, Val Metrics: {metrics[-1]:.4e}")
            else:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4e}, Val Loss = {val_loss:.4e}")
        if scheduler is not None:
            # Update the learning rate
            scheduler.step()

    # Save the best model to disk
    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), str(save_path + "/best_model"))
    return model, history

def test_don_model(model, test_loader, scalers):
    model.eval()
    all_test = []
    all_preds = []  # To store predictions
    with torch.no_grad():
        for X_batch, Y_batch, Z_batch in test_loader:
            Y_test_temp = Y_batch.numpy() 
            Y_pred_temp = model(X_batch, Z_batch)
            all_test.append(Y_test_temp)
            all_preds.append(Y_pred_temp.cpu().numpy())  # Collect predictions

    # Concatenate all predictions into a single array
    Y_pred_scaled = np.vstack(all_preds)
    Y_test_scaled = np.vstack(all_test)  # Shape: (n_test, output_dim)

    # Reverse scaling and compute test loss
    Y_test = scalers['Y'].inverse_transform(Y_test_scaled)  # Unscale Y_test
    Y_pred = scalers['Y'].inverse_transform(Y_pred_scaled)  # Unscale Y_pred
    test_loss_mse = np.mean((Y_test - Y_pred) ** 2)
    test_loss_l2 = np.linalg.norm(Y_test - Y_pred) / np.linalg.norm(Y_test)

    print(f"Test Loss (MSE): {test_loss_mse:.4e}")
    print(f"Test Loss (L2): {test_loss_l2:.4e}")
    losses = {
        "MSE": np.float64(test_loss_mse),
        "L2": np.float64(test_loss_l2)
    }
    return Y_test, Y_pred, losses

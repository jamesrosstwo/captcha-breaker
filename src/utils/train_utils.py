import torch
import torchvision
import time

# TODO maybe use a single config dict or kwargs so we can have tons of options
def train(model: torch.nn.Module, optimizer: torch.optim.Optimizer, train_loader, val_loader, loss_fn: torch.nn.BCELoss, epochs: 10, device=torch.device('cuda')):
    for epoch in range(epochs):
        model.train()
        correct_train = 0
        # Training
        train_loss_sum = 0.0  # Reset train_loss_sum for each epoch
        for (image, question, targets) in train_loader:
            image, targets = image.to(device), targets.to(device)

            # Calculate Training Loss
            optimizer.zero_grad()  # Zero the parameter gradients
            
            predictions = model(image, question)
            train_loss = loss_fn(predictions, targets)
            train_loss.backward()
            optimizer.step()  # Perform a single optimization step

            train_loss_sum += train_loss.item() * image.size(0)
            
            predicted = (predictions >= 0.5).float()
            correct_train += (predicted == targets).float().sum().item()

        # Calculate average losses
        train_accuracy = correct_train / len(train_loader.dataset)
        train_loss_avg = train_loss_sum / len(train_loader.dataset)
            
        # Validation
        model.eval()  # Set model to evaluation mode
        val_loss_sum = 0.0
        correct_val = 0  # Variable to track the number of correctly predicted instances

        with torch.no_grad():  # Disable gradient computation during evaluation
            for (val_image, val_question, val_targets) in val_loader:
                val_image, val_targets = val_image.to(device), val_targets.to(device)

                val_predictions = model(val_image, val_question)
                val_loss = loss_fn(val_predictions, val_targets)
                val_loss_sum += val_loss.item() * val_image.size(0)

                predicted = (val_predictions >= 0.5).float()
                correct_val += (predicted == val_targets).float().sum().item()

        val_loss_avg = val_loss_sum / len(val_loader.dataset)
        val_accuracy = correct_val / len(val_loader.dataset)

        print(f'Epoch {epoch+1}/{epochs}, Training Loss: {train_loss_avg:.4f}, Validation Loss: {val_loss_avg:.4f}, Train Accuracy: {train_accuracy}, Validation Accuracy: {val_accuracy:.4f}')

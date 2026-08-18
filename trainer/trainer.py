from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from config import CLASS_NAMES, DEVICE


class AdvancedTrainer:
    def __init__(self, model, device=DEVICE, logger=None):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "lr": [],
        }
        self.logger = logger

    def train(
        self,
        train_loader,
        val_loader,
        backbone,
        kernel_name,
        epochs=100,
        patience=10,
        checkpoint_path=None,
        metadata=None,
    ):
        if epochs <= 0:
            raise ValueError(f"epochs must be positive, got {epochs}")
        if patience <= 0:
            raise ValueError(f"patience must be positive, got {patience}")
        if len(train_loader) == 0 or len(val_loader) == 0:
            raise ValueError("Training and validation loaders must not be empty")

        criterion = nn.CrossEntropyLoss()
        trainable_parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not trainable_parameters:
            raise ValueError("Model has no trainable parameters")
        optimizer = optim.AdamW(trainable_parameters, lr=2e-4, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=5,
            factor=0.5,
        )

        best_acc = float("-inf")
        best_epoch = 0
        early_stop_counter = 0
        checkpoint = Path(checkpoint_path).expanduser().resolve() if checkpoint_path else None
        if checkpoint:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
        run_metadata = dict(metadata or {})

        for epoch in range(1, epochs + 1):
            self.model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                predicted = outputs.argmax(dim=1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            average_train_loss = train_loss / len(train_loader)
            train_acc = 100.0 * correct / total
            val_loss, val_acc = self.validate(val_loader, criterion)
            current_lr = optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(average_train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)
            scheduler.step(val_acc)

            improved = val_acc > best_acc
            if improved:
                best_acc = val_acc
                best_epoch = epoch
                early_stop_counter = 0
                if checkpoint:
                    torch.save(
                        {
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "epoch": epoch,
                            "best_val_acc": best_acc,
                            "backbone": backbone,
                            "kernel_name": kernel_name,
                            "class_names": list(run_metadata.get("class_names", CLASS_NAMES)),
                            "seed": run_metadata.get("seed"),
                            "config": run_metadata,
                        },
                        checkpoint,
                    )
            else:
                early_stop_counter += 1

            if self.logger:
                self.logger.log_metrics(
                    backbone=backbone,
                    kernel_name=kernel_name,
                    epoch=epoch,
                    train_loss=average_train_loss,
                    val_loss=val_loss,
                    train_acc=train_acc,
                    val_acc=val_acc,
                    learning_rate=current_lr,
                )

            print(
                f"Epoch {epoch}/{epochs}: Train Loss={average_train_loss:.4f} "
                f"Train Acc={train_acc:.2f}% | Val Loss={val_loss:.4f} "
                f"Val Acc={val_acc:.2f}% | LR={current_lr:.2e}"
            )

            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}")
                break

        if checkpoint and not checkpoint.is_file():
            raise RuntimeError(f"Training completed without creating checkpoint: {checkpoint}")
        return {
            "best_val_acc": best_acc,
            "best_epoch": best_epoch,
            "epochs_completed": len(self.history["train_loss"]),
            "checkpoint_path": str(checkpoint) if checkpoint else None,
        }

    def validate(self, loader, criterion):
        if len(loader) == 0:
            raise ValueError("Validation loader must not be empty")
        self.model.eval()
        loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                loss += criterion(outputs, labels).item()
                predicted = outputs.argmax(dim=1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        return loss / len(loader), 100.0 * correct / total

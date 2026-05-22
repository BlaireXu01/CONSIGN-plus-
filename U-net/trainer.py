import torch
from torch.cuda.amp import autocast
from helpers import plot_losses, recursive_to
import matplotlib.pyplot as plt
from pytictoc import TicToc


class FSSTrainer:
    def __init__(self, model, optimizer, scaler, lr_sched, training_loader, val_loader,
                 device, name, patience, half_precision):
        self._model = model
        self._optimizer = optimizer
        self._lr_sched = lr_sched
        self._scaler = scaler

        self._training_loader = training_loader
        self._val_loader = val_loader

        self._device = device
        self._name = name
        self._epoch = 0
        self._plot_train = []
        self._plot_val = []
        self._patience = patience
        self._half_precision = half_precision

    def train(self, max_epochs):
        print(f"Training epochs {self._epoch + 1} to {max_epochs}. Moving model to {self._device}.")
        self._model.to(self._device)
        # counter and loss for early stopping:
        patience = self._patience
        counter = 0
        best_val_loss = 0

        for epoch in range(self._epoch + 1, max_epochs + 1):
            self._epoch = epoch
            t = TicToc()
            print(f"Starting epoch {epoch}/{max_epochs} with lr={self._lr_sched.get_last_lr()}")
            t.tic()
            patience, counter, best_val_loss, train_loss, val_loss, end_flag = self._train_epoch(counter,best_val_loss, patience)
            t.toc()
            print("###################")
            if train_loss == torch.nan or val_loss == 0.0:
                print("NaN Error")
                break

            self._plot_train.append(train_loss)
            self._plot_val.append(val_loss)

            plot_losses(self._plot_val, self._plot_train, str(self._name))

            self._lr_sched.step()

            if epoch%100==0:
                self.save_checkpoint('latest')

            if end_flag == 1:
                break


        print(f"Finished training!")

        plot_losses(self._plot_val, self._plot_train, str(self._plot_val[-1]))


    def _train_epoch(self, counter,best_val_loss, patience):
        """Do one epoch of training and validation."""

        self._model.train(True)
        _, _, _ , train_loss, _ = self._run_epoch(counter, best_val_loss, patience, mode='train', loader=self._training_loader)
        torch.cuda.empty_cache()

        self._model.train(False)
        with torch.no_grad():
            patience, counter, best_val_loss, val_loss, end_flag = self._run_epoch(counter, best_val_loss, patience, mode='val', loader=self._val_loader)

        return patience, counter, best_val_loss, train_loss, val_loss, end_flag


    def _run_epoch(self, counter, best_val_loss, patience, mode, loader):
        end_flag = 0
        count = 0
        # initialize val loss
        if mode=='val':
            val_loss = 0.0
            is_val = True
        else:
            train_loss = 0.0
            is_val = False

        # start loop over the dataset

        for query in loader:
            query = recursive_to(query, self._device)

            self._optimizer.zero_grad()

            if self._half_precision:
                with autocast():
                    loss = self._model(query, is_val, epoch=self._epoch)
            else:
                loss_cnn = self._model(query, is_val, epoch=self._epoch)

            # backprop
            if mode == 'train':
                if self._half_precision:
                    self._scaler.scale(loss).backward()
                    self._scaler.step(self._optimizer)
                    self._scaler.update()
                else:
                    loss_cnn.backward()
                    self._optimizer.step()

                train_loss += (loss_cnn).item()

            # update validation loss
            elif mode == 'val':
                count+=1
                val_loss += loss_cnn

        if mode == 'train':
            print(f"Avg Train Loss={train_loss / len(loader)}")

        # check if the validation loss decreased
        else:
            avg_val_loss = val_loss / len(loader)

            print(f"Avg DICE Loss={avg_val_loss}")
            if avg_val_loss > best_val_loss:
                best_val_loss = avg_val_loss
                print(f"Best DICE Loss={best_val_loss}")
                print('****')
                self.save_checkpoint('best')
                counter = 0  # Reset the counter if validation loss improves
            else:
                counter += 1

            if counter >= patience:
                print("Early stopping.")
                end_flag = 1
        if mode == 'train':
            return patience, counter, best_val_loss, train_loss / len(loader), 0
        else:
            return patience, counter, best_val_loss, val_loss / len(loader), end_flag

    def save_checkpoint(self, string):
        """Saves a checkpoint of the network and other variables."""
        state = {
            'epoch': self._epoch,
            'net_type': type(self._model).__name__,
            'net': self._model.state_dict(),
            'optimizer' : self._optimizer.state_dict(),
            'device' : self._device,
        }
        file_path = str(self._name)+string+'.pth'
        torch.save(state, file_path)

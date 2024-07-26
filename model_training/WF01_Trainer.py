import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from torch.utils.data import DataLoader, Subset

from util.matrices_computation import calculate_metrics


class WF01_Trainer():
    def __init__(self, model_make, train_set, test_set, k_folds, epochs, batch_size, criterion_make, optimizer_make,
                 scheduler_make, random_seed):
        # define some basic parameters
        self.random_seed = random_seed

        self.model = None
        self.model_make = model_make
        self.train_set = train_set
        self.test_set = test_set

        self.k_folds = k_folds
        self.epochs = epochs
        self.batch_size = batch_size

        # Define loss function and optimizer
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.criterion_make = criterion_make
        self.optimizer_make = optimizer_make
        self.scheduler_make = scheduler_make

        self.metrics_df = None

        self._gpu_check()
        self._randomness_contrtol()

    def _randomness_contrtol(self):
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)

        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

    def _gpu_check(self):
        print(f'CUDA availability check: {torch.cuda.is_available()}')
        print(f'CUDA devices count: {torch.cuda.device_count()}')
        print(f'CUDA devices name: {torch.cuda.get_device_name(0)}')
        print(f'CUDA current device: {torch.cuda.current_device()}')

    def _initialize_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _reset_models(self):
        self.model = self.model_make().cuda()
        self.model.apply(self._initialize_weights)
        self.optimizer = self.optimizer_make(self.model)
        self.scheduler = self.scheduler_make(self.optimizer)
        self.criterion = self.criterion_make()

    def _metrics_dict(self, *metrics):
        dict_metrics = {
            'fold': metrics[0],
            'epoch': metrics[1],
            'phase': metrics[2],  # 'train' 'val' 'test'
            'loss': metrics[3],
            'accuracy': metrics[4],
            'precision': metrics[5],
            'recall': metrics[6],
            'specificity': metrics[7],
            'f1_score': metrics[8],
            'tp': metrics[9],
            'fp': metrics[10],
            'tn': metrics[11],
            'fn': metrics[12]
        }

        return dict_metrics

    def save_metrics(self, path, filename):
        if not os.path.exists(path):
            os.makedirs(path)

        if self.metrics_df is None:
            raise Exception('No training/testing metrics available')
        else:
            self.metrics_df.to_csv(f'{path}/{filename}', index=False)
            print(f'{filename} is saved to {path}')

    def _train_model(self, train_loader):
        self.model.train()
        total_loss = 0
        all_labels_train = []
        all_outputs_train = []

        # Training Loop
        for inputs_2d, inputs_1d, labels in train_loader:
            inputs_2d, inputs_1d, labels = inputs_2d.cuda(), inputs_1d.cuda(), labels.cuda()

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs_2d, inputs_1d)

            # Compute the loss
            loss = self.criterion(outputs, labels)
            total_loss += loss.item()

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            all_labels_train.append(labels)
            all_outputs_train.append(outputs)

        acc_train, pre_train, rec_train, spe_train, f1_train, cm_train = calculate_metrics(all_outputs_train,
                                                                                           all_labels_train)
        average_loss_train = total_loss / len(train_loader)

        return average_loss_train, acc_train, pre_train, rec_train, spe_train, f1_train, cm_train

    def _val_model(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_labels_val = []
        all_outputs_val = []

        with torch.no_grad():
            for inputs_2d, inputs_1d, labels in val_loader:
                inputs_2d, inputs_1d, labels = inputs_2d.cuda(), inputs_1d.cuda(), labels.cuda()

                # Forward pass
                outputs = self.model(inputs_2d, inputs_1d)

                # Compute the loss
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                all_labels_val.append(labels)
                all_outputs_val.append(outputs)

        acc_val, pre_val, rec_val, spe_val, f1_val, cm_val = calculate_metrics(all_outputs_val, all_labels_val)

        average_loss_val = total_loss / len(val_loader)

        return average_loss_val, acc_val, pre_val, rec_val, spe_val, f1_val, cm_val

    def K_fold_train_and_val(self, model_save_path, model_save_name):
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=self.random_seed)

        metrics_df = pd.DataFrame(columns=['fold', 'epoch', 'phase',
                                        'loss', 'accuracy', 'precision', 'recall', 'specificity', 'f1_score',
                                        'tp', 'fp', 'tn', 'fn'])

        for fold, (train_ids, val_ids) in enumerate(kf.split(self.train_set)):
            print(f'Fold {fold}/{self.k_folds-1}...')

            # create new model, optimizor and schedular
            self._reset_models()

            # Sample elements randomly from a given list of ids
            train_subsampler = Subset(self.train_set, train_ids)
            val_subsampler = Subset(self.train_set, val_ids)

            # Define data loaders for training and testing data in this fold
            train_loader = DataLoader(train_subsampler, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_subsampler, batch_size=self.batch_size, shuffle=False)

            for epoch in range(self.epochs):
                print(f'Epoch {epoch}/{self.epochs-1}')
                start_time = time.time()

                train_loss, train_acc, train_pre, train_rec, train_spe, train_f1, train_cm = self._train_model(train_loader)
                val_loss, val_acc, val_pre, val_rec, val_spe, val_f1, val_cm = self._val_model(val_loader)

                dict_train = self._metrics_dict(fold, epoch, 'train',
                    train_loss, train_acc, train_pre, train_rec, train_spe, train_f1, *train_cm)

                dict_val = self._metrics_dict(fold, epoch, 'val',
                                                val_loss, val_acc, val_pre, val_rec, val_spe, val_f1,
                                                *val_cm)

                temp_df_train = pd.DataFrame([dict_train])
                temp_df_val = pd.DataFrame([dict_val])
                metrics_df = pd.concat([metrics_df, temp_df_train, temp_df_val], ignore_index=True)

                print(f'Train Loss {train_loss:.6f}, Acc: {train_acc*100:.4f}%, Pre: {train_pre*100:.4f}%, Rec: {train_rec*100:.4f}%, Spe: {train_spe*100:.4f}%, F1: {train_f1*100:.6f}%')
                print(f'Val   Loss {val_loss:.6f}, Acc: {val_acc*100:.4f}%, Pre: {val_pre*100:.4f}%, Rec: {val_rec*100:.4f}%, Spe: {val_spe*100:.4f}%, F1: {val_f1*100:.6f}%')

                ### experimental observation ###
                temp_df_test = self.test_model(reload=False, fold=fold, epoch=epoch)
                metrics_df = pd.concat([metrics_df, temp_df_test], ignore_index=True)


                # scheduler.step()
                self.scheduler.step()

                # save the model for each fold
                torch.save(self.model.state_dict(), f'{model_save_path}/{model_save_name}_fold{fold}_epoch{epoch}.pth')

                end_time = time.time()
                epoch_duration = end_time - start_time
                print(f'[{epoch_duration:.2f} seconds]\n')

        self.metrics_df = metrics_df

    def train_on_full_dataset(self, model_save_path, model_save_name):
        print("Start training on the full dataset")

        metrics_df = pd.DataFrame(columns=['fold', 'epoch', 'phase',
                                           'loss', 'accuracy', 'precision', 'recall', 'specificity', 'f1_score',
                                           'tp', 'fp', 'tn', 'fn'])

        # Reset model, optimizer, criterion, and scheduler for full training
        self._reset_models()

        train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            print(f'Epoch {epoch}/{self.epochs-1}')
            start_time = time.time()

            train_loss, train_acc, train_pre, train_rec, train_spe, train_f1, train_cm = self._train_model(train_loader)

            dict_fulltrain = self._metrics_dict(1, epoch, 'full_train', train_loss, train_acc, train_pre, train_rec, train_spe, train_f1, *train_cm)
            temp_df = pd.DataFrame([dict_fulltrain])
            metrics_df = pd.concat([metrics_df, temp_df], ignore_index=True)

            self.scheduler.step()

            print(f'Train Loss {train_loss:.6f}, Acc: {train_acc*100:.4f}%, Pre: {train_pre*100:.4f}%, Rec: {train_rec*100:.4f}%, Spe: {train_spe*100:.4f}%, F1: {train_f1*100:.6f}')

            if (epoch+1)%10 == 0:
                torch.save(self.model.state_dict(), f'{model_save_path}/{model_save_name}_epoch{epoch}.pth')

            end_time = time.time()
            epoch_duration = end_time - start_time
            print(f' --{epoch_duration:.2f} seconds\n')

        self.metrics_df = metrics_df



    def test_model(self, reload, metrics_save_path=None, model_weights_path=None, fold=None, epoch=None):
        # reload model
        if reload:
            self.model = self.model_make().cuda()
            self.model.load_state_dict(torch.load(model_weights_path))

        metrics_df = pd.DataFrame(columns=['fold', 'epoch', 'phase',
                                           'loss', 'accuracy', 'precision', 'recall', 'specificity', 'f1_score',
                                           'tp', 'fp', 'tn', 'fn'])

        # set model to correct mode
        self.model.eval()

        total_loss = 0
        all_labels_test = []
        all_outputs_test = []

        test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)

        with torch.no_grad():
            for inputs_2d, inputs_1d, labels in test_loader:
                # Move data to GPU if available
                inputs_2d, inputs_1d, labels = inputs_2d.cuda(), inputs_1d.cuda(), labels.cuda()

                # Forward pass
                outputs = self.model(inputs_2d, inputs_1d)

                # Compute loss
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                all_labels_test.append(labels)
                all_outputs_test.append(outputs)

        acc_test, pre_test, rec_test, spe_test, f1_test, cm_test = calculate_metrics(all_outputs_test, all_labels_test)

        # Calculate average loss and accuracy
        test_loss = total_loss / len(test_loader)

        dict_test = self._metrics_dict(fold, epoch, 'test',
                    test_loss, acc_test, pre_test, rec_test, spe_test, f1_test, *cm_test)
        temp_df = pd.DataFrame([dict_test])

        if not metrics_save_path is None:
            metrics_df = pd.concat([metrics_df, temp_df], ignore_index=True)
            metrics_df.to_csv(metrics_save_path)

        print(f'Test  Loss {test_loss:.6f}, Acc: {acc_test*100:.4f}%, Pre: {pre_test*100:.4f}%, Rec: {rec_test*100:.4f}%, Spe: {spe_test*100:.4f}%, F1: {f1_test*100:.6f}')

        return temp_df

    def get_AUC(self, model_weights_path=None):
        self.model = self.model_make().cuda()
        self.model.load_state_dict(torch.load(model_weights_path))

        # set model to correct mode
        self.model.eval()

        total_loss = 0
        all_labels_test = []
        all_outputs_test = []

        test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)

        with torch.no_grad():
            for inputs_2d, inputs_1d, labels in test_loader:
                # Move data to GPU if available
                inputs_2d, inputs_1d, labels = inputs_2d.cuda(), inputs_1d.cuda(), labels.cuda()

                # Forward pass
                outputs = self.model(inputs_2d, inputs_1d)

                # Flatten the outputs and labels to 1D arrays
                all_labels_test.append(labels.view(-1).cpu().numpy())
                all_outputs_test.append(outputs.view(-1).cpu().numpy())

        labels_array = np.concatenate(all_labels_test)
        outputs_array = np.concatenate(all_outputs_test)

        # Calculate ROC AUC and PR AUC using the true labels and predicted probabilities
        roc_auc = roc_auc_score(labels_array, outputs_array)
        precision, recall, _ = precision_recall_curve(labels_array, outputs_array)
        pr_auc = auc(recall, precision)

        return roc_auc, pr_auc









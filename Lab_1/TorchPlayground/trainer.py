from datetime import datetime

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from dataset import Circles
from net import StudyAI
from visualize_utils import make_meshgrid, predict_proba_om_mesh, plot_predictions

from torch.utils.tensorboard import SummaryWriter


class Trainer:

    def __init__(self, model, lr, optimzer=None, critetion=None):

        self.model = model
        self.criterion = CrossEntropyLoss()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        cuda = torch.cuda.is_available()
        self.device = torch.device("coda:0" if cuda else "cpu")

        self.experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter('runs/' + self.experiment_name)


    def fit(self, train_dataloader, n_epochs):

        self.model.train()

        for epoch in range(n_epochs):
            print("epoch: ", epoch)
            epoch_loss = 0

            for i, (x_batch, y_batch) in enumerate(train_dataloader):

                x_batch = torch.tensor(x_batch)
                y_batch = torch.tensor(y_batch, dtype=torch.long)

                if(epoch == 0) and (i == 0):
                    self.writer.add_graph(self.model, x_batch)

                self.optimizer.zero_grad()
                output = self.model(x_batch)
                loss = self.criterion(output, y_batch)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                # if (epoch > 40) and (i == 0):

                if  (epoch < 3):
                    print("save image", epoch, " ", i)

                    train_dataset = train_dataloader.dataset

                    X_train, X_test, y_train, y_test = get_data_from_datasets(train_dataset,
                                                                              train_dataset)

                    xx, yy = make_meshgrid(X_train, X_test, y_train, y_test)
                    Z = predict_proba_om_mesh_tensor(self, xx, yy)

                    plot_title = "nn_predictions/nn_predictions_{}_{}.png".format(str(epoch).rjust(4, '0'), str(i).rjust(4, '0'))

                    plot_predictions(xx, yy, Z, X_train=X_train, X_test=X_test,
                                     y_train=y_train, y_test=y_test,
                                     plot_name=plot_title)




            print(epoch_loss / len(train_dataloader))

            self.writer.add_scalar('trainig loss on epoch end', epoch_loss)


    def predict(self, test_dataloader):

        all_outputs = torch.tensor([], dtype=torch.long)
        self.model.eval()

        with torch.no_grad():
            for i, (x_batch, y_batch) in enumerate(test_dataloader):
                output_batch = self.model(x_batch)

                _, predicted = torch.max(output_batch.data, 1)
                print(predicted)

                all_outputs = torch.cat((all_outputs, predicted), 0)

        return all_outputs

    def predict_proba(self, test_dataloader):

        all_outputs = torch.tensor([], dtype=torch.long)

        self.model.eval()

        with torch.no_grad():
            for i, (x_batch, y_batch) in enumerate(test_dataloader):
                output_batch = self.model(x_batch)
                all_outputs = torch.cat((all_outputs, output_batch), 0)

        return all_outputs

    def predict_proba_tensor(self, test_dataloader):
        self.model.eval()

        with torch.no_grad():
            output = self.model(test_dataloader)

        return output


def get_data_from_datasets(train_dataset, test_dataset):
    X_train = train_dataset.X.astype(np.float32)
    X_test = test_dataset.X.astype(np.float32)

    y_train = train_dataset.y.astype(np.int)
    y_test = test_dataset.y.astype(np.int)

    return X_train, X_test, y_train, y_test


def predict_proba_om_mesh_tensor(clf, xx, yy):
    q = torch.Tensor(np.c_[xx.ravel(), yy.ravel()])
    Z = clf.predict_proba_tensor(q)[:, 1]
    Z = Z.reshape(xx.shape)
    return Z

if __name__ == "__main__":

    layers_list_example = [(2, 5), (5, 2)]
    model = StudyAI(layers_list_example)

    trainer = Trainer(model, lr=0.01)
    print(trainer.device)

    train_dataset = Circles(n_samples=5000, shuffle=True, noise=0.1, random_state=0, factor=.5)
    test_dataset = Circles(n_samples=1000, shuffle=True, noise=0.1, random_state=2, factor=.5)

    print(train_dataset)


    train_dataloader = DataLoader(train_dataset, batch_size=50, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=50, shuffle=True)

    trainer.fit(train_dataloader, n_epochs=100)


    test_predicion_proba = trainer.predict_proba(test_dataloader)

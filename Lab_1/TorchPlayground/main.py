from torch.utils.data import DataLoader

from TorchPlayground.dataset import Circles, Moons
from TorchPlayground.net import StudyAI
from TorchPlayground.trainer import Trainer, get_data_from_datasets, predict_proba_om_mesh_tensor
from TorchPlayground.visualize_utils import make_meshgrid, plot_predictions


if __name__ == "__main__":

    layers_list_example = [(2, 7, 'relu'), (7, 6, 'relu'), (6, 4, 'relu'),  (4, 2, 'tanh')]

    model = StudyAI(layers_list_example)

    trainer = Trainer(model, lr=0.01)
    print(trainer.device)

    # train_dataset = Circles(n_samples=5000, shuffle=True, noise=0.1, random_state=0, factor=.5)
    # test_dataset = Circles(n_samples=1000, shuffle=True, noise=0.1, random_state=2, factor=.5)

    train_dataset = Moons(n_samples=5000, shuffle=True, noise=0.1, random_state=0)
    test_dataset = Moons(n_samples=1000, shuffle=True, noise=0.1, random_state=2)

    print(train_dataset)


    train_dataloader = DataLoader(train_dataset, batch_size=50, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=50, shuffle=True)

    trainer.fit(train_dataloader, n_epochs=20)


    test_predicion_proba = trainer.predict_proba(test_dataloader)

    X_train, X_test, y_train, y_test = get_data_from_datasets(train_dataset,
                                                              test_dataset)

    xx, yy = make_meshgrid(X_train, X_test, y_train, y_test)

    Z = predict_proba_om_mesh_tensor(trainer, xx, yy)

    plot_predictions(xx, yy, Z, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
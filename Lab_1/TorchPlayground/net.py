#Архитектура сети
import logging

import torch

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(processName)-10s %(name)s - %(levelname)s: %(message)s")


class StudyAI(torch.nn.Module):

    def __init__(self, layers_list):
        super(StudyAI, self).__init__()

        logger = logging.getLogger("net")
        file_handler = logging.FileHandler("NetArchitecture.log")
        logger.addHandler(file_handler)

        layers = []

        for i, layer in enumerate(layers_list):
            layers.append(torch.nn.Linear(layer[0], layer[1]))
            logger.info(f"Добавлен линейный слой {i} с параметрами input: {layer[0]}, output: {layer[1]}")

            if (layer[2] == 'sigmoid'):
                logger.info(f"Активационная функция {i} - го слоя - Sigmoid")
                layers.append(torch.nn.Sigmoid())

            elif (layer[2] == 'relu'):
                logger.info(f"Активационная функция {i} - го слоя - ReLU")
                layers.append(torch.nn.ReLU())

            elif (layer[2] == 'tanh'):
                logger.info(f"Активационная функция {i} - го слоя - Tanh")
                layers.append(torch.nn.Tanh())

            else:
                logger.info(f"Задана неверная функция в слое {i}")



        layers.append(torch.nn.Softmax())
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":

    layers_list_example = [(2, 5, 'sigmoid'), (5, 3, 'tanh'), (3, 6, 'relu'), (6, 2, 'relu')]
    example_net = StudyAI(layers_list_example)
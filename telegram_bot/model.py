from __future__ import print_function, division
from PIL import Image as PIL_Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
#from fastai.vision import load_learner, Image

from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
#import numpy as np
import torchvision
from torchvision import datasets, models, transforms
#import matplotlib.pyplot as plt
#import time
#import os
import torchvision
#from torch.autograd import Variable


#model.load_state_dict(torch.load('Inceptionv3.pth'))
# В данном классе мы хотим полностью производить всю обработку картинок, которые поступают к нам из телеграма.
# Это всего лишь заготовка, поэтому не стесняйтесь менять имена функций, добавлять аргументы, свои классы и
# все такое.
class ClassPredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to_tensor = transforms.ToTensor()
        self.resnet = torchvision.models.resnet18()
        self.resnet.avgpool = nn.AdaptiveAvgPool2d(1)
        self.model = models.resnet18(pretrained=True)
        self.model = self.model.cuda()
        self.model.classifier = nn.Linear(25088, 2)

        self.model.load_state_dict(torch.load('../model/Inceptionv3.pth'))


    def predict(self, img_stream):
        # Этот метод по переданным картинкам в каком-то формате (PIL картинка, BytesIO с картинкой
        # или numpy array на ваш выбор). В телеграм боте мы получаем поток байтов BytesIO,
        # а мы хотим спрятать в этот метод всю работу с картинками, поэтому лучше принимать тут эти самые потоки
        # и потом уже приводить их к PIL, а потом и к тензору, который уже можно отдать модели.
        # Не забудьте перенести все трансофрмации, которые вы использовали при тренировке
        # Для этого будет удобно сохранить питоновский объект с ними в виде файла с помощью pickle,
        # а потом загрузить здесь.

        inputs = self.process_image(img_stream)
        self.model.eval()
        logit = self.model(inputs).cpu()

        # Обработка картинки сейчас производится в методе process image, а здесь мы должны уже применить нашу
        # модель и вернуть вектор предсказаний для нашей картинки

        # Для наглядности мы сначала переводим ее в тензор, а потом обратно
        return self.model.predict(self.process_image(img_stream))[0]

    # В predict используются некоторые внешние функции, их можно добавить как функции класса
    # Если понятно, что функция является служебной и снаружи использоваться не должна, то перед именем функции
    # принято ставить _ (выглядит это так: def _foo() )
    # ниже пример того, как переносить методы
    def process_image(self, img_stream):
        # используем PIL, чтобы получить картинку из потока и изменить размер
        image = PIL_Image.open(img_stream).resize((256, 256))
        val = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        val(image)
        return image

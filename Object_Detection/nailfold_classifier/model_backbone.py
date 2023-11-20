# coding:utf-8
import torch
import torchvision
from torch import nn


class Backbone(object):
    def __init__(self, in_channels=3, out_dimension=10, model_name="resnet50", pretrained=False):
        self.in_channels = in_channels
        self.out_dimension = out_dimension
        self.model_name = model_name
        self.pretrained = pretrained

    def build_model(self):
        if self.model_name.startswith("mobilenet"):
            self._build_mobilenet()
        if self.model_name.startswith("resnet"):
            self._build_resnet()
        if self.model_name.startswith("resnext"):
            self._build_resnext()
        if self.model_name.startswith("densenet"):
            self._build_densenet()
        if self.model_name.startswith("shufflenet"):
            self._build_shufflenet()
        if self.model_name.startswith("squeezenet"):
            self._build_squeezenet()
        if self.model_name.startswith("wide_resnet"):
            self._build_wide_resnet()
        return self.model, self.train_parameters, self.pretrained_parameters

    def _build_mobilenet(self):
        if self.model_name == "mobilenet_v2":
            self.model = torchvision.models.mobilenet_v2(pretrained=self.pretrained)
            self.model.classifier[1] = nn.Linear(in_features=1280, out_features=self.out_dimension)
            nn.init.kaiming_normal_(self.model.classifier[1].weight, mode="fan_in", nonlinearity="relu")
        if self.model_name == "mobilenet_v3_small":
            self.model = torchvision.models.mobilenet_v3_small(pretrained=self.pretrained)
            self.model.classifier[3] = nn.Linear(in_features=1024, out_features=self.out_dimension)
            nn.init.kaiming_normal_(self.model.classifier[3].weight, mode="fan_in", nonlinearity="relu")
        if self.model_name == "mobilenet_v3_large":
            self.model = torchvision.models.mobilenet_v3_large(pretrained=self.pretrained)
            self.model.classifier[3] = nn.Linear(in_features=1280, out_features=self.out_dimension)
            nn.init.kaiming_normal_(self.model.classifier[3].weight, mode="fan_in", nonlinearity="relu")

        if not self.pretrained:
            self.train_parameters = self.model.parameters()
            self.pretrained_parameters = []
        else:
            if self.model_name == "mobilenet_v2":
                self.train_parameters_id = list(map(id, self.model.classifier[1].parameters()))
                self.pretrained_parameters = filter(lambda p: id(p) not in self.train_parameters_id,
                                                    self.model.parameters())
                self.train_parameters = self.model.classifier[1].parameters()
            if self.model_name == "mobilenet_v3_small":
                self.train_parameters_id = list(map(id, self.model.classifier[3].parameters()))
                self.pretrained_parameters = filter(lambda p: id(p) not in self.train_parameters_id,
                                                    self.model.parameters())
                self.train_parameters = self.model.classifier[3].parameters()
            if self.model_name == "mobilenet_v3_large":
                self.train_parameters_id = list(map(id, self.model.classifier[3].parameters()))
                self.pretrained_parameters = filter(lambda p: id(p) not in self.train_parameters_id,
                                                    self.model.parameters())
                self.train_parameters = self.model.classifier[3].parameters()

    def _build_resnet(self):
        if self.model_name == "resnet18":
            self.model = torchvision.models.resnet18(pretrained=self.pretrained)
            self.model.fc = nn.Linear(in_features=512, out_features=self.out_dimension)
            nn.init.kaiming_normal_(self.model.fc.weight, mode="fan_in", nonlinearity="relu")
        if self.model_name == "resnet34":
            self.model = torchvision.models.resnet34(pretrained=self.pretrained)
            self.model.fc = nn.Linear(in_features=512, out_features=self.out_dimension)
            nn.init.kaiming_normal_(self.model.fc.weight, mode="fan_in", nonlinearity="relu")
        if self.model_name == "resnet50":
            self.model = torchvision.models.resnet50(pretrained=self.pretrained)
            self.model.fc = nn.Linear(in_features=2048, out_features=self.out_dimension)
            nn.init.kaiming_normal_(self.model.fc.weight, mode="fan_in", nonlinearity="relu")
        if self.model_name == "resnet101":
            self.model = torchvision.models.resnet101(pretrained=self.pretrained)
            self.model.fc = nn.Linear(in_features=2048, out_features=self.out_dimension)
            nn.init.kaiming_normal_(self.model.fc.weight, mode="fan_in", nonlinearity="relu")

        if not self.pretrained:
            self.train_parameters = self.model.parameters()
            self.pretrained_parameters = []
        else:
            self.train_parameters_id = list(map(id, self.model.fc.parameters()))
            self.pretrained_parameters = filter(lambda p: id(p) not in self.train_parameters_id,
                                                self.model.parameters())
            self.train_parameters = self.model.fc.parameters()

    def _build_resnext(self):
        if self.model_name == "resnext50_32x4d":
            self.model = torchvision.models.resnext50_32x4d(pretrained=self.pretrained)
            self.model.fc = nn.Linear(in_features=2048, out_features=self.out_dimension)
            nn.init.kaiming_normal_(self.model.fc.weight, mode="fan_in", nonlinearity="relu")
        if self.model_name == "resnext101_32x8d":
            self.model = torchvision.models.resnext101_32x8d(pretrained=self.pretrained)
            self.model.fc = nn.Linear(in_features=2048, out_features=self.out_dimension)
            nn.init.kaiming_normal_(self.model.fc.weight, mode="fan_in", nonlinearity="relu")

        if not self.pretrained:
            self.train_parameters = self.model.parameters()
            self.pretrained_parameters = []
        else:
            self.train_parameters_id = list(map(id, self.model.fc.parameters()))
            self.pretrained_parameters = filter(lambda p: id(p) not in self.train_parameters_id,
                                                self.model.parameters())
            self.train_parameters = self.model.fc.parameters()

    def _build_densenet(self):
        if self.model_name == "densenet121":
            self.model = torchvision.models.densenet121(pretrained=self.pretrained)
            self.model.classifier = nn.Linear(in_features=1024, out_features=self.out_dimension)
            nn.init.kaiming_normal_(self.model.classifier.weight, mode="fan_in", nonlinearity="relu")
        if self.model_name == "densenet161":
            self.model = torchvision.models.densenet161(pretrained=self.pretrained)
            self.model.classifier = nn.Linear(in_features=2208, out_features=self.out_dimension)
            nn.init.kaiming_normal_(self.model.classifier.weight, mode="fan_in", nonlinearity="relu")
        if self.model_name == "densenet169":
            self.model = torchvision.models.densenet169(pretrained=self.pretrained)
            self.model.classifier = nn.Linear(in_features=1664, out_features=self.out_dimension)
            nn.init.kaiming_normal_(self.model.classifier.weight, mode="fan_in", nonlinearity="relu")

        if not self.pretrained:
            self.train_parameters = self.model.parameters()
            self.pretrained_parameters = []
        else:
            self.train_parameters_id = list(map(id, self.model.classifier.parameters()))
            self.pretrained_parameters = filter(lambda p: id(p) not in self.train_parameters_id,
                                                self.model.parameters())
            self.train_parameters = self.model.classifier.parameters()

    def _build_shufflenet(self):
        if self.model_name == "shufflenet_v2_x0_5":
            self.model = torchvision.models.shufflenet_v2_x0_5(pretrained=self.pretrained)
            self.model.fc = nn.Linear(in_features=1024, out_features=self.out_dimension)
            nn.init.kaiming_normal_(self.model.fc.weight, mode="fan_in", nonlinearity="relu")
        if self.model_name == "shufflenet_v2_x1_0":
            self.model = torchvision.models.shufflenet_v2_x1_0(pretrained=self.pretrained)
            self.model.fc = nn.Linear(in_features=1024, out_features=self.out_dimension)
            nn.init.kaiming_normal_(self.model.fc.weight, mode="fan_in", nonlinearity="relu")
        if self.model_name == "shufflenet_v2_x1_5":
            self.model = torchvision.models.shufflenet_v2_x1_5(pretrained=self.pretrained)
            self.model.fc = nn.Linear(in_features=1024, out_features=self.out_dimension)
            nn.init.kaiming_normal_(self.model.fc.weight, mode="fan_in", nonlinearity="relu")
        if self.model_name == "shufflenet_v2_x2_0":
            self.model = torchvision.models.shufflenet_v2_x2_0(pretrained=self.pretrained)
            self.model.fc = nn.Linear(in_features=2048, out_features=self.out_dimension)
            nn.init.kaiming_normal_(self.model.fc.weight, mode="fan_in", nonlinearity="relu")

        if not self.pretrained:
            self.train_parameters = self.model.parameters()
            self.pretrained_parameters = []
        else:
            self.train_parameters_id = list(map(id, self.model.fc.parameters()))
            self.pretrained_parameters = filter(lambda p: id(p) not in self.train_parameters_id,
                                                self.model.parameters())
            self.train_parameters = self.model.fc.parameters()

    def _build_squeezenet(self):
        if self.model_name == "squeezenet1_0":
            self.model = torchvision.models.squeezenet1_0(pretrained=self.pretrained)
            self.model.classifier[1] = nn.Conv2d(in_channels=512, out_channels=self.out_dimension, kernel_size=(1, 1),
                                                 stride=(1, 1))
            nn.init.kaiming_normal_(self.model.classifier[1].weight, mode="fan_in", nonlinearity="relu")
        if self.model_name == "squeezenet1_1":
            self.model = torchvision.models.squeezenet1_1(pretrained=self.pretrained)
            self.model.classifier[1] = nn.Conv2d(in_channels=512, out_channels=self.out_dimension, kernel_size=(1, 1),
                                                 stride=(1, 1))
            nn.init.kaiming_normal_(self.model.classifier[1].weight, mode="fan_in", nonlinearity="relu")

        if not self.pretrained:
            self.train_parameters = self.model.parameters()
            self.pretrained_parameters = []
        else:
            self.train_parameters_id = list(map(id, self.model.classifier[1].parameters()))
            self.pretrained_parameters = filter(lambda p: id(p) not in self.train_parameters_id,
                                                self.model.parameters())
            self.train_parameters = self.model.classifier[1].parameters()

    def _build_wide_resnet(self):
        if self.model_name == "wide_resnet50_2":
            self.model = torchvision.models.wide_resnet50_2(pretrained=self.pretrained)
            self.model.fc = nn.Linear(in_features=2048, out_features=self.out_dimension)
            nn.init.kaiming_normal_(self.model.fc.weight, mode="fan_in", nonlinearity="relu")
        if self.model_name == "wide_resnet101_2":
            self.model = torchvision.models.wide_resnet101_2(pretrained=self.pretrained)
            self.model.fc = nn.Linear(in_features=2048, out_features=self.out_dimension)
            nn.init.kaiming_normal_(self.model.fc.weight, mode="fan_in", nonlinearity="relu")

        if not self.pretrained:
            self.train_parameters = self.model.parameters()
            self.pretrained_parameters = []
        else:
            self.train_parameters_id = list(map(id, self.model.fc.parameters()))
            self.pretrained_parameters = filter(lambda p: id(p) not in self.train_parameters_id,
                                                self.model.parameters())
            self.train_parameters = self.model.fc.parameters()


if __name__ == '__main__':
    model_names = [
        # "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large",
        # "resnet18", "resnet34", "resnet50", "resnet101",
        # "resnext50_32x4d", "resnext101_32x8d",
        # "densenet121", "densenet161", "densenet169",
        # "shufflenet_v2_x0_5", "shufflenet_v2_x1_0",
        # "squeezenet1_0", "squeezenet1_1",
        "wide_resnet50_2", "wide_resnet101_2"
    ]
    device = torch.device("cpu")
    data = torch.randn((2, 3, 224, 224), device=device)
    for model_name in model_names:
        print(model_name)
        backbone = Backbone(out_dimension=10, model_name=model_name, pretrained=False)
        model, train_params, pretrained_params = backbone.build_model()
        res = model(data)
        print(res.shape)

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, output_dim),
        )

    def forward(self, x):
        return self.layers(x)

class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

class AutoEncoder(nn.Module):
    def __init__(self, class_num, feature_dim, dim):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(dim, feature_dim)
        self.decoder = Decoder(feature_dim, dim)
        self.centroids = Parameter(torch.Tensor(class_num, feature_dim))     #  (class_num, feature_dim)
        init.xavier_normal_(self.centroids)

        self.concat_centr = torch.Tensor(class_num, feature_dim)
        init.xavier_normal_(self.concat_centr)

    def forward(self, x):
        enc_feature = self.encoder(x)
        dec_enf = self.decoder(enc_feature)
        distances = torch.sum(torch.pow(enc_feature.unsqueeze(1) - self.centroids, 2), 2)
        q = 1.0 / (1.0 + distances)
        q = (q.t() / torch.sum(q, 1)).t()
        return enc_feature, dec_enf, q



# MNIST-USPS and Fashion-MV
class CE(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(CE, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, output_dim)  # Assuming input image size is 32x32
        )

    def forward(self, x):
        return self.layers(x)


class CD(nn.Module):
    def __init__(self, input_dim, output_channels):
        super(CD, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, output_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid()  # Assuming output range is [0, 1]
        )

    def forward(self, x):
        return self.layers(x)

class CAE(nn.Module):
    def __init__(self, class_num, feature_dim):
        super(CAE, self).__init__()
        self.encoder = CE(1, feature_dim)
        self.decoder = CD(feature_dim, 1)
        self.centroids = Parameter(torch.Tensor(class_num, feature_dim))
        init.xavier_normal_(self.centroids)

        self.concat_centr = torch.Tensor(class_num, feature_dim)
        init.xavier_normal_(self.concat_centr)

    def forward(self, x):
        enc_feature = self.encoder(x)
        dec_enf = self.decoder(enc_feature)
        distances = torch.sum(torch.pow(enc_feature.unsqueeze(1) - self.centroids, 2), 2)
        q = 1.0 / (1.0 + distances)
        q = (q.t() / torch.sum(q, 1)).t()
        return enc_feature, dec_enf, q



class CAENet(nn.Module):
    def __init__(self, class_num, feature_dim):
        super(CAENet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, feature_dim)


        self.reverse_fc1 = nn.Linear(feature_dim, 50)
        self.reverse_fc2 = nn.Linear(50, 320)
        self.deconv2 = nn.ConvTranspose2d(20, 10, kernel_size=6, stride=6)  # Inverse of conv1
        self.deconv1 = nn.ConvTranspose2d(10, 1, kernel_size=5, stride=5, padding=2)

        self.centroids = Parameter(torch.Tensor(class_num, feature_dim))  # 大小为10*10

    def forward(self, x):
        # Original CNNMnist forward pass
        x1 = F.relu(F.max_pool2d(self.conv1(x), 2)) # torch.Size([128, 10, 12, 12])
        x1 = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x1)), 2))# torch.Size([128, 20, 4, 4])
        x1 = x1.view(-1, x1.shape[1] * x1.shape[2] * x1.shape[3]) # torch.Size([128, 320])
        x1 = F.relu(self.fc1(x1)) # torch.Size([128, 50])
        x1 = F.dropout(x1, training=self.training) # torch.Size([128, 50])
        x1 = self.fc2(x1) # torch.Size([128, 10])
        zs = x1

        # Reverse CNN forward pass
        x2 = self.reverse_fc1(x1) # torch.Size([128, 50])
        x2 = F.dropout(x2, training=self.training)
        x2 = F.relu(self.reverse_fc2(x2)) # 320
        x2 = x2.view(-1, 20, 4, 4)
        x2 = F.relu(F.max_pool2d(self.conv2_drop(self.deconv2(x2)), 2)) # torch.Size([128, 10, 12, 12])
        x2 = F.relu(F.max_pool2d(self.deconv1(x2), 2))  #  torch.Size([128, 1, 28, 28])
        xrs = x2


        init.uniform_(self.centroids)
        q = 1.0 / (1.0 + torch.sum(torch.pow(zs.unsqueeze(1) - self.centroids, 2), 2))
        q = (q.t() / torch.sum(q, 1)).t()
        return zs, xrs, q




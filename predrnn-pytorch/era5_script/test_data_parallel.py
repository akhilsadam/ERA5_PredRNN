import tensorflow as tf
tf.config.list_physical_devices('GPU')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Parameters and DataLoaders
input_size = 5
output_size = 2

batch_size = 30
data_size = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True)

class Model(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.mean = nn.Parameter(torch.tensor([0, 1.39538188e-04,  1.99499756e-04,  1.47195304e-04,
                                            3.02930121e-05, -1.19205533e-04, -3.21148090e-04, -5.80101568e-04,
                                            -8.30103159e-04, -1.01917370e-03, -9.90518653e-04, -1.02008264e-03,
                                            -9.83945028e-04, -9.25706082e-04, -8.45840217e-04, -7.55326795e-04,
                                            -6.86238213e-04, -6.25248607e-04, -5.60617360e-04, -4.71305105e-04,
                                            -3.53708443e-04, -2.18092631e-04, -3.34181112e-04, -1.63298334e-04]), requires_grad=False)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(), "Input device", input.get_device(), "self.mean device", self.mean.get_device(),
              "output size", output.size())

        return output
    
model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
devices = [i for i in range(3)]
print(devices)
model = nn.DataParallel(model, device_ids=devices).to(device)

for data in rand_loader:
    input = data.to(device)
    output = model(input)
    print("Outside: input size", input.size(),
          "output_size", output.size())
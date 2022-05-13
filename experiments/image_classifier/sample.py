from datasets import CompoundDataset
from functions import train, test
import models

import torch
from torch.utils.data import DataLoader

torch.manual_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_dataset = CompoundDataset(
                            "../../datasets/negev/articles_molecules/preprocess256/aug2",
                            "../../datasets/negev/not_molecules/preprocess256",
                            "test"
                          )

test_dataloader = DataLoader(test_dataset, batch_size=64)

model = models.VGG16()
model.load_state_dict(torch.load("saved_models/try.pt"))
model.to(device)

nb_errors = test(test_dataloader, model, device)

print("Test dataset size:", len(test_dataset))
print("Total errors predicting test dataset:", nb_errors)
print("Error percentage:", nb_errors/len(test_dataset)*100)
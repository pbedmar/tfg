import torch

results = torch.load("stored_results_synthetic_dataset/grid_search_results.pt")
print(results.shape)

model = 2
weight_initialization = 1
optimizer = 2

for element in results[model,weight_initialization,optimizer,:,3,0]:
    print("%.3f" % element.item())

print()
for element in results[model,weight_initialization,optimizer,:,3,1]:
    print("%.3f" % element.item())
import pandas as pd
import torch

# define path for csv file
#path = "/home/g037552/Bachelorarbeit/ba/output_old/2025-14-03_HSN-Model_Adadelta_LR0.03_Epochs501_dropout-off_sigmoid_on_predictedParameter.csv"
#path = "/home/g037552/Bachelorarbeit/ba/output_old/2025-12-03_HSN-Model_RMSprop_LR0.00003_Epochs501_dropout-off_sigmoid_on_predictedParameter.csv"
#path = "/home/g037552/Bachelorarbeit/ba/output_old/2025-15-03_FB-Model_RMSprop_LR0.0005_Epochs501_dropout-off_sigmoid_on_predictedParameter.csv"
path = "/home/g037552/Bachelorarbeit/ba/output_old/2025-29-03_FB-Model_Adadelta_LR9.0_Epochs501_dropout-off_sigmoid_on_MSE-loss_predictedParameter.csv"


#weighted MSEs
#path = "/home/g037552/Bachelorarbeit/ba/output_old/2025-19-03_FB-Model_RMSprop_LR0.0005_Epochs501_dropout-off_sigmoid_on_weighted-mse_predictedParameter.csv"
#path = "/home/g037552/Bachelorarbeit/ba/output_old/2025-15-03_FB-Model_RMSprop_LR0.0005_Epochs10_dropout-off_sigmoid_on_weighted-mse_predictedParameter.csv"
#path = "/home/g037552/Bachelorarbeit/ba/output_old/2025-21-03_FB-Model_Adadelta_LR9.0_Epochs501_dropout-off_sigmoid_on_weighted-mse_predictedParameter.csv"


# read csv file with pandas
data = pd.read_csv(path).to_numpy()
# store ground truth values
groundtruth = torch.from_numpy(data[:,:6])
# store predicted values
predictions = torch.from_numpy(data[:,6:])
# calculate error between ground truth and predicted values
error = (groundtruth - predictions)**2
# calculate MSE per line (axis=1)
mse_per_line = error.mean(axis=1)
# calculate total MSE (MSE per column (axis=0)), based on results from mse_per_line 
mse_total = mse_per_line.mean(axis=0)
# calculate total standard deviation, based on the resutls from mse_per_line
std_total = mse_per_line.std(axis=0).tolist()
# calculate MSE per column for all 6 error values individually
mse_values = error.mean(axis=0).sqrt().tolist()
# calculate standard deviation per column for all 6 error values indiviually
std_values = error.std(axis=0).tolist()

# list the 16 results with the maximum error and their indices
max_16_values = torch.topk(mse_per_line, 16, 0, True)
print(max_16_values)
# list the 16 results with the minimum error and their indices
min_16_values = torch.topk(mse_per_line, 16, 0, False)
print(min_16_values)

names = ["Gender     ", "Age        ", "Muscle     ", "Weight     ", "Height     ", "Propertions"]
dict(zip(names, mse_values))
print(dict(zip(names, mse_values)))
print("MSE (total)", mse_total)
print("Standard deviation (total):", std_total)
print("Standard deviation (per value)", std_values)




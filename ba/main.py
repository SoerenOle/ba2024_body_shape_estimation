import sys
import torch
import datetime
from torch import nn
from ba import MakeHumanDatasetFrontView
from ba import HSNet
from ba import FBNet
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import argparse
from pathlib import Path


class WeightedMSELoss(nn.Module):
    def forward(self, targets, inputs):
        device = ("cuda" if torch.cuda.is_available() else "cpu")
        #weights = torch.tensor([4.0,1.5,4.0,3.0,1.0,2.0], device=device)
        weights = torch.tensor([3.5,1.5,3.0,2.5,1.0,2.0], device=device)
        weighted_squared_error = ((inputs - targets)**2 ) * weights
        mse_loss = weighted_squared_error.mean()
        return mse_loss


def main():

    ########
    # TODO #
    ########
    '''
    #optimizer_choices = 
    [x for x in vars(torch.optim).keys() if x[0].isupper() and x != "Optimizer"]
    '''
    
    ########
    # TODO #
    ########
    '''
    # Paths:
    PROJECT_ROOT_PATH = Path(__file__).parents[1]
    WORKSPACE_PATH = Path("/works/ws-temp/workspace_name")
    #DATA_BASE_PATH = PROJECT_ROOT_PATH / "data"
    DATA_BASE_PATH = WORKSPACE_PATH / "data"
    #LOGS_BASE_PATH = PROJECT_ROOT_PATH / "logs"
    LOGS_BASE_PATH = WORKSPACE_PATH / "logs"
    '''

    # Definiere argparse
    parser = argparse.ArgumentParser()                                          # Create an ArgumentParser object
    parser.add_argument("--lr", type=float, default=1e-3)                       # Define command-line argument for learning rate
    parser.add_argument("--optimizer", 
                        type=str, 
                        choices=["SGD", "Adam", "Adadelta", "RMSprop"], 
                        default="Adadelta")                                      # Define command-line argument for optimizer options
    args = parser.parse_args()                                                  # Parse the arguments
    
    # Log-Datei erstellen
    learning_rate = "_LR=" + str(args.lr)
    optimizer_name =  "_" + args.optimizer
    network_name = "_FB_model"
    dropout_setting = "_dropout_off"
    epoch_numbers = "_epochs=501"
    sigmoid_setting = "_sigmoid_on"
    loss_fn = "mse"
    date = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S.%f")[:-3]
    #writer = SummaryWriter("/home/g037552/Bachelorarbeit/logs/" + date + learning_rate + optimizer_name)
    writer = SummaryWriter(Path(__file__).parents[1] / "logs" / (date + learning_rate + optimizer_name + network_name + dropout_setting + epoch_numbers + sigmoid_setting + loss_fn))
    #print(Path(__file__).parents[1] / "logs" / (date + learning_rate + optimizer_name))


    # device configuration
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    # data paths
    #path_parameter = "../Data/table_combinations/combinations_table.csv"
    #path_image = "../Bachelorarbeit_Data/Blender Results/results_top-view_w-o_breasts-param/results_blender"
    #path_parameter = "Data/table_combinations/combinations_table.csv"
    path_image = "/home/g037552/Bachelorarbeit_Data/Blender Results/results_top-view_w-o_breasts-param/results_blender"
    path_parameter = "/home/g037552/Bachelorarbeit/Data/table_combinations/combinations_table.csv"

    # hyper parameters
    num_epochs=501
    num_classes=6
    batch_size=96
    learning_rate = 5e-7 #RMSprop

    # initialize training_dataset
    dataset = MakeHumanDatasetFrontView(path_parameter, path_image)
    #dataset = torch.utils.data.Subset(dataset, list(range(0, 10000)))
    #dataset = torch.utils.data.Subset(dataset, dataset[1200])
    
    #training_data = dataset
    training_data, validation_data, test_data_synth = torch.utils.data.random_split(dataset, [8400, 2100, 2625])
    #training_data, validation_data = torch.utils.data.random_split(dataset, [10000, 3125])
    #training_data, validation_data = torch.utils.data.random_split(dataset, [8, 2])

    training_dataloader = DataLoader(training_data, batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_data, batch_size, shuffle=False)
    test_dataloader_synth = DataLoader(test_data_synth, batch_size, shuffle=False)

    # initialize model
    #model = HSNet(num_classes).to(device)
    model = FBNet(num_classes).to(device)

    # initialize the loss function
    loss_fn = nn.MSELoss()
    #loss_fn = WeightedMSELoss()

    # optimizer mit argparse
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.lr)
    
    #optimizer
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

    #total_step=5
    total_step=len(training_dataloader)
    print(total_step)

    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Decay LR by 0.1 every 10 epochs   
    
    # keine gradienten und optimierung
    #writer.add_scalar("Loss/val", avg_loss, 0)
    
    # Validation Loop
    model.eval()
    for i, (images, vectors) in enumerate(validation_dataloader):
        
        # Move data to the appropriate device
        images, vectors = images.to(device), vectors.to(device)
        # Forward pass: Compute predictions
        predictions = model(images)
        # Compute the loss
        loss = loss_fn(predictions, vectors)     
        # Accumulate the loss for this batch
        #epoch_loss += loss.item()
        # Log training loss to TensorBoard
        #writer.add_scalar("Loss/val", loss.item(), epoch*len(validation_dataloader)+i) 
        writer.add_scalar("Loss/val", loss.item(), 0) 


    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Training Loop
        model.train()  # Set the model to training mode
        
        # Initialisierung "Best Average Loss für Validation"
        best_avg_loss_eval = float("inf") 
        best_weights = model.state_dict()

        epoch_loss = 0.0  # To track the average loss in this epoch
        for i, (images, vectors) in enumerate(training_dataloader):
            # Move data to the appropriate device
            images, vectors = images.to(device), vectors.to(device)
            # Forward pass: Compute predictions
            predictions = model(images)
            # Compute the loss
            loss = loss_fn(predictions, vectors)
            # Backward pass: Compute gradients
            optimizer.zero_grad() 
            loss.backward()           
            # Update model weights
            optimizer.step()  
            # Accumulate the loss for this batch
            epoch_loss += loss.item()
            # Log training loss to TensorBoard
            writer.add_scalar("Loss/train", loss.item(), epoch*len(training_dataloader)+i) 
            
            # Print progress every 10 batches
            if (i + 1) % 10 == 0 or (i + 1) == len(training_dataloader):
                print(f"  Batch {i+1}/{total_step}, Loss: {loss.item():.4f}")
        
        # Calculate average loss for the epoch
        avg_loss = epoch_loss / len(training_dataloader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")
        
        #scheduler.step()  # Call at the end of each epoch
            
        model.eval()
        epoch_loss = 0.0 
        for i, (images, vectors) in enumerate(validation_dataloader):
            # Move data to the appropriate device
            images, vectors = images.to(device), vectors.to(device)
            # Forward pass: Compute predictions
            predictions = model(images)
            # Compute the loss
            loss = loss_fn(predictions, vectors) 
            # Accumulate the loss for this batch
            epoch_loss += loss.item()

        avg_loss_eval_validation = epoch_loss / len(validation_dataloader)
        writer.add_scalar("Loss/val", avg_loss_eval_validation, epoch+1)
    
        # to find the best loss
        if avg_loss_eval_validation < best_avg_loss_eval:
            best_avg_loss_eval = avg_loss_eval_validation
            best_weights = model.state_dict()


    # Close TensorBoard writer
    #writer.close()  

    # TEST-Schleife 
    epoch_loss = 0.0 
    with open("/home/g037552/Bachelorarbeit/output/2025-29-03_FB-Model_Adadelta_LR9.0_Epochs501_dropout-off_sigmoid_on_MSE-loss_predictedParameter.csv", "w") as csvfile:
        for i, (images, vectors) in enumerate(test_dataloader_synth): #hier testloader
            # Move data to the appropriate device
            images, vectors = images.to(device), vectors.to(device)
            # Forward pass: Compute predictions
            predictions = model(images)
            # Compute the loss
            loss = loss_fn(predictions, vectors)
            # Accumulate the loss for this batch
            epoch_loss += loss.item()
            for j in range (vectors.shape[0]):
                csvfile.write(str(vectors[j,0].squeeze().item())+",")
                csvfile.write(str(vectors[j,1].squeeze().item())+",")
                csvfile.write(str(vectors[j,2].squeeze().item())+",")
                csvfile.write(str(vectors[j,3].squeeze().item())+",")
                csvfile.write(str(vectors[j,4].squeeze().item())+",")
                csvfile.write(str(vectors[j,5].squeeze().item())+",")
                csvfile.write(str(predictions[j,0].squeeze().item())+",")
                csvfile.write(str(predictions[j,1].squeeze().item())+",")
                csvfile.write(str(predictions[j,2].squeeze().item())+",")
                csvfile.write(str(predictions[j,3].squeeze().item())+",")
                csvfile.write(str(predictions[j,4].squeeze().item())+",")
                csvfile.write(str(predictions[j,5].squeeze().item())+"\n")               
    # Log training loss to TensorBoard
    avg_loss_eval = epoch_loss / len(test_dataloader_synth)
    writer.add_scalar("Loss/test", avg_loss, epoch+1)


    # To SAVE turn on 
    torch.save(best_weights, f"/home/g037552/Bachelorarbeit/saved_model/model_epoch_{epoch+1}.pth")

    # Close TensorBoard writer
    writer.close()
   
    # Open Tensorboard
    #tensorboard --logdir logs
    #tensorboard --samples_per_plugin=scalars=7000 --logdir logs  

if __name__ == '__main__':
    sys.exit(main())
# Problem Set

In this problem set, we will implement Low-Rank Adaptation(LoRA) finetuning from scratch on the MNIST dataset. 

## A. Split Dataset
To simulate the domain shift, we will split the MNIST dataset into two domains: source domain (first 5 classes) and target domain (last 5 classes).

* Load the dataset, convert it to tensor use `torchvision.transforms.ToTensor()`, and normalize the data with the mean 0.1307 and std 0.3081, which are the global mean and standard deviation of the MNIST dataset. You may use `torchvision.transforms.Normalize()` for normalization.
* Split the dataset into two domains: source domain (first 5 classes) and target domain (last 5 classes). You may use `torch.utils.data.sampler.SubsetRandomSampler` for splitting the dataset.
* For each domain, create three dataloaders using `torch.utils.data.DataLoader`: one for training, one for validation, and one for testing. Use a batch size of 64 for all dataloaders.

## B. Pre-Train a Model
We will use a simple Multi-Layer Perceptron(MLP) as the base model. The MLP will have 2 hidden layers with 128 units and ReLU activation functions. The output layer will have 10 units (number of classes in MNIST).

* Implement the MLP model that consists of 2 hidden layers with 64 units. Each hidden layer is followed by a ReLU activation function and a dropout layer with a dropout rate of 0.1. The output layer should have 5 units with softmax activation to output the class probabilities.

* Train the model on the source domain using the source training dataloader. Use the AdamW optimizer with a learning rate of 1e-3 and 10 epochs. Use nll_loss as the loss function. For each epoch, calculate the validation loss and accuracy on the source test dataloader.

* Save the model with the best validation accuracy on the source validation dataloader. You may use `torch.save()` to save the model.

* Visualize the training loss, validation loss, and validation accuracy. 

* Print test loss, and test accuracy.

* Visualize the number of learnable parameters in the model, which can be calculated using the `model.parameters()` function. This will be used later to compare the number of parameters in the base model and the adapted model during the training process.

## C. Fully Fine-Tune the Model
We will now fully fine-tune the pre-trained model on the target domain using all of the parameters in the base model.

* Load the pre-trained model from the saved file.

* Train the model on the target domain using the target training dataloader. Use the AdamW optimizer with a learning rate of 1e-3 and 10 epochs. Use nll_loss as the loss function. For each epoch, calculate the validation loss and accuracy on the target test dataloader.

* Print the test loss and test accuracy.

* Visualize the number of learnable parameters in the model.

## D. Adapt the Model
We will now adapt the pre-trained model on the target domain using the LoRA algorithm. The LoRA algorithm consists of two steps: adaptation and adaptation regularization.

* Construct a new model class by inheriting all the layers from the pre-trained model. In addition, for each layer, add a new LoRA layer that consist of matrix A and matrix B. Initialize matrix A and matrix B with the identity matrix. Use the formula in the LoRA paper to calculate the output of the LoRA layer. The new class should have 2 hyper-parameters: rank and alpha. The rank hyper-parameter specifies the rank of the low-rank approximation, and the alpha hyper-parameter, default as 1, specifies the weight of the adaptation regularization term.

* Similar to the C part, train the adapted model on the target domain using the target training dataloader. Use the AdamW optimizer with a learning rate of 1e-3 and 10 epochs. Use nll_loss as the loss function. For each epoch, calculate the validation loss and accuracy on the target test dataloader. The only difference is that you should use [8, 16] as the rank hyper-parameter. 

* For each rank, print the test loss and test accuracy.

* For each rank, visualize the number of learnable parameters in the model

## E. Analysis
In this part, we will analyze the results of the experiments.

* Compare the test accuracy of the pre-trained model, the fully fine-tuned model, and the adapted model.

* Compare the number of learnable parameters in the pre-trained model, the fully fine-tuned model, and the adapted model.
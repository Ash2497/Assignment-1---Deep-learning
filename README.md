# Assignment-1---Deep-learning
Code repository for assignment 1 of the Deep Learning course.
The code contains functions to predict the classes from the fashion MNIST dataset, and subsequently, the MNIST dataset. Standard back propogation algorithms, namely, 1) Stochastic and batch Gradient descent, 2) Monentum based gradient descent, 3) Nesterov accelerated gradient descent, 4) RMSprop 5) ADAM and 6) NADAM algorithms have been incroporated. The user can call each of these functions at a time to train the model. Further details are provided while discussing the backpropagation section.

FORWARD AND BACKWARD PROPGATION
The input size of each data set (MNIST and fashion-MNIST) is a an image containing 28x28 pixels (784 in total). The training set contains 60000 examples, leading to matrix with dimensions 60000x28x28. Each example is reconstructed by reshaping the image to a row vector conatining 784 pixels. The resulting matrix is therefore of size 60000x784. The matrix is then shuffled and split into batches of size determined by the user. The true classifications of the training data are then copied into a vector which is essentially the one hot vector of the classification. The function  batch_split(batch_size,X_train,Y_train,X_test,Y_test) shuffles and splits the dataset. 

Two methods have been used to initalised the weights and biases, the Xavier initialisation ('xv') and the random initalisation ('rand'). The function initialize_params(no_hidden,size_hidden,spec) is used to initalise the weights. The parameter 'spec' here refers to the type of initialisation.

The rest of the hyperparameters, such as the size of the hidden layers, number of hidden layers and the activation function can be specified by the user. Then, the forward_prop(x,w,b,act_func,no_hidden) function is used to perform the forward propagation. 

After obtaining the activation and pre-activation units from the forward propagation. The gradient of the weights are then obtained by invoking the backward_prop(y,y_pred,w,b,h,a,a_f,no_hidden,lam,act_func,loss_func) function. This function returns the gradients which are then used to train the weights as per the described learning algorithm. 


TRAINING THE MODEL
The function train_model(no_hidden,size_hidden,bs,max_iterations,learning_rate,learn_algo,lam,spec,act_func,loss_func) is used to train the model. It calls on the functions to perform forward propogation, backward propogation and the weights update based on the prescribed learning algorithm. After each epoch, the accuracy and the loss on the training and validation set is computed using the test_model(w,b,x_test,y_test,y_test_enc,act_func,no_hidden,loss_func,lam) function. While one can call the function seperately to test the model for specific hyperparameter values, we have used wandb to perform sweeps and identify optimal hyperparameters, as stated in the assignment. The method of sweep is specified and the parameter values/range for each hyperparameter is specified in the parameters_dict dictionary. WANDB performs a sweep and logs the specified results (accuracy and loss in the training and validation sets) for each sweep. The sweep is performed by calling the train2(config=None), which passes on the parameters from the dictonary to the train_model function for that particular run. On analysizing the results, one can identify the optimum set of parameters which can be further tuned to optimize the perfomance of the model. 

TESTING THE MODEL-CONFUSION MATRIX
On identifying the optimum paramters the function test_model_cf(w,b,x_test,y_test,y_test_enc,act_func,no_hidden,loss_func,lam) is called to display the confusion matrix. The matrix is obtained by the predictions on the test data, using the weights trained by suitable hyperparamters. 

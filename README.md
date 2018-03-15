# A simple implementation of the backpropagation neural networks with gradient descend 

**Forward Propagation**  
Any kinds of backpropagation neural networks start from forward propagation. Let's try to implement a neural network with 2 hidden layers.  
In the above diagram, the forward propagation process is very intuitive.  
In this example, we are trying to train a N.N. model that can correctly identify the "cat" or "non-cat" images. 
In other words, it is a binary image classification task. So, we can say, there should have one, and only one neuron, in the output
layer, which indicates "1 (cat)" or "0 (non-cat)", respectively.  
What about the input layer? Actually, the number of neurons in the input layer is basically equal to the size of my input data's 
feature space. In this example, our training data is 209 images (each has 64 X 64 pixels and RGB layers), or we can say, if we admit the 
essential of the linear transformation (<a href="https://www.codecogs.com/eqnedit.php?latex=y&space;=&space;wx&space;&plus;&space;b" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y&space;=&space;wx&space;&plus;&space;b" title="y = wx + b" /></a>) 
between two layers (of course, involved all the neurons if they are fully connected) is the calculations of matrices
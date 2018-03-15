# A simple implementation of the backpropagation neural networks with gradient descend 

##**Background and N.N. Setup**  
In this example, we are trying to train a N.N. model that can correctly identify the "cat" or "non-cat" images. 
In other words, it is a binary image classification task. So, we can say, there should have one, and only one neuron, in the output
layer, which indicates "1 (cat)" or "0 (non-cat)", respectively.  
What about the input layer? Actually, the number of neurons in the input layer is basically equal to the size of my input data's 
feature space. In this example, our training data is 209 images (each has 64 X 64 pixels and RGB layers), or we can say, if we admit the 
essential of the linear transformation (<a href="https://www.codecogs.com/eqnedit.php?latex=y&space;=&space;wx&space;&plus;&space;b" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y&space;=&space;wx&space;&plus;&space;b" title="y = wx + b" /></a>) 
between two layers (of course, involved all the neurons if they are fully connected) is the calculations of matrices. We can regarding our
input as a tensor (size = 209 X 64 X 64 X 3, each element/line is a image, each image is a 64 X 64 matrix, each element in the matrix is a 
3-dimensional vector (R, G, B), and each dimension of the vector is a color value ranges 0 ~ 255, ). Here, we would not use any advanced
techniques (e.g. convolution) to transfer the tensor to a matrix, to make it calculable. We will process the elements by simply "flatten"
all the 3-dimensional vectors (from 64 X 64 X 3 to 1 X 12288). If we express the matrices with numpy array, in other words, we just 
simply get rid of all the bracket.  
<a href="https://www.codecogs.com/eqnedit.php?latex=([[1,2,3],&space;[4,5,6],&space;...&space;,[62,63,64]],&space;...&space;,[[1,2,3],&space;[4,5,6],&space;...&space;,[62,63,64]])&space;\rightarrow&space;\rightarrow&space;(1,2,3,4,5,&space;...&space;,62,&space;63,&space;64,&space;1,&space;2,&space;3,&space;...,&space;62,&space;63,&space;64)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?([[1,2,3],&space;[4,5,6],&space;...&space;,[62,63,64]],&space;...&space;,[[1,2,3],&space;[4,5,6],&space;...&space;,[62,63,64]])&space;\rightarrow&space;\rightarrow&space;(1,2,3,4,5,&space;...&space;,62,&space;63,&space;64,&space;1,&space;2,&space;3,&space;...,&space;62,&space;63,&space;64)" title="([[1,2,3], [4,5,6], ... ,[62,63,64]], ... ,[[1,2,3], [4,5,6], ... ,[62,63,64]]) \rightarrow \rightarrow (1,2,3,4,5, ... ,62, 63, 64, 1, 2, 3, ..., 62, 63, 64)" /></a>     

##**Forward Propagation**
Any kinds of backpropagation neural networks start from forward propagation. Let's try to implement a neural network with 2 hidden layers.  
In the above diagram, the forward propagation process is very intuitive.
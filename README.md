# A simple implementation of the backpropagation neural networks with gradient descend 

## **Background and N.N. Setup**  
In this example, we are trying to train a N.N. model that can correctly identify the "cat" or "non-cat" images. 
In other words, it is a binary image classification task. So, we can say, there should have one, and only one neuron, in the output
layer, which indicates "1 (cat)" or "0 (non-cat)", respectively.  
What about the input layer? Actually, the number of neurons in the input layer is basically equal to the size of my input data's 
feature space. In this example, our training data is 209 images (each has 64 X 64 pixels and RGB layers), or we can say, if we admit the 
essential of the linear transformation (<a href="https://www.codecogs.com/eqnedit.php?latex=y&space;=&space;wx&space;&plus;&space;b" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y&space;=&space;wx&space;&plus;&space;b" title="y = wx + b" /></a>) 
between two layers (of course, involved all the neurons if they are fully connected) is the calculations of matrices. We can regarding our
input as a tensor (size = 209 X 64 X 64 X 3, each element/line is a image, each image is a 64 X 64 matrix, each element in the matrix is a 
3-dimensional vector (R, G, B), and each dimension of the vector is a color value ranges 0 ~ 255). Here, we would not use any advanced
techniques (e.g. convolution) to transfer the tensor to a matrix, to make it calculable. We will process the elements by simply "flatten"
all the 3-dimensional vectors (from 64 X 64 X 3 to 1 X 12288). If we express the matrices with numpy array, in other words, we just 
simply get rid of all the bracket:  
<a href="https://www.codecogs.com/eqnedit.php?latex=([[1,2,3],&space;[4,5,6],&space;...&space;,[62,63,64]],&space;...&space;,[[1,2,3],&space;[4,5,6],&space;...&space;,[62,63,64]])&space;\rightarrow&space;\rightarrow&space;(1,2,3,4,5,&space;...&space;,62,&space;63,&space;64,&space;1,&space;2,&space;3,&space;...,&space;62,&space;63,&space;64)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?([[1,2,3],&space;[4,5,6],&space;...&space;,[62,63,64]],&space;...&space;,[[1,2,3],&space;[4,5,6],&space;...&space;,[62,63,64]])&space;\rightarrow&space;\rightarrow&space;(1,2,3,4,5,&space;...&space;,62,&space;63,&space;64,&space;1,&space;2,&space;3,&space;...,&space;62,&space;63,&space;64)" title="([[1,2,3], [4,5,6], ... ,[62,63,64]], ... ,[[1,2,3], [4,5,6], ... ,[62,63,64]]) \rightarrow \rightarrow (1,2,3,4,5, ... ,62, 63, 64, 1, 2, 3, ..., 62, 63, 64)" /></a>  
Now, our input is a 209 X 12288 matrix, correspondingly, our input layer should have 12288 neurons.
## **Forward Propagation**  
Any kinds of backpropagation neural networks start from forward propagation. Let's try to implement a neural network with 2 hidden layers.  
[image_forward]  
In the above diagram, the forward propagation process is very intuitive. Let's use the propagation process between input layer and hidden
layer 1 as an example, then every step will be the same, all the way to the output layer.  
<a href="https://www.codecogs.com/eqnedit.php?latex=a_{H_{1}}&space;=&space;W_{input\rightarrow&space;H_{1}}&space;\cdot&space;a_{input}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a_{H_{1}}&space;=&space;W_{input\rightarrow&space;H_{1}}&space;\cdot&space;a_{input}" title="a_{H_{1}} = W_{input\rightarrow H_{1}} \cdot a_{H_{1}}" /></a> (we don't add the bias here for continent)  
<a href="https://www.codecogs.com/eqnedit.php?latex=h_{H_1}&space;=&space;g&space;(a_{H_{1}})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?h_{H_1}&space;=&space;g&space;(a_{H_{1}})" title="h_{H_1} = g (a_{H_{1}})" /></a>  
At the end, <a href="https://www.codecogs.com/eqnedit.php?latex=h_{output}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?h_{output}" title="h_{output}" /></a> is the output, shaped as a 209 X 1 vector, indicating the predicted label for each image.  
## **Prediction Error and Cost Function**  
After the first time forward propagation, we get the predicted output(label) <a href="https://www.codecogs.com/eqnedit.php?latex=h_{output}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?h_{output}" title="h_{output}" /></a>, which you can imagine it will performs
as bad as random guess the label, because the weights between layers are randomly initialized. Therefore, our input data dot product by
 these random weights will still be random. The only contribution is, after the calculations between the matrices, the 12288 non-sense-like 
 features now is represented as a value, which kind-of indicates how much the image probably "looks like" a cat or not.  
Before we start the backward propagation to modify the non-sense weights to make-sense weights, we would like to know, how much we are currently
far away from making the prefect prediction, with 100% accuracy. 
<a href="https://www.codecogs.com/eqnedit.php?latex=Error&space;=&space;\frac{1}{2}&space;\sum&space;(label&space;-&space;h_{output})^{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Error&space;=&space;\frac{1}{2}&space;\sum&space;(label&space;-&space;h_{output})^{2}" title="Error = \frac{1}{2} \sum (label - h_{output})^{2}" /></a>  
_(Observing the above equation (cost function), a question come to my mind: why 1/2 of the L2 distance?_  
_The answer is for beauty.)_  
## **Gradient Descend**  
Now, let's stand one step back to speculate what we have on hand now. We have our input, a big matrix. We have the operation (basically just
linear transformation and activation function, nothing else) between layers. We have our prediction error. In this case, can we regard our
whole forward propagation process as this:  
<a href="https://www.codecogs.com/eqnedit.php?latex=Error&space;=&space;f_{input\rightarrow&space;H_{1}}(f_{H_{1}\rightarrow&space;H_{2}}(f_{H_{2}\rightarrow&space;output}&space;(Input)))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Error&space;=&space;f_{input\rightarrow&space;H_{1}}(f_{H_{1}\rightarrow&space;H_{2}}(f_{H_{2}\rightarrow&space;output}&space;(Input)))" title="Error = f_{input\rightarrow H_{1}}(f_{H_{1}\rightarrow H_{2}}(f_{H_{2}\rightarrow output} (Input)))" /></a>  
Therefore, what we can do to minimize the error?  
As we all know, **the partial derivative (or gradient) on certain variable reflects how sensitive the output to the variation of the certain variable.**
Then, the P.D. of the above equation is:  
[]  

reflects how  Can we directly do the derivative to the above P.D.E. and find the solution <a href="https://www.codecogs.com/eqnedit.php?latex=W_{input\rightarrow&space;H1},&space;W_{H1\rightarrow&space;H2},&space;W_{H2\rightarrow&space;output}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?W_{input\rightarrow&space;H1},&space;W_{H1\rightarrow&space;H2},&space;W_{H2\rightarrow&space;output}" title="W_{input\rightarrow H1}, W_{H1\rightarrow H2}, W_{H2\rightarrow output}" /></a>
(since we cannot change our input matrix)?  
Regardless of the computational difficulty, the answer is still **NO**. Because ever though each step we only do the linearly calculation,  
the process is nested, not parallelled. One step derivative is easy to fall into the pitfall of local minimization.   

   
## **Backward propagation**  
AAA       

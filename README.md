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
![image](http://github.com/YIHE1992/Back-Propagation-N.N.-/raw/master/forward.jpg)  
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
_(When observing the above equation (cost function), a question come to my mind: why 1/2 of the L2 distance?_  
_The answer is for beauty.)_  
## **Gradient Descend**  
Now, let's stand one step back to speculate what we have on hand now. We have our input, a big matrix. We have the operation (basically just
linear transformation and activation function, nothing else) between layers. We have our prediction error. In this case, can we regard our
whole forward propagation process as this:  
<a href="https://www.codecogs.com/eqnedit.php?latex=Error&space;=&space;f_{H_{2}\rightarrow&space;output}(f_{H_{1}\rightarrow&space;H_{2}}(f_{input\rightarrow&space;H_{1}}(Input)))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Error&space;=&space;f_{H_{2}\rightarrow&space;output}(f_{H_{1}\rightarrow&space;H_{2}}(f_{input\rightarrow&space;H_{1}}(Input)))" title="Error = f_{H_{2}\rightarrow output}(f_{H_{1}\rightarrow H_{2}}(f_{input\rightarrow H_{1}}(Input)))" /></a>  
Therefore, what we can do to minimize the error?  
As we all know, **the partial derivative (or gradient) on certain variable reflects how sensitive the output to the variation of the certain variable.**  
Can we directly do the partial derivative to the above equation and find the solution <a href="https://www.codecogs.com/eqnedit.php?latex=W_{input\rightarrow&space;H1},&space;W_{H1\rightarrow&space;H2},&space;W_{H2\rightarrow&space;output}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?W_{input\rightarrow&space;H1},&space;W_{H1\rightarrow&space;H2},&space;W_{H2\rightarrow&space;output}" title="W_{input\rightarrow H1}, W_{H1\rightarrow H2}, W_{H2\rightarrow output}" /></a>
(since we cannot change our input matrix)? **Yes**, but the normal method to solve the P.D.E. is easy to fall into the pitfall of local minimization.  
To avoid the pitfall, we will do the P.D. layer by layer by using chain rule, with the direction of backward propagation.       
## **Backward propagation**  
### **B.P. step 1 <a href="https://www.codecogs.com/eqnedit.php?latex=output\rightarrow&space;H_{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?output\rightarrow&space;H_{2}" title="output\rightarrow H_{2}" /></a>:**  
<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;Error}{\partial&space;W_{H_{2}\rightarrow&space;output}}=\frac{\partial&space;Error}{\partial&space;a_{output}}\cdot&space;\frac{\partial&space;a_{output}}{\partial&space;W_{H_{2}\rightarrow&space;output}}\\&space;\mapsto&space;\left\{\begin{matrix}\frac{\partial&space;Error}{\partial&space;a_{output}}=\frac{1}{2}\frac{\partial&space;\sum&space;(label-h_{output})^{2}}{\partial&space;a_{output}}=-{g}'(a_{output})(label-h_{output})\\&space;\frac{\partial&space;a_{output}}{\partial&space;W_{H_{2}\rightarrow&space;output}}=h_{H_{2}},a_{output}=W_{H_{2}\rightarrow&space;output}\cdot&space;h_{H_{2}}\\&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;Error}{\partial&space;W_{H_{2}\rightarrow&space;output}}=\frac{\partial&space;Error}{\partial&space;a_{output}}\cdot&space;\frac{\partial&space;a_{output}}{\partial&space;W_{H_{2}\rightarrow&space;output}}\\&space;\mapsto&space;\left\{\begin{matrix}\frac{\partial&space;Error}{\partial&space;a_{output}}=\frac{1}{2}\frac{\partial&space;\sum&space;(label-h_{output})^{2}}{\partial&space;a_{output}}=-{g}'(a_{output})(label-h_{output})\\&space;\frac{\partial&space;a_{output}}{\partial&space;W_{H_{2}\rightarrow&space;output}}=h_{H_{2}},a_{output}=W_{H_{2}\rightarrow&space;output}\cdot&space;h_{H_{2}}\\&space;\end{matrix}\right." title="\frac{\partial Error}{\partial W_{H_{2}\rightarrow output}}=\frac{\partial Error}{\partial a_{output}}\cdot \frac{\partial a_{output}}{\partial W_{H_{2}\rightarrow output}}\\ \mapsto \left\{\begin{matrix}\frac{\partial Error}{\partial a_{output}}=\frac{1}{2}\frac{\partial \sum (label-h_{output})^{2}}{\partial a_{output}}=-{g}'(a_{output})(label-h_{output})\\ \frac{\partial a_{output}}{\partial W_{H_{2}\rightarrow output}}=h_{H_{2}},a_{output}=W_{H_{2}\rightarrow output}\cdot h_{H_{2}}\\ \end{matrix}\right." /></a>  
In this equation set, <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;Error}{\partial&space;W_{H_{2}\rightarrow&space;output}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;Error}{\partial&space;W_{H_{2}\rightarrow&space;output}}" title="\frac{\partial Error}{\partial W_{H_{2}\rightarrow output}}" /></a> 
is called the gradient of <a href="https://www.codecogs.com/eqnedit.php?latex=W_{H_{2}\rightarrow&space;output}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?W_{H_{2}\rightarrow&space;output}" title="W_{H_{2}\rightarrow output}" /></a>. 
Through the function of <a href="https://www.codecogs.com/eqnedit.php?latex=\Delta&space;W_{H_{2}\rightarrow&space;output}&space;=&space;-l.r.&space;\times&space;\frac{\partial&space;Error}{\partial&space;W_{H_{2}\rightarrow&space;output}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Delta&space;W_{H_{2}\rightarrow&space;output}&space;=&space;-l.r.&space;\times&space;\frac{\partial&space;Error}{\partial&space;W_{H_{2}\rightarrow&space;output}}" title="\Delta W_{H_{2}\rightarrow output} = -l.r. \times \frac{\partial Error}{\partial W_{H_{2}\rightarrow output}}" /></a>, 
we can clearly know how much adjust we need to add on our current <a href="https://www.codecogs.com/eqnedit.php?latex=W_{H_{2}\rightarrow&space;output}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?W_{H_{2}\rightarrow&space;output}" title="W_{H_{2}\rightarrow output}" /></a>.  
### **B.P. step 2 <a href="https://www.codecogs.com/eqnedit.php?latex=H_{2}\rightarrow&space;H_{1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?H_{2}\rightarrow&space;H_{1}" title="H_{2}\rightarrow H_{1}" /></a>**  
Between the hidden layers, the equation set will be a little bit different:  
<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;Error}{\partial&space;input\rightarrow&space;H_{1}}=\frac{\partial&space;Error}{\partial&space;a_{H_{1}}}\cdot&space;\frac{\partial&space;a_{H_{1}}}{\partial&space;W_{input}\rightarrow&space;H_{1}}\\&space;\mapsto&space;\left\{\begin{matrix}\frac{\partial&space;Error}{\partial&space;a_{H_{1}}}=gradient(a_{H_{2}})\cdot\frac{\partial&space;a_{H_{2}}}{\partial&space;a_{H_{1}}}\\&space;\frac{\partial&space;a_{H_{1}}}{\partial&space;W_{input\rightarrow&space;H_{1}}}=a_{input}&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;Error}{\partial&space;input\rightarrow&space;H_{1}}=\frac{\partial&space;Error}{\partial&space;a_{H_{1}}}\cdot&space;\frac{\partial&space;a_{H_{1}}}{\partial&space;W_{input}\rightarrow&space;H_{1}}\\&space;\mapsto&space;\left\{\begin{matrix}\frac{\partial&space;Error}{\partial&space;a_{H_{1}}}=gradient(a_{H_{2}})\cdot\frac{\partial&space;a_{H_{2}}}{\partial&space;a_{H_{1}}}\\&space;\frac{\partial&space;a_{H_{1}}}{\partial&space;W_{input\rightarrow&space;H_{1}}}=a_{input}&space;\end{matrix}\right." title="\frac{\partial Error}{\partial input\rightarrow H_{1}}=\frac{\partial Error}{\partial a_{H_{1}}}\cdot \frac{\partial a_{H_{1}}}{\partial W_{input}\rightarrow H_{1}}\\ \mapsto \left\{\begin{matrix}\frac{\partial Error}{\partial a_{H_{1}}}=gradient(a_{H_{2}})\cdot\frac{\partial a_{H_{2}}}{\partial a_{H_{1}}}\\ \frac{\partial a_{H_{1}}}{\partial W_{input\rightarrow H_{1}}}=a_{input} \end{matrix}\right." /></a>  
Mostly, the difference comes from the second equation. Here, we only have 2 hidden layers. However, the equation will not be too much 
different, if we have more than 2 hidden layers. Let's take a look at the unknown part <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;a_{output}}{\partial&space;a_{H_{2}}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;a_{output}}{\partial&space;a_{H_{2}}}" title="\frac{\partial a_{output}}{\partial a_{H_{2}}}" /></a>. 
Obviously, there are relationship between the top and button of the D.E.: <a href="https://www.codecogs.com/eqnedit.php?latex=a_{output}=W_{H_{2}\rightarrow&space;output}\cdot&space;h_{H_{2}}=W_{H_{2}\rightarrow&space;output}\cdot&space;{g}'(a_{H_{2}})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a_{output}=W_{H_{2}\rightarrow&space;output}\cdot&space;h_{H_{2}}=W_{H_{2}\rightarrow&space;output}\cdot&space;{g}'(a_{H_{2}})" title="a_{output}=W_{H_{2}\rightarrow output}\cdot h_{H_{2}}=W_{H_{2}\rightarrow output}\cdot {g}'(a_{H_{2}})" /></a>  
So we can solve as: <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;a_{output}}{a_{H_{2}}}=W_{H_{2}\rightarrow&space;output}\cdot&space;{g}'(a_{H_{2}})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;a_{output}}{a_{H_{2}}}=W_{H_{2}\rightarrow&space;output}\cdot&space;{g}'(a_{H_{2}})" title="\frac{\partial a_{output}}{a_{H_{2}}}=W_{H_{2}\rightarrow output}\cdot {g}'(a_{H_{2}})" /></a>  
In this case, <a href="https://www.codecogs.com/eqnedit.php?latex=\Delta&space;W_{H_{1}\rightarrow&space;H_{2}}=-l.r.\times&space;\frac{\partial&space;Error}{\partial&space;W_{H_{1}\rightarrow&space;H_{2}}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Delta&space;W_{H_{1}\rightarrow&space;H_{2}}=-l.r.\times&space;\frac{\partial&space;Error}{\partial&space;W_{H_{1}\rightarrow&space;H_{2}}}" title="\Delta W_{H_{1}\rightarrow H_{2}}=-l.r.\times \frac{\partial Error}{\partial W_{H_{1}\rightarrow H_{2}}}" /></a>  
We can also expand the above equation to show the detail <a href="https://www.codecogs.com/eqnedit.php?latex=\Delta&space;W_{H_{1}\rightarrow&space;H_{2}}=-l.r.\times&space;{g}'(a_{H_{2}})\cdot&space;W_{H_{2}\rightarrow&space;output}\cdot&space;gradient(a_{output})\times&space;h_{H_{1}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Delta&space;W_{H_{1}\rightarrow&space;H_{2}}=-l.r.\times&space;{g}'(a_{H_{2}})\cdot&space;W_{H_{2}\rightarrow&space;output}\cdot&space;gradient(a_{output})\times&space;h_{H_{1}}" title="\Delta W_{H_{1}\rightarrow H_{2}}=-l.r.\times {g}'(a_{H_{2}})\cdot W_{H_{2}\rightarrow output}\cdot gradient(a_{output})\times h_{H_{1}}" /></a>.  
Here we can get the knowledge: **no matter how many hidden layers, the equation will be the same.**   
### **B.P. step 3 <a href="https://www.codecogs.com/eqnedit.php?latex=H_{1}&space;\rightarrow&space;input" target="_blank"><img src="https://latex.codecogs.com/gif.latex?H_{1}&space;\rightarrow&space;input" title="H_{1} \rightarrow input" /></a>**  
The last step of backpropagation is not too much difference from the previous part:  
<a href="https://www.codecogs.com/eqnedit.php?latex=x&space;\frac{\partial&space;Error}{\partial&space;input\rightarrow&space;H_{1}}=\frac{\partial&space;Error}{\partial&space;a_{H_{1}}}\cdot&space;\frac{\partial&space;a_{H_{1}}}{\partial&space;W_{input}\rightarrow&space;H_{1}}\\&space;\mapsto&space;\left\{\begin{matrix}\frac{\partial&space;Error}{\partial&space;a_{H_{1}}}=gradient(a_{H_{2}})\cdot\frac{\partial&space;a_{H_{2}}}{\partial&space;a_{H_{1}}}\\&space;\frac{\partial&space;a_{H_{1}}}{\partial&space;W_{input\rightarrow&space;H_{1}}}=a_{input}&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?x&space;\frac{\partial&space;Error}{\partial&space;input\rightarrow&space;H_{1}}=\frac{\partial&space;Error}{\partial&space;a_{H_{1}}}\cdot&space;\frac{\partial&space;a_{H_{1}}}{\partial&space;W_{input}\rightarrow&space;H_{1}}\\&space;\mapsto&space;\left\{\begin{matrix}\frac{\partial&space;Error}{\partial&space;a_{H_{1}}}=gradient(a_{H_{2}})\cdot\frac{\partial&space;a_{H_{2}}}{\partial&space;a_{H_{1}}}\\&space;\frac{\partial&space;a_{H_{1}}}{\partial&space;W_{input\rightarrow&space;H_{1}}}=a_{input}&space;\end{matrix}\right." title="x \frac{\partial Error}{\partial input\rightarrow H_{1}}=\frac{\partial Error}{\partial a_{H_{1}}}\cdot \frac{\partial a_{H_{1}}}{\partial W_{input}\rightarrow H_{1}}\\ \mapsto \left\{\begin{matrix}\frac{\partial Error}{\partial a_{H_{1}}}=gradient(a_{H_{2}})\cdot\frac{\partial a_{H_{2}}}{\partial a_{H_{1}}}\\ \frac{\partial a_{H_{1}}}{\partial W_{input\rightarrow H_{1}}}=a_{input} \end{matrix}\right." /></a>  
Then, we can get: <a href="https://www.codecogs.com/eqnedit.php?latex=\Delta&space;W_{input\rightarrow&space;H_{1}}=-l.r.\times&space;{g}'(a_{H_{1}})\cdot&space;W_{H_{1}\rightarrow&space;H_{2}}\cdot&space;gradient(a_{H_{2}})\times&space;a_{input}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Delta&space;W_{input\rightarrow&space;H_{1}}=-l.r.\times&space;{g}'(a_{H_{1}})\cdot&space;W_{H_{1}\rightarrow&space;H_{2}}\cdot&space;gradient(a_{H_{2}})\times&space;a_{input}" title="\Delta W_{input\rightarrow H_{1}}=-l.r.\times {g}'(a_{H_{1}})\cdot W_{H_{1}\rightarrow H_{2}}\cdot gradient(a_{H_{2}})\times a_{input}" /></a>  
### **B.P. step 4 update the weight**  
<a href="https://www.codecogs.com/eqnedit.php?latex=\left\{\begin{matrix}&space;(new)W_{input&space;\rightarrow&space;H_{1}}=W_{input&space;\rightarrow&space;H_{1}}&plus;\Delta&space;W_{input&space;\rightarrow&space;H_{1}}\\&space;(new)W_{H_{1}\rightarrow&space;H_{2}}=W_{H_{1}\rightarrow&space;H_{2}}&plus;\Delta&space;W_{H_{1}\rightarrow&space;H_{2}}\\&space;(new)W_{H_{2}\rightarrow&space;output}=W_{H_{2}\rightarrow&space;output}&plus;\Delta&space;W_{H_{2}\rightarrow&space;output}\\&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\left\{\begin{matrix}&space;(new)W_{input&space;\rightarrow&space;H_{1}}=W_{input&space;\rightarrow&space;H_{1}}&plus;\Delta&space;W_{input&space;\rightarrow&space;H_{1}}\\&space;(new)W_{H_{1}\rightarrow&space;H_{2}}=W_{H_{1}\rightarrow&space;H_{2}}&plus;\Delta&space;W_{H_{1}\rightarrow&space;H_{2}}\\&space;(new)W_{H_{2}\rightarrow&space;output}=W_{H_{2}\rightarrow&space;output}&plus;\Delta&space;W_{H_{2}\rightarrow&space;output}\\&space;\end{matrix}\right." title="\left\{\begin{matrix} (new)W_{input \rightarrow H_{1}}=W_{input \rightarrow H_{1}}+\Delta W_{input \rightarrow H_{1}}\\ (new)W_{H_{1}\rightarrow H_{2}}=W_{H_{1}\rightarrow H_{2}}+\Delta W_{H_{1}\rightarrow H_{2}}\\ (new)W_{H_{2}\rightarrow output}=W_{H_{2}\rightarrow output}+\Delta W_{H_{2}\rightarrow output}\\ \end{matrix}\right." /></a>  
## Epoch Finished! Do more epochs to minimize the Error by updating the <a href="https://www.codecogs.com/eqnedit.php?latex=[W_{input&space;\rightarrow&space;H_{1}}&space;,W_{H_{1}\rightarrow&space;H_{2}},W_{H_{2}\rightarrow&space;output}]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?[W_{input&space;\rightarrow&space;H_{1}}&space;,W_{H_{1}\rightarrow&space;H_{2}},W_{H_{2}\rightarrow&space;output}]" title="[W_{input \rightarrow H_{1}} ,W_{H_{1}\rightarrow H_{2}},W_{H_{2}\rightarrow output}]" /></a>  
  
   
  
       

# MReLU  
## 问题描述    
现有激活函数均未能有效考虑训练过程卷积层输出特征图中不同像素点具有不同的激活程度，从而无法为那些处于不同激活程度的像素点提供具有差异的梯度响应。在卷积神经网络的训练进程中，越难以学习的特征通常具有越小的激活程度，此类难以学习的特征通常出现在类别相近的图片内容中。为了提升卷积神经网络对困难特征的学习能力，应使激活函数为不同激活程度的像素点提供不同的梯度响应，以便网络将更多注意力集中于困难特征之上。  
## Activation Function  
  * ### 自适应激活函数族  
    自适应激活函数族旨在提出一个自适应激活函数空间用于为特征图中处于不同激活程度的像素点提供具有差异的梯度响应，因此该函数空间中的函数应该具有如下两个性质：  
    * 1）函数空间中的函数应包含一定数量的可学习参数以便在网络训练中充分利用数据信息，达到更好的数据适应性；  
    * 2）函数空间中的函数应具有明显的梯度变化， 以应对处于不同激活程度的像素点，然而为了保持激活函数几乎处处可导和局部平滑，函数空间中的函数应为分段函数，在保持局部平滑的前提下在不同区间中使用不同的梯度响应函数，以实现函数整体上的梯度变化。  
    **该函数族定义如下：**  
    ![自适应激活函数族](https://github.com/895999803/MReLU/blob/master/Activation_Function_Family.jpg)  
  * ### 多斜率自适应激活函数(MReLU)
    多斜率自适应激活函数是自适应激活函数族的一个实例。该实例函数充分继承了自适应激活函数的优点，同时又兼顾了函数本身的简洁性，便于该激活函数在网络中的实际应用。  
    **该函数定义如下：**  
    ![MReLU定义](https://github.com/895999803/MReLU/blob/master/MReLU.jpg)  
    * #### MReLU的优势
      下图中给出了MReLU函数的一个具体示例：  
      ![MReLU示例](https://github.com/895999803/MReLU/blob/master/MReLU_Example.jpg)  
      该示例直观的展现了MReLU所具有的两大特性：  
       * 1）该函数包含有限个跳跃间断点，既为每个区间上的函数选择提供了更大的灵活性，又能自适应地为不同区间上的像素点提供对应的梯度响应；  
       * 2）该函数在不同区间上具有非单调性，其在一定程度上破坏了像素点在激活前后的有序性，为低激活像素点提供了更多的机会获得更高程度的激活，同时对高激活像素点具有一定的抑制作用，其有助于网络将更多的注意力集中到需要被提升的像素点之上。  
    * #### MReLU对激活分布的影响
      下图对比了常见激活函数与MReLU在CIFAR-10数据集上，NIN第三层激活层激活图中像素点激活值的分布情况：   
      ![激活分布对比](https://github.com/895999803/MReLU/blob/master/Comparison.jpg)  
      从图中我们得到了以下两个观察：  
      * 1）GReLU和MReLU的激活分布明显不同于其他激活函数，这一现象直观说明为不同激活程度的像素点提供具有差异的梯度响应能够明显改变像素点的激活分布；  
      * 2）相比于GReLU，MReLU在正值域中倾向于使像素点获得更高程度的激活，这一现象直观说明MReLU所具有的非单调特性对于激活分布的有效增强。  
       
## 实验结果  
为了充分验证MReLU在不同网络不同数据集上的适应性。网络结构上使用了三种不同的网络结构分别为：NIN，MobileNet-V2和ResNet-18。数据集上使用三种常见图片分类数据集：CIFAR-10，CIFAR-100和ImageNet。网络结构上即兼顾了不同的深度，又兼顾了不同的运行速率。数据集的选择上即兼顾了不同的复杂度，又兼顾了不同的数据容量。由此可见，本次实验中使用的网络结构和数据集充分包含了网络结构和数据集的各种类别，足以充分验证和对比不同激活函数的性能优劣。  
**注：ImageNetS——由于ImageNet数据集过于庞大，实验中从原始数据集中随机抽取了100个子类构成实际的训练数据集**

NIN (%)|ReLU|LReLU|PReLU|RReLU|ELU|SELU|Swish|BReLU|GReLU|MReLU  
----|----|----|----|----|----|----|----|----|----|----
CIFAR-10|86.27|86.34|86.59|87.67|86.76|85.95|85.71|83.87|86.65|**87.96**
CIFAR-100|64.95|65.50|66.70|68.62|66.07|67.34|66.25|59.37|**70.41**|69.01
ImageNetS|**78.23**|77.53|74.73|73.6|77.05|70.65|64.65|45.15|70.08|76.05

ResNet-18 (%)|ReLU|LReLU|PReLU|RReLU|ELU|SELU|Swish|BReLU|GReLU|MReLU
----|----|----|----|----|----|----|----|----|----|----
CIFAR-10|88.19|87.86|88.32|88.22|87.35|88.03|88.48|88.12|88.42|**88.56**
CIFAR-100|72.22|72.71|73.11|72.63|72.75|73.52|72.01|72.31|72.58|**73.54**
ImageNetS|83.8|83.02|82.62|82.92|83.87|83.87|83.27|82.65|83.65|**83.95**  

MobileNet-V2 (%)|ReLU|LReLU|PReLU|RReLU|ELU|SELU|Swish|BReLU|GReLU|MReLU
----|----|----|----|----|----|----|----|----|----|----
CIFAR-10|70.45|71.31|76.09|71.4|74.5|76.04|71.12|72.93|76.41|**78.33**
CIFAR-100|46.84|49.31|45.06|46.49|53.02|51.45|48.75|51.44|36.83|**55.08**
ImageNetS|76.52|76.02|74.87|**77.20**|76.0|70.82|76.42|75.8|73.57|76.62  

**MReLU实验结果汇总：**  
 NIN+CIFAR-10：**最优分类准确率**   
 NIN+CIFAR-100：次优分类准确率  
 ResNet-18+CIFAR-10，CIFAR-100，ImageNetS：**最优分类准确率**    
 MobileNet-V2+CIFAR-10，CIFAR-100：**最优分类准确率**    
 MobileNet-V2+ImageNetS：次优分类准确率  
实验结果从整体上表明了MReLU的普遍优势和普遍适应性。

## 参考文献
[1] Nair V, Hinton G E. Rectified Linear Units Improve Restricted Boltzmann Machines Vinod Nair. [C]International Conference on Machine Learning, Haifa, IsraelNair, 2010.  
[2] Maas A L, Hannun A Y, Ng A Y. Rectifier nonlinearities improve neural network acoustic models. [C]International Conference on Machine Learning, Atlanta, GA, USA, 2013.  
[3] He K, Zhang X, Ren S, et al. Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification. [C]IEEE International Conference on Computer Vision, CentroParque Convention Center, Santiago, Chile, 2015.  
[4] Xu B, Wang N, Chen T, et al. Empirical Evaluation of Rectified Activations in Convolutional Network[OL]. [2015-05-05]. https://arxiv.org/abs/ 1505.00853.  
[5] Clevert D A, Unterthiner T, Hochreiter S. Fast and accurate deep network learning by exponential linear units (ELUs). [C] International Conference on Learning Representations, San Juan, Puerto Rico, 2016.  
[6] Klambauer G, Unterthiner T, Mayr A, et al. Self-normalizing neural networks. [C]Advances in Neural Information Processing Systems, Long Beach, California, USA, 2017.  
[7] Ramachandran P, Zoph B, Le Q V. Searching for Activation Functions [OL]. [2017-10-16]. https://arxiv.org/abs/1710.05941.  
[8] Krizhevsky A, Hinton G. Convolutional deep belief networks on cifar-10[J]. Unpublished manuscript, 2010, 40(7): 1-9. http://www.cs.utoronto.ca/~kriz/ conv-cifar10-aug2010.pdf.  
[9] Chen Z, Ho P. Deep Global-Connected Net With The Generalized Multi-Piecewise ReLU Activation in Deep Learning[OL]. [2018-06-19]. https://arxiv.org/abs/1807.03116.  
[10] 刘海，刘波，胡瑜. 多斜率自适应激活函数[J]. 空间控制技术与应用, 2020, 46(3): 29-37.  
[11] Lin M, Chen Q, Yan S. Network in network[OL]. [2013-12-16]. https://arxiv.org/abs/1312.4400.  
[12] He K, Zhang X, Ren S, et al. Deep residual learning for image recognition. [C]IEEE Conference on Computer Vision and Pattern Recognition, Las Vegas, Nevada, USA, 2016.  
[13] Sandler M, Howard A, Zhu M, et al. Mobilenetv2: Inverted residuals and linear bottlenecks. [C]IEEE Conference on Computer Vision and Pattern Recognition, Salt Lake City, USA, 2018.  
[14] Krizhevsky A, Hinton G. Learning multiple layers of features from tiny images. Computer Science Department[J]. University of Toronto, Tech. Rep, 2009, 1(4): 7.  
[15] Deng J, Dong W, Socher R, et al. ImageNet: A large-scale hierarchical image database. [C] IEEE Conference on Computer Vision and Pattern Recognition, Miami, FL, USA, 2009.  





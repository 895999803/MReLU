# MReLU  
## 问题描述    
  现有激活函数均未有效考虑训练过程中卷积层输出特征图中不同像素点具有不同的激活程度，从而无法为这些处于不同激活程度的像素点提供具有差异的梯度响应。在卷积神经网络的训练进程中，越难以学习的特征通常具有越小的激活程度，此类难以学习的特征通常出现在类别相近的图片内容中。为了提升卷积神经网络对困难特征的学习能力，应使激活函数为不同激活程度的像素点提供不同的梯度响应，以便网络将更多注意力集中于困难特征之上。  
## Activation Function  
  * ### 自适应激活函数族  
    自适应激活函数族旨在提出一个自适应激活函数空间用于为特征图中处于不同激活程度的像素点提供具有差异的梯度响应，因此该函数空间中的函数应该具有如下两个性质：  
    * 1）函数空间中的函数应包含一定数量的可学习参数以便在网络训练中充分利用数据信息，达到更好的数据适应性；  
    * 2）函数空间中的函数应具有明显的梯度变化， 以应对处于不同激活程度的像素点，然而为了保持激活函数几乎处处可导和局部平滑，函数空间中的函数应为分段函数，在保持局部平滑的前提下在不同区间中使用不同的梯度响应函数，以实现函数整体上的梯度变化。  
    该函数族定义如下：  
    ![自适应激活函数族](https://github.com/895999803/MReLU/blob/master/Activation_Function_Family.jpg)  
  * ### 多斜率自适应激活函数(MReLU)
    多斜率自适应激活函数是自适应激活函数族的一个实例。该实例函数充分继承了自适应激活函数的优点，同时又兼顾了函数本身的简洁性，便于该激活函数在网络中的实际应用，该函数定义如下:  
    ![MReLU](https://github.com/895999803/MReLU/blob/master/MReLU.jpg)  
    * #### MReLU的优势
      MReLU的两大特性：  
      * 1）该函数包含有限个跳跃间断点，既为每个区间上的函数选择提供了更大的灵活性，又能自适应地为不同区间上的像素点提供对应的梯度响应；  
      * 2）该函数在不同区间上具有非单调性，其在一定程度上破坏了像素点在激活前后的有序性，为低激活像素点提供了更多的机会获得更高程度的激活，同时对高激活像素点具有一定的抑制作用，其有助于网络将更多的注意力集中到需要被提升的像素点之上。  
      下图中给出了MReLU函数的一个具体示例，该示例直观的展现了MReLU所具有的两大特性。  
      ![MReLU](https://github.com/895999803/MReLU/blob/master/MReLU.jpg)  
    * #### MReLU对激活分布的影响
    
    


# MReLU  
## 问题描述    
  现有激活函数均未有效考虑训练过程中卷积层输出特征图中不同像素点具有不同的激活程度，从而无法为这些处于不同激活程度的像素点提供具有差异的梯度响应。在卷积神经网络的训练进程中，越难以学习的特征通常具有越小的激活程度，此类难以学习的特征通常出现在类别相近的图片内容中。为了提升卷积神经网络对困难特征的学习能力，应使激活函数为不同激活程度的像素点提供不同的梯度响应，以便网络将更多注意力集中于困难特征之上。  
## MReLU Active Function  
  ### * 自适应激活函数族  
    自适应激活函数族旨在提出一个自适应激活函数空间用于为特征图中处于不同激活程度的像素点提供具有差异的梯度响应，因此该函数空间中的函数应该具有如下两个性质：1）函数空间中的函数应包含一定数量的可学习参数以便在网络训练中充分利用数据信息，达到更好的数据适应性；2）函数空间中的函数应具有明显的梯度变化， 以应对处于不同激活程度的像素点，然而为了保持激活函数几乎处处可导和局部平滑，函数空间中的函数应为分段函数，在保持局部平滑的前提下在不同区间中使用不同的梯度响应函数，以实现函数整体上的梯度变化。
    


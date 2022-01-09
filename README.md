# Deep morphological simplification network (MS-Net) for Brain MR Image Registration
Code review processing. Welcome for any bug report!

## Background
Deformable brain MR image registration is challenging due to large inter-subject anatomical variation. 
For example, the highly complex cortical folding pattern makes it hard to accurately align corresponding cortical structures of individual images. 
In this paper, we propose a novel deep learning way to simplify the difficult registration problem of brain MR images. 
Specifically, we train a morphological simplification network (MS-Net), 
which can generate a simple image with less anatomical details based on the complex input. 
With MS-Net, the complexity of the fixed image or the moving image under registration can be reduced gradually, 
thus building an individual (simplification) trajectory represented by MS-Net outputs. 
Since the generated images at the ends of the two trajectories (of the fixed and moving images) are so simple and very similar in appearance, 
they are easy to register. Thus, the two trajectories can act as a bridge to link the fixed and the moving images, and guide their registration. 
Our experiments show that the proposed method can achieve highly accurate registration performance on different datasets (i.e., NIREP, LPBA, IBSR, CUMC, and MGH). 
Moreover, the method can be also easily transferred across diverse image datasets and obtain superior accuracy on surface alignment. 
We propose MS-Net as a powerful and flexible tool to simplify brain MR images and their registration. 
To our knowledge, this is the first work to simplify brain MR image registration by deep learning, instead of estimating deformation field directly.



  * If you use MS-Net, please cite:
    
    **Deep morphological simplification network (MS-Net) for guided registration of brain magnetic resonance images**   
    Dongming Wei, Lichi Zhang, Zhengwang Wu, Xiaohuan Cao, Gang Li, Dinggang Shen, Qian Wang\
    [Pattern Recognition](https://www.sciencedirect.com/science/article/abs/pii/S0031320319304716)

    **Morphological Simplification of Brain MR Images by Deep Learning for Facilitating Deformable Registration**    
    Dongming Wei, Sahar Ahmad, Zhengwang Wu, Xiaohuan Cao, Xuhua Ren, Gang Li, Dinggang Shen, Qian Wang\
    [MLMI 2019](https://link.springer.com/chapter/10.1007/978-3-030-32692-0_24)

## Instruction
1. Simplified Image Generation
    + For generating the simplified image, please use `python predict.py`. The pretrained model is supplied.




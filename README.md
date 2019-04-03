# Deep EPCC

This is the software package for deep neural networks using polyhedral conic function loss. We integrated the polyhedral conic classification loss function into ResNet-101 deep neural network architecture and tested the resulting deep neural network classifier on different classification problems including binary classification, multi-class and multi-label classification. 

Majority of the programs are identical to the ones in classical MatConvNet toolbox, but we made minor changes to integrate the polyhedral conic function loss. To this end, we added **“Polyhedral.m”** file into directory “ResNet-Matconvnet/dependencies/matconvnet/ matlab/+dagnn/” and we revised the file **“vl_nnloss.m”** under the directory “ResNet-Matconvnet/dependencies/matconvnet/”.

**For binary classification**, labels must be given as “+1” for positive class samples and “-1” for negatives.  Please see **“run_master_pascalvoc.m”** file to train the network for binary classification (this is more like a one-class problem where the positive class samples are surrounded by negatives). Please also check “pascalvoc_imdb_aeroplane.mat” for assigning labels to the images.

**For multi-class classification problems with single label**, labeling is identical to the classical label assignment in MatConvNet toolbox.  To train the ResNet-101 network using polyhedral conic function loss, method term must be set to “**method = ‘polyhedral’** ” in run_master.m file.

**For multi-label classification**, labels must be given in one-hot vector format. Please check **“coco_imdb.mat”** file for multi-label assignment. To train the ResNet-101 network using polyhedral conic function loss, method term must be set to “**method = ‘polyhedral_ml’**” in **run_master.m** or **run_master_coco** files.

For testing trained models, please see the files **“run_our_test_pascalvoc.m”**, **“run_our_test_cifar.m”**, **“run_our_test_coco.m”** files.

Download trained models from [here](https://github.com/user/repo/blob/branch/other_file.md)

Please cite the following papers if you use this package.

[1] H. Cevikalp, H. Sağlamlar, “Polyhedral conic classifiers for computer vision applications and open set recognition,” IEEE Transactions on PAMI (under review).

[2] H. Cevikalp, B. Triggs, “Polyhedral conic classifiers for visual object detection and classification,” CVPR, 2017.

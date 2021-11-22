## Team members:
1. Nidhi Chandra
2. Yamini Kashyap
3. Indrajeet Nandy

Credits to https://github.com/huyvnphan/PyTorch_CIFAR10 for setting up pretrained model on CIFAR10 data.

Install requirements:
!pip install pytorch_lightning

To run:
!python load_and_test.py
This test script will load models one by one and evaluate and print test accuracy on them- original pretrained model followed by one-shot pruned(50%, 75%, 90%) models and iteratively pruned models(50%, 75%, 90%) respectively. 

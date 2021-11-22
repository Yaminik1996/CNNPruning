## Team members:
1. Nidhi Chandra
2. Yamini Kashyap
3. Indrajeet Nandy

## Steps to run the project:
1. Create a virtual environment where you can install the required libraries and run the pruned models:

        
        virtualenv -p python3 pruning
        source pruning/bin/activate
        
2. Run the shell script **run.sh**, which downloads the model and also runs load_and_test.py

      ```
      source run.sh
      ```

Find documentation for this code at docs/documentation.pdf <br>
Credits to https://github.com/huyvnphan/PyTorch_CIFAR10 for setting up pretrained model on CIFAR10 data.

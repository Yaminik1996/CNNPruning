import torch
import math
from data import CIFAR10Data
from module import CIFAR10Module
from pytorch_lightning import Trainer
from argparse import ArgumentParser

def calculate_sparsity(model):
  total_count = 0
  zero_count = 0
  
  for buffer_name, buffer in model.named_buffers():
    zero_count += torch.sum(buffer == 0).item()
    if zero_count>0:
      total_count += buffer.nelement()

  print("Total params: ", total_count)
  print("Zero params: ", zero_count)
  return (math.ceil(zero_count*100/total_count))

class Args:
  def __init__(self):
    self.batch_size = 256
    self.classifier = 'resnet18'
    self.data_dir = 'data/huy/cifar10'
    self.dev = 0
    self.download_weights = 0
    self.gpu_id = '0'
    self.learning_rate = 0.01
    self.logger = 'tensorboard'
    self.max_epochs = 100
    self.num_workers = 8
    self.precision = 32
    self.pretrained = 1
    self.test_phase = 1
    self.weight_decay = 0.01

if __name__ == "__main__":
  trainer = Trainer(
            fast_dev_run=bool(0),
            logger=logger if not bool(0 + 1) else None,
            deterministic=True,
            weights_summary=None,
            #log_every_n_steps=1,
            max_epochs=100,
            checkpoint_callback=None,
            precision=32,
            accelerator="cpu"
        )

  args = Args()
  data = CIFAR10Data(args)

  model1 = torch.load("one_shot_50.pth",map_location=torch.device('cpu'))
  print("Sparsity of model", calculate_sparsity(model1))
  trainer.test(model1, data.test_dataloader())

  model2 = torch.load("one_shot_75.pth",map_location=torch.device('cpu'))
  print("Sparsity of model", calculate_sparsity(model2))
  trainer.test(model2, data.test_dataloader())

  model3 = torch.load("one_shot_90.pth",map_location=torch.device('cpu'))
  print("Sparsity of model", calculate_sparsity(model3))
  trainer.test(model3, data.test_dataloader())

  model4 = torch.load("iterative_50.pth",map_location=torch.device('cpu'))
  print("Sparsity of model", calculate_sparsity(model4))
  trainer.test(model4, data.test_dataloader())

  model5 = torch.load("iterative_75.pth",map_location=torch.device('cpu'))
  print("Sparsity of model", calculate_sparsity(model5))
  trainer.test(model5, data.test_dataloader())

  model6 = torch.load("iterative_90.pth",map_location=torch.device('cpu'))
  print("Sparsity of model", calculate_sparsity(model6))
  trainer.test(model6, data.test_dataloader())
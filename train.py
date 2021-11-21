import os
from argparse import ArgumentParser

import numpy
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import copy
import math

from data import CIFAR10Data
from module import CIFAR10Module

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

def main(args):

    print(type(args))
    print(args)

    if bool(args.download_weights):
        CIFAR10Data.download_weights()
    else:
        seed_everything(0)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

        if args.logger == "wandb":
            logger = WandbLogger(name=args.classifier, project="cifar10")
        elif args.logger == "tensorboard":
            logger = TensorBoardLogger("cifar10", name=args.classifier)

        checkpoint = ModelCheckpoint(monitor="acc/val", mode="max", save_last=False)

        trainer = Trainer(
            fast_dev_run=bool(args.dev),
            logger=logger if not bool(args.dev + args.test_phase) else None,
            gpus=-1,
            deterministic=True,
            weights_summary=None,
            #log_every_n_steps=1,
            max_epochs=args.max_epochs,
            checkpoint_callback=checkpoint,
            precision=args.precision,
        )

        model = CIFAR10Module(args)
        data = CIFAR10Data(args)

        if bool(args.pretrained):
            state_dict = os.path.join(
                "cifar10_models", "state_dicts", args.classifier + ".pt"
            )
            model.model.load_state_dict(torch.load(state_dict))

        if bool(args.test_phase):
            trainer.test(model, data.test_dataloader())
        else:
            trainer.fit(model, data)
            trainer.test()

        # One-shot pruning
        model1 = copy.deepcopy(model)
        model_copy1 = model1.model
        print("Copied 1")

        parameters_to_prune =[]
        for module_name, module in model1.named_modules():
                if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                    parameters_to_prune.append((module, "weight"))
        print("Layers to prune {}".format((len(parameters_to_prune))))

        prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.5,
        )
        print("Sparsity of model1", calculate_sparsity(model_copy1))
        trainer.test(model1, data.test_dataloader())
        torch.save(model1, "one_shot_50.pth")

        model2 = copy.deepcopy(model)
        model_copy2 = model2.model
        print("Copied 2")

        parameters_to_prune = []
        for module_name, module in model2.named_modules():
          if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, "weight"))

        print("Layers to prune {}".format((len(parameters_to_prune))))

        prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.75,
        )
        print("Sparsity of model2", calculate_sparsity(model_copy2))
        trainer.test(model2, data.test_dataloader())
        torch.save(model2, "one_shot_75.pth")

        model3 = copy.deepcopy(model)
        model_copy3 = model3.model
        print("Copied 3")

        parameters_to_prune = []
        for module_name, module in model3.named_modules():
          if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, "weight"))

        print("Layers to prune {}".format((len(parameters_to_prune))))

        prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.90,
        )
        print("Sparsity of model3", calculate_sparsity(model_copy3))
        trainer.test(model3, data.test_dataloader())
        torch.save(model3, "one_shot_90.pth")

        # #Iterative pruning
        model4 = copy.deepcopy(model)
        model_copy4 = model4.model
        print("Copied 4")

        parameters_to_prune = []
        for module_name, module in model4.named_modules():
                if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                    parameters_to_prune.append((module, "weight"))

        print("Layers to prune {}".format((len(parameters_to_prune))))

        #Iterative pruning 13% 5 times -> 50% sparsity
        for i in range(5):
          prune.global_unstructured(
          parameters_to_prune,
          pruning_method=prune.L1Unstructured,
          amount=0.13,
          )
        
        print("Sparsity of model4", calculate_sparsity(model_copy4))
        trainer.test(model4, data.test_dataloader())
        torch.save(model4, "iterative_50.pth")
        
        model5 = copy.deepcopy(model)
        model_copy5 = model5.model
        print("Copied 5")

        parameters_to_prune = []
        for module_name, module in model5.named_modules():
                if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                    parameters_to_prune.append((module, "weight"))

        print("Layers to prune {}".format((len(parameters_to_prune))))

        #Iterative pruning 24.3% 5 times -> 75% sparsity
        for i in range(5):
          prune.global_unstructured(
          parameters_to_prune,
          pruning_method=prune.L1Unstructured,
          amount=0.243,
          )
        print("Sparsity of model5", calculate_sparsity(model_copy5))
        trainer.test(model5, data.test_dataloader())
        torch.save(model5, "iterative_75.pth")
        

        model6 = copy.deepcopy(model)
        model_copy6 = model6.model
        print("Copied 6")

        parameters_to_prune = []
        for module_name, module in model6.named_modules():
                if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                    parameters_to_prune.append((module, "weight"))

        print("Layers to prune {}".format((len(parameters_to_prune))))

        #Iterative pruning 36.9% 5 times -> 90% sparsity
        for i in range(5):
          prune.global_unstructured(
          parameters_to_prune,
          pruning_method=prune.L1Unstructured,
          amount=0.369,
          )
        print("Sparsity of model6", calculate_sparsity(model_copy6))
        trainer.test(model6, data.test_dataloader())
        torch.save(model6, "iterative_90.pth")


if __name__ == "__main__":
    parser = ArgumentParser()

    # PROGRAM level args
    parser.add_argument("--data_dir", type=str, default="/data/huy/cifar10")
    parser.add_argument("--download_weights", type=int, default=0, choices=[0, 1])
    parser.add_argument("--test_phase", type=int, default=0, choices=[0, 1])
    parser.add_argument("--dev", type=int, default=0, choices=[0, 1])
    parser.add_argument(
        "--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"]
    )

    # TRAINER args
    parser.add_argument("--classifier", type=str, default="resnet18")
    parser.add_argument("--pretrained", type=int, default=0, choices=[0, 1])

    parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpu_id", type=str, default="0")

    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    args = parser.parse_args()
    main(args)

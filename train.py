import torch
import numpy as np
import math

if __name__ == "__main__":
    x = torch.randn(512)
    for i in range(100):
        w = torch.randn(512, 512)
        x = w @ x
        print(x)

    print(x.mean(), x.std())
    pass

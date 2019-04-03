import numpy as np
import torch
import cv2
start=torch.tensor([1,6,2,2])
prediction = start.view(1, 6,2*2)
prediction = prediction.transpose(1, 2).contiguous()
prediction = prediction.view(1, 2*2*3, 2)
# b=start.view(1, 255, 13*13)
# b= b.transpose(1, 2).contiguous()
# b = b.view(1, 13*13*3, 85)
print(prediction)
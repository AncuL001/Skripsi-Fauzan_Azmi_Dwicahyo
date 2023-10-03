# YOLO Implementation

Source: https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/YOLO

Associated Video: https://www.youtube.com/watch?v=n9_XyCGr-MI

## Notes

### 
$$
n_{out} = \left\lfloor\frac{n_{in} + 2p - k}{s}\right\rfloor +1
$$

$$
n_{in} = \text{input dimension}\\
n_{out} = \text{output dimension}\\
k = \text{kernel size}\\
p = \text{padding}\\
s = \text{stride}\\
$$

```py
def calcOutputDim(inputDim, kernelSize, stride, padding):
    return (inputDim - kernelSize + 2 * padding) // stride + 1
```
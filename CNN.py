import numpy as np
import math


class filer(object):
    # Input_channel为每一个kernel的depth，output_channel为kernel的个数（该类中为权重初始化服务）
    # kernel_size为kernel的尺寸
    def __init__(self, input_channel, output_channel, kernel_size):
        self.weights = np.random.uniform(0, 1, (output_channel, input_channel, kernel_size, kernel_size))
        self.bias = 0
        self.delta_weights = 0
        self.delta_bias = 0

    @property
    def get_weights(self):
        return self.weights

    @property
    def get_bias(self):
        return self.bias

    def unpdate(self, learning_rate):
        self.weights += learning_rate * self.delta_weights
        self.bias += learning_rate * self.delta_bias


# 实现功能：前向计算、反向传播以及更新权重参数
class conv_layer(object):
    # Input_channel为每一个kernel的depth，output_channel为kernel的个数（该类中为权重初始化服务）
    # kernel_size为kernel的尺寸
    def __init__(self, input_channel, output_channel, kernel_size, padding=1, stride=1):
        # 1、输入与输出图尺寸
        self.input_height = None
        self.input_width = None
        # 2、卷积相关参数
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        # 3、卷积核对象
        self.kernel_array = filer(self.input_channel, self.output_channel, self.kernel_size)
        # 4、激活函数与优化器参数
        self.activator = None
        self.learning_rate = 0.001

    # 前向计算
    def forward(self, input_array):
        # 1、输入图像，只支持C*H*W
        if input_array.ndim == 3:
            self.input_height = input_array.shape[-2]
            self.input_width = input_array.shape[-1]
            self._calc_output_size()
        else:
            raise ValueError('dim must be 3')
        # 2、padding输入图像
        if self.padding is not 0:
            input_array_padding = self._padding(input_array)
        else:
            input_array_padding = input_array
        # 3、执行卷积操作
        output_array = self._conv2d(input_array_padding)
        return output_array

    # 反向传播
    def backward(self):
        pass

    # 更新权重
    def update(self):
        pass

    # padding有两种思路：1、构建padding后尺寸的zero矩阵，然后将重心区域填入原图像
    #                  2、直接在原图像的基础上，左右上下concatenate zero padding矩阵
    def _padding(self, input_array):
        zero_row = np.zeros((self.input_channel, self.padding, self.input_width))
        zero_col = np.zeros((self.input_channel, self.input_height + 2 * self.padding, self.padding))
        # 需要注意图像尺寸有两种情况，分别是H*W和C*H*W
        input_array_padding = np.concatenate((zero_row, input_array, zero_row), axis=-2)
        input_array_padding = np.concatenate((zero_col, input_array_padding, zero_col), axis=-1)
        return input_array_padding

    def _compute_cov(self, input_array, weights):
        res = np.zeros((1, self.out_height, self.out_width))
        for y in range(self.out_height):
            for x in range(self.out_width):
                input_patch = input_array[:, y:y + self.kernel_size, x:x + self.kernel_size]
                res[0][y][x] = np.multiply(input_patch, weights).sum()
        return res

    def _conv2d(self, input_array):
        # 分多个不同filter分别做conv2d操作
        output_array = np.zeros((self.output_channel, self.out_height, self.out_width))
        for i in range(self.output_channel):
            weights = self.kernel_array.get_weights[i]
            res = self._compute_cov(input_array, weights)
            output_array[i, :, :] = res
        return output_array

    def _calc_output_size(self):
        # 需要思考在输出图像尺寸上的取整问题？
        self.out_height = int((self.input_height - self.kernel_size + 2 * self.padding) / self.stride + 1)
        self.out_width = int((self.input_width - self.kernel_size + 2 * self.padding) / self.stride + 1)

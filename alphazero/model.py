import os

import haiku as hk
import jax
import jax.numpy as jnp
from jax.nn import relu


class ResBlock(hk.Module):
    def __init__(self, dim: int, name: str):
        super().__init__(name)
        self.conv1 = hk.Conv2D(
            output_channels=dim//2, kernel_shape=3, data_format='NCHW', name='first_convolution')
        self.conv2 = hk.Conv2D(
            output_channels=dim, kernel_shape=3, data_format='NCHW', name='second_convolution')

    def __call__(self, x):
        residual = relu(self.conv1(x))
        residual = self.conv2(residual)
        return x + residual


class ValueHead(hk.Module):
    def __init__(self):
        super().__init__('value_head')
        self.linear1 = hk.Linear(output_size=256, name='first_linear')
        self.linear2 = hk.Linear(output_size=256, name='second_linear')
        self.output = hk.Linear(output_size=1, name='output_layer')

    def __call__(self, x):
        x = relu(self.linear1(x))
        x = relu(self.linear2(x))
        return self.output(x)


class PolicyHead(hk.Module):
    def __init__(self):
        super().__init__('policy_head')
        self.linear1 = hk.Linear(output_size=256, name='first_linear')
        self.linear2 = hk.Linear(output_size=256, name='second_linear')
        self.output = hk.Linear(output_size=81, name='output_layer')

    def __call__(self, x):
        x = relu(self.linear1(x))
        x = relu(self.linear2(x))
        # output logits
        return self.output(x)


class ResNet(hk.Module):
    def __init__(self, name=None):
        super().__init__(name)
        self.conv = hk.Conv2D(output_channels=8, kernel_shape=3,
                              data_format='NCHW', name='input_convolution')
        self.resblock1 = ResBlock(8, name='first_residual_block')
        self.resblock2 = ResBlock(8, name='second_residual_block')
        self.flatten = hk.Flatten(name='flatten')
        self.value_head = ValueHead()
        self.policy_head = PolicyHead()

    def __call__(self, x):
        x = relu(self.conv(x))
        x = relu(self.resblock1(x))
        x = relu(self.resblock2(x))
        feature = self.flatten(x)
        return self.value_head(feature), self.policy_head(feature)


def resnet_forward(state):
    resnet = ResNet()
    return resnet(state)


if __name__ == '__main__':
    dummy_state = jnp.zeros((1, 17, 9, 9))
    network = hk.without_apply_rng(hk.transform(resnet_forward))
    model = hk.experimental.tabulate(network)(dummy_state)
    dir_path = os.path.dirname(__file__)
    with open(os.path.join(dir_path, 'model_visual.txt'), 'w') as fp:
        fp.write(model)
    key = jax.random.PRNGKey(0)
    params = network.init(rng=key, state=dummy_state)
    val, logits = network.apply(params, jnp.ones((10, 17, 9, 9)))
    print(val.shape, logits.shape)

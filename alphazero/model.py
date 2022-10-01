import os

import haiku as hk
import jax
import jax.numpy as jnp
from jax.nn import relu


class ResBlock(hk.Module):
    def __init__(self, dim: int, is_train:bool, name: str):
        super().__init__(name)
        self.is_train = is_train

        self.conv1 = hk.Conv2D(
            output_channels=dim//2, kernel_shape=3, data_format='NCHW', name='first_convolution')
        
        self.batch_norm = hk.BatchNorm(create_scale=True, create_offset=True,
                                        decay_rate=0.9, data_format='NCHW', name='internal_batch_norm')

        self.conv2 = hk.Conv2D(
            output_channels=dim, kernel_shape=3, data_format='NCHW', name='second_convolution')

    def __call__(self, x):
        residual = relu(self.conv1(x))
        residual = self.batch_norm(residual, is_training=self.is_train)
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
    def __init__(self, is_train: bool, name=None):
        super().__init__(name)
        self.is_train = is_train

        self.conv = hk.Conv2D(output_channels=8, kernel_shape=3,
                              data_format='NCHW', name='input_convolution')
        self.batch_norm1 = hk.BatchNorm(create_scale=True, create_offset=True,
                                        decay_rate=0.9, data_format='NCHW', name='input_conv_batch_norm')

        self.resblock1 = ResBlock(dim=8, is_train=is_train, name='first_residual_block')
        self.batch_norm2 = hk.BatchNorm(create_scale=True, create_offset=True,
                                        decay_rate=0.9, data_format='NCHW', name='first_resblock_batch_norm')

        self.resblock2 = ResBlock(dim=8, is_train=is_train, name='second_residual_block')
        self.batch_norm3 = hk.BatchNorm(create_scale=True, create_offset=True,
                                        decay_rate=0.9, data_format='NCHW', name='second_resblock_batch_norm')

        self.flatten = hk.Flatten(name='flatten')
        self.value_head = ValueHead()
        self.policy_head = PolicyHead()

    def __call__(self, x):
        x = relu(self.conv(x))
        x = self.batch_norm1(x, is_training=self.is_train)

        x = relu(self.resblock1(x))
        x = self.batch_norm2(x, is_training=self.is_train)

        x = relu(self.resblock2(x))
        x = self.batch_norm3(x, is_training=self.is_train)

        feature = self.flatten(x)
        return self.value_head(feature), self.policy_head(feature)


def create_model(training: bool):
    '''
    create a haiku transformed residual network function
    '''
    def forward(state):
        resnet = ResNet(is_train=training)
        return resnet(state)
    model = hk.without_apply_rng(hk.transform_with_state(forward))
    return model


def init_model(model, seed: int = 0):
    '''
    initialize a haiku transformed function given the random seed
    return parameters, state
    '''
    dummy_state = jnp.zeros((1, 17, 9, 9))
    key = jax.random.PRNGKey(seed)
    params, state = model.init(rng=key, state=dummy_state)
    return params, state


def model_summary(model):
    dummy_state = jnp.zeros((1, 17, 9, 9))
    summary = hk.experimental.tabulate(model)(dummy_state)
    dir_path = os.path.dirname(__file__)
    with open(os.path.join(dir_path, 'model_visual.txt'), 'w') as fp:
        fp.write(summary)


if __name__ == '__main__':
    model = create_model(True)
    model_summary(model)
    params, state = init_model(model, seed=0)
    model2 = create_model(False)
    (val, logits), state = model2.apply(params, state, jnp.ones((10, 17, 9, 9)))
    print(val.shape, logits.shape)

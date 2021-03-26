import cv2
from tensorflow.python.keras.engine.base_preprocessing_layer import PreprocessingLayer




def preprocessing_fun(img):
    dst = cv2.fastNlMeansDenoisingColored(img.astype(np.uint8), None, 10, 10, 7, 21)
    return dst




class RandomCutout(PreprocessingLayer):
    """Creates random mask on the image.
  Input shape:
    4D tensor with shape:
    `(samples, height, width, channels)`, data_format='channels_last'.
  Output shape:
    4D tensor with shape:
    `(samples, height, width, channels)`, data_format='channels_last'.
  Attributes:
    mask: A tuple or a list with two values `mask-height` and `mask-width`.
    seed: Integer. Used to create a random seed.
    name: A string, the name of the layer.
  Raise:
    ValueError: if mask is not a list or tuple of two values.
    InvalidArgumentError: if mask_size (mask_height x mask_width) can't be divisible by 2. 
  """
    def __init__(self, mask, seed=None, name=None, **kwargs):
        self.mask = mask
        if isinstance(mask, (tuple, list)) and len(mask) == 2:
            self.lower = mask[0]
            self.upper = mask[1]
            
        else:
            raise ValueError('RandomCutout layer {name} received an invalid mask '
                       'argument {arg}. only list or touple of size 2 should be passed'.format(name=name, arg=mask))

        self.seed = seed
        self.input_spec = InputSpec(ndim=4)
        super(RandomCutout, self).__init__(name=name, **kwargs)
        base_preprocessing_layer._kpl_gauge.get_cell('V2').set('RandomCutout')

    def call(self, inputs, training=True):
        if training is None:
            training = K.learning_phase()

        def random_cutout_inputs():
            return tfa.image.random_cutout(inputs, (self.lower, self.upper), constant_values = 0)

        output = control_flow_util.smart_cond(training, random_cutout_inputs,
                                              lambda: inputs)
        output.set_shape(inputs.shape)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'mask': self.mask,
            'seed': self.seed,
        }
        
        base_config = super(RandomCutout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


import tensorflow as tf 
from tensorflow.keras import layers 
import numpy as np 
import net_model
from parse_arguments import parse_opts
#from mean import get_mean, get_std
from spatial_transforms import *
from temporal_transforms import *
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose
from dataset import get_training_set, get_validation_set, get_test_set

batch_size = 4
### -------------------------------------- Data loader 
opt = parse_opts()

spatial_transform = Compose([
            RandomRotate(),
            RandomResize(),            
            ToTensor(opt.norm_value)
        ])
temporal_transform = TemporalRandomCrop(opt.sample_duration)
target_transform = ClassLabel()

# Training and Validation data
training_data = enumerate(get_training_set(opt, spatial_transform, temporal_transform, target_transform))
validation_data = enumerate(get_validation_set(opt,spatial_transform, temporal_transform, target_transform))

tf_training_data = tf.data.Dataset.from_tensor_slices(training_data).batch(batch_size).shuffle(100)
tf_validation_data = tf.data.Dataset.from_tensor_slices(validation_data).batch(batch_size)


### -------------------------------------- Model
my_network = net_model.Model(batch_size).createModel()
my_network.summary()
### ---------------- Training and Evaluation part 
my_network.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer = tf.keras.optimizers.RMSprop(),
    metrics = ["accuracy"]
)

history = my_network.fit(tf_training_data[0],tf_training_data[1], batch_size=batch_size, epochs=2, validation_split=0.2)


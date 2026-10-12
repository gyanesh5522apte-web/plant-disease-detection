import tensorflow as tf

print("TensorFlow version:", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")
print("GPUs found:", gpus)

if len(gpus) > 0:
    print("GPU is available - training will use the M4 chip")
else:
    print("No GPU found - training will use the CPU (slower)")
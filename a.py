import tensorflow as tf
from tensorflow.keras import Model


class MyModel(Model):
    def __init__(self):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(units=4,use_bias=True)

    def call(self,t,v):
        print(t.device)

        z = self.fc1(t)
        input()
        return z

if tf.config.list_physical_devices("GPU"):
    with tf.device("GPU:0"):
        # model = MyModel()
        x1 = tf.random.uniform((1000,190))
        x2 = tf.random.uniform((1000,4))
        
        a = tf.data.Dataset.from_tensor_slices((x1,x2)).batch(32)
        for batch in a:

            print(batch[0].device)
        # for batch in a:
        #     x,y = batch
        #     print(x.get_shape())
        #     l = model(x,y)




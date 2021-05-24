# Tensorflow C/CPP object detection using SavedModel

1. Download C_Api from (https://www.tensorflow.org/install/lang_c)
2. Open Visual Studio and Link header and library file
3. Also Configure OpenCV to Visual Studio
## Add source example "Test.cpp"

## Tensorflow C_API GPU memory alocation.
1. Find serialized protobuf string using python
```
import tensorflow as tf  
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.33, 
                            allow_growth=True,
                            visible_device_list='0')
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
serialized = config.SerializeToString()
list(map(hex, serialized))
print(list(map(hex, serialized)))
```
Output will be as follows
```
[ 0x32, 0xe, 0x9, 0x1d, 0x5a,0x64, 0x3b, 0xdf, 0x4f, 0xd5, 0x3f, 0x20, 0x1, 0x2a, 0x1, 0x30 ]
```
2. Set the serialized protobuf string to C_API code

```
graph = TF_NewGraph();
TF_Status* Status = TF_NewStatus();
TF_SessionOptions* SessionOpts = TF_NewSessionOptions();
uint8_t config[16] = { 0x32, 0xe, 0x9, 0x1d, 0x5a,0x64, 0x3b, 0xdf, 0x4f, 0xd5, 0x3f, 0x20, 0x1, 0x2a, 0x1, 0x30 };
TF_SetConfig(SessionOpts, (void*)config, 16, Status);
```

* Ref (https://github.com/Neargye/hello_tf_c_api) (https://github.com/danishshres)

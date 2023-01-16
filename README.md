# MobileNet

This repository provides support for inference with tflite weights of mobilenet.

**Important**: This is an experimental repository and it is not attached or included to any of our production ready repositories.

### Model Settings
```yaml
iou_thresh: 0.4, # Float Intersection of Union threshold
conf_thresh: 0.5, # Float class confidence threshold
```


### Model Config
```yaml
input_shape:
filter_class_ids:
model_path:
```

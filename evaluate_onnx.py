from fast_plate_ocr import ONNXPlateRecognizer

import time

m = ONNXPlateRecognizer(model_path='./exported_models/model_858_96.onnx',config_path='./config.yaml')
start_time=time.time()
print(m.run('./cropped_license_plates_zero_padding/track0092/track0092_01.jpg'))
end_time=time.time()
duration= 1000*(end_time-start_time)
print(duration)
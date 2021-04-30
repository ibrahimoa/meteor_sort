import tflite_runtime
from tflite_runtime.interpreter import Interpreter
import numpy as np
import time
from PIL import Image, ImageOps
import os
from os.path import join

print('Imports done succesfully!')

def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image
  
def classify_image(interpreter, image, top_k=1):
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  return np.squeeze(interpreter.get_tensor(output_details[0]['index'])) #['index']

# Load the model:
interpreter = Interpreter('meteorLiteModel.tflite')
print('Model loaded succesfully!')

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
print(input_details)
output_details = interpreter.get_output_details()
print(output_details)


# Classify the images.
# Total 6252 non meteors and 7140 meteors
time_start = time.time()

data_folder = join(os.getcwd(), 'data')
total_output: float = 0.0
total_classification_time: float = 0.0
total_files : int = 0
true_positives: int = 0
false_positives: int = 0
true_negatives: int = 0
false_negatives: int = 0

for root, dirs, files in os.walk(data_folder):
  for file in files:
    # Load an image to be classified:
    image = Image.open(join(root,file)).convert("L").resize((256, 256))
    image_array = np.array(image)
    image_array = np.true_divide(image_array, 255)
    image_array = image_array.reshape(256, 256, 1)
    # Classify the image:
    time1 = time.time()
    output = classify_image(interpreter, image_array.astype('float32'))
    time2 = time.time()
    if file.startswith('n'): # non_meteor (1)
      if output >= 0.5:
          true_negatives += 1
      else:
          false_positives += 1
    else: # meteor (0)
      if output < 0.5:
          true_positives += 1
      else:
          false_negatives += 1
          
    total_classification_time += np.round(time2-time1, 4)
    total_output += output
    total_files += 1
    
time_end = time.time()

print(f'\nTotal files classified: {total_files}')
print(f'True positivies : {true_positives}')
print(f'False positivies: {false_positives}')
print(f'True negatives  : {true_negatives}')
print(f'False negatives : {false_negatives}')
print(f'\n##################################\n')

precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1 = 2*recall*precision / (recall + precision)
print(f'Precision: {np.round(precision*100, 2)}')
print(f'Recall  : {np.round(recall*100, 2)}')
print(f'F1 score : {np.round(f1, 3)}')
print('\n##################################\n')
# Meteors output should be 0. Non meteors's should be 1.
#print('Output sum: ', np.round(total_output, 2))
#print('Total meteors: 7140')
#print('Total non-meteors: 6252')
total_time = np.round(time_end-time_start, 2)
total_classification_time = np.round(total_classification_time, 2)

print(f"Total time = {total_time} s")
print(f"Total classificaiton time = {total_classification_time} s")
try:
  average_time = np.round(total_time / total_files * 1000, 2)
except:
  average_time = 0
try:
  average_time_classify = np.round(total_classification_time / total_files * 1000, 2)
except:
  average_time_classify = 0
print(f"Average time : {average_time} ms")
print(f"Average classificaiton time : {average_time_classify} ms")
  
  
  
  
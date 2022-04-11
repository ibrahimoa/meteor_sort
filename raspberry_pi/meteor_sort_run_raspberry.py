import tflite_runtime
from tflite_runtime.interpreter import Interpreter
import numpy as np
import time
from PIL import Image, ImageOps
import os
from os.path import join

# Log that we manage to import everything correctly.
print('log: imports done successfully!')


def set_input_tensor(interpreter, image):
    """Feed the interpreter with the image to be classified (input data)."""
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=1):
    """Classify the image passed according to the specified interpreter."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    return np.squeeze(interpreter.get_tensor(output_details[0]['index']))


if __name__ == "__main__":
    # Load the model.
    interpreter = Interpreter('meteor_sort_tflite_model.tflite')
    print('log: model loaded successfully!')

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    print(f"log: input data details -> {input_details}")
    output_details = interpreter.get_output_details()
    print(f"log: output data details -> {output_details}")

    # Set the initial value of some variables.
    data_folder = join(os.getcwd(), 'data')
    total_output: float = 0.0
    total_classification_time: float = 0.0
    total_images: int = 0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    # Classify the images and measure the time that it takes.
    time_start = time.time()
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            # Load an image to be classified:
            image = Image.open(join(root, file)).convert("L").resize((256, 256))
            image_array = np.array(image)
            # we need to do this because the images were reshaped also for training.
            image_array = np.true_divide(image_array, 255)
            image_array = image_array.reshape(256, 256, 1)
            # Classify the image.
            time_load = time.time()
            output = classify_image(interpreter, image_array.astype('float32'))
            time_classify = time.time()
            # Check if the result is correct and store it.
            if file.startswith('n'):  # non_meteor (1)
                if output >= 0.5:
                    true_negatives += 1
                else:
                    false_positives += 1
            else:  # meteor (0)
                if output < 0.5:
                    true_positives += 1
                else:
                    false_negatives += 1

            total_classification_time += np.round(time_classify - time_load, 4)
            total_output += output
            total_images += 1
    time_end = time.time()

    # Log the basic results (total images and confusion matrix).
    print(f'\nlog: total images classified = {total_images}')
    print(f'log: true positives = {true_positives}')
    print(f'log: false positives = {false_positives}')
    print(f'log: true negatives  = {true_negatives}')
    print(f'log: false negatives = {false_negatives}\n')

    # Log some more sophisticated performance measures.
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * recall * precision / (recall + precision)
    print(f'\nlog: precision = {np.round(precision * 100, 2)}')
    print(f'log: recall = {np.round(recall * 100, 2)}')
    print(f'log: F1 score = {np.round(f1, 3)}\n')

    # Calculate the total time that took to classify all the images (in s).
    total_time = np.round(time_end - time_start, 2)
    total_classification_time = np.round(total_classification_time, 2)
    print(f"\nlog: total time = {total_time} s")
    print(f"log: total classification time = {total_classification_time} s\n")

    # Calculate the average time that classify an image takes (in ms).
    average_time = np.round(total_time / total_images * 1000, 2)
    average_time_classify = np.round(total_classification_time / total_images * 1000, 2)
    print(f"\nlog: average time = {average_time} ms")
    print(f"log: average classification time = {average_time_classify} ms")

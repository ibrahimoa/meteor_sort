# Meteor classification using Neural Networks

> Author: Ibrahim Oulad Amar

Final degree project which consists of creating a meteor classification system using Deep Learning techniques.

The classifier uses 256x256 images as input. The images shall be the MAXPIXEL ones in the FTP format. In this case
I used the data provided by the University of Western Ontario. I am thankful to researcher Denis Vida for his help
in obtaining these data.

The data was split in two sets, training (85%) and validation (15%). The model is a CNN + MaxPool + BatchNormalization
(total 12 layers) along with 3 fully connected layers. The total number of parameters is 49,449, of which 512 are 
not-trainable. The model performance metrics are:

- Model Precision: 0.931 (93.1%)
- Model Recall: 0.941 (94.1%)
- Model F1 Score: 0.936

The model is the one defined in the file `final_model_weights.h5` in the folder 
`meteor_sort/results/weights/`.

Finally, you can find the thesis document in the root of the project with the name `TFG_Oulad_Amar_Ibrahim.pdf` or read it directly from [here](https://github.com/ibrahimoa/meteor_sort/blob/main/TFG_Oulad_Amar_Ibrahim.pdf).

> If you have any suggestions or questions don't hesitate to reach me at my email: **ibrahim.ouladamar@gmail.com**

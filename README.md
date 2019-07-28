## Find Wally

### Intro

The children book series [Where's Wally?](https://en.wikipedia.org/wiki/Where%27s_Wally%3F) first appeared in Great Britain in 1987. Each book is full of different pictures. Each picture represents a different scene with dozens of different characters. One of which is Wally. The goal of this "game" is to find Wally.

Original Books             |  Random Scene
:-------------------------:|:-------------------------:
<img src="books.jpg" alt="drawing" width="400"/>  |  <img src="https://media.buzz.ie/uploads/2016/06/Zalando-Festival-Map-No-Title-Final-2.jpg" alt="drawing" width="400"/>

With this project, we wanted to build software which helps to identify Wally in just a few seconds by using computer vision and machine learning. 
<img src="wip.jpg" alt="drawing" width="400"/>

Two approaches were implemented and compared:
1. [Find Wally](https://github.com/TyLeRRR/find-wally/tree/master/autoML) with Google AutoML Vision.
2. [Find Wally](https://github.com/TyLeRRR/find-wally/tree/master/handmade) with Tensorflow object detection model *Faster RCNN Inception v2* & OpenCV.

### Tensorflow + OpenCV Implementation
We use the open-source machine learning library Tensorflow, namely - pre-trained model *Faster RCNN Inception v2*. With the help of OpenCV, we are streaming a video via the webcam and use the model to find Wally.

#### Dataset Preparation
To build the training set, we have to create a series of Where's Wally puzzle pictures with the spots where Wally appears. We used two Wally books and photographed them entirely with different resolutions and image qualities. We collected ~200 images. 20% of which were used for the evaluation. 

Our next step was to label all the pictures. Using the [LabelImg](https://github.com/tzutalin/labelImg) application, we created the XML files with x and y coordinates with Wally object for each image. With the help of `xml_to_csv.py`, we convert all XML to a CSV files. 
The training CSV and the training dataset must be packed into a `.tfrecord` file. For that, we used `create_tf_record.py`. The same happens with the evaluation CSV and the evaluation dataset. Now we have two `.tfrecord` files: ` train.record` and `eval.record`.

#### Model Preparation
We used the *RCNN Inception v2* model, which was trained on the COCO dataset. To tune it for our needs, we use a pipeline configuration script - `faster_rcnn_inception_v2_coco.config`. 
The following parameters must be adjusted:
- `fine_tune_checkpoint` - path to the checkpoints in GC Bucket.
- `num_steps` - number of iterations, in our case 200 000.
- `input_path` - for `train_input_reader` (path to train.record file).
- `input_path` - for `eval_input_reader` (path to eval.record file).
- `num_examples` - evaluation dataset size.
- `max_evals` - number of iterations for each evaluation.

The last but not least we have to configure `labels.txt` map. It contains the list of labels for our objects. Since we are only looking for one object type, this file consists of only one Wally object.

#### Training
For the training, we used Google Cloud. The `.tfrecord` files and other configs were uploaded into the bucket so that the training job can use them for training. The general rule was to end the training when the loss on our evaluation set is no longer decreasing or is generally very small (in our case below 0.01).
After a few hours and 70 000 iterations, we got a model that can identify Wally on the new photos.

### GC AutoML Vision Implementation
Our first approach was to use OpenCV for finding and cutting out all the faces in an image. Afterward, send these them to the Google Cloud for analysis.
But the problem here was that the facial recognition algorithms, provided by OpenCV, weren't compatible with the faces found in Wally books. In our tests with different variations of the Hair Cascade Classifier algorithm (for facial recognition), not a single face was detected. This is most likely because we had a cartoonish style that does not have the characteristics of an average human face.  
Finally, we decided to take another approach - `predict_wally.py`. 

The implementation consists of two parts: cut the image into smaller pieces & send them to the Google AutoML service. 

Splitting the image into smaller pictures is required so that the service can respond with the information about: How high is the probability that Wally is present in this particular section of the image? Otherwise, if we sent the whole picture to the service, we would have received whether Wally is present on the foto or not. After all, Waldo is always there. Moreover, the result would be very inaccurate due to the complexity of the picture. Splitting the image is done by using OpenCV in the `crop_image()`. As arguments, the function requires the image path and the size of the sections. The cropped images are stored in the `cropped_images` folder.

In the second part, the images are sent via API to the Google Cloud. `get_prediction()` takes an image as an argument and is called for each previously cut image. The `predict()` requires: the model ID, a payload (an object with the cut image), and eventual parameters (in our case `score_threshold`, which was set to 0.0 to filter no values).

### Final Results
We have succeeded to build the model that finds Wally via the webcam video feed and on the webcam quality photos with the 85% accuracy.













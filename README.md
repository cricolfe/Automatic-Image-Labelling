# Automatic-Image-Labelling
The aim of this toolbox is to label images automatically. Images are captured under controlled background and objects are labelled easily using 2D computer vision techniques. Afterwards, detected objects are combined with images of standard backgrounds to create a rich image set to train deep models. 
Example is done using images that contains hands with fingers. Hands (objects) are labelled under five clases (one, thwo, three, four, five)
Results are ready to train a deep learning model using Tensorflow.

### Prerequisites:
- opencv-python
- libconf
- python-tk
- pandas
- matplotlib
- :coffee: 
- :pizza:

## Step 1
Take images using cCapture.py with a webcam for example. 

Organize all images into classses according with the different sets you want to identify. To organize images with hands folders with names "one", "two", "three", "four", "five" are created. Each folder has images of one class. There is also a folder with images of backgrounds to combine with images of the classes called "backgrounds". If this folder is empty, original images are not combined with backgrounds. It is necessary also an image of the backgound without objects to threshold pixels images and remove background. This image is called "imbk.jpg". Results will be stored in another folder called "results".  Following figure shows the set of folders to organize all images in classes and compute results.

![CVAT screenshot](cvat/apps/documentation/static/documentation/images/cvat.jpg)

## Documentation

-   [User's guide](cvat/apps/documentation/user_guide.md)
-   [XML annotation format](cvat/apps/documentation/xml_format.md)
-   [AWS Deployment Guide](cvat/apps/documentation/AWS-Deployment-Guide.md)
-   [Questions](#questions)

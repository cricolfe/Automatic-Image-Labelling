![Cover](documentation/cover.png)

Manual image objects labelling is a very tedious and time consuming task. The aim of this toolbox is to facilitate the initial step of creating a set of labelled objects in images to train deep models. Here, images are captured under controlled background and objects are labelled easily using 2D computer vision techniques. Afterwards, detected objects are combined with images of standard backgrounds to create a rich image set to train deep models. 
Example is done using images that contains hands with fingers. Hands (objects) are labelled under five clases (one, thwo, three, four, five)
Results are ready to train a deep learning model using Tensorflow.

**Prerequisites:**
- opencv-python
- libconf
- python-tk
- pandas
- matplotlib
- :coffee: 
- :pizza:

## Step 1
Take images using cCapture.py with a webcam for example. It takes one image every 200 ms. Background of images have to be flat with no textures and color have to be different of objects to label. You can use a cardboard as background. A clear example is the image under the title.

Organize all images into classses according with the different sets you want to identify. To organize images with hands, folders with names "one", "two", "three", "four", "five" are created. Each folder has images of one class. There is also a folder with images of backgrounds to combine with images of classes called "backgrounds". If this folder is empty, original images are not combined with backgrounds. It is necessary also an image of the backgound without objects to threshold pixels images and remove background. This image is called "imbk.jpg". Results will be stored in another folder called "results".  Following figure shows the set of folders to organize all images in classes and compute results. In folder images\ you have this structure with all images.

![Folders](documentation/folders.png)

## Step 2
Configure image labelling task using file "cFile.cfg". It is important to configure the HSV channel to threshold the image. It should be 0 for the Hue layer where colours are defined and 1 for Saturation layer. Background is removed as follow. First using the image  of background "imbk.jpg", the mean value for H or S channel is computed according with the selected channel for thresholding. This value together with the threshold value establishes a band of background pixels. This value is used to threshold incoming images. All image labelling parameters are shown in the following image.

![Parameters](documentation/parameters.png)

## Step 3
Run the program cAnalisys.py with the configuration file cFile.cfg

With parameter WaitTime set to 0, you can see how the algortihm is working with every image just pushing one key. With the HSV decomposition images, you can see which layer differentiate object pixels and background pixels better. In this case, H layer is the best one. With this information reconfigure the HSV channel and threshold to select pixels.

![Algorithm](documentation/algorithm.png)

## Documentation

-   [User's guide](cvat/apps/documentation/user_guide.md)
-   [XML annotation format](cvat/apps/documentation/xml_format.md)
-   [AWS Deployment Guide](cvat/apps/documentation/AWS-Deployment-Guide.md)
-   [Questions](#questions)

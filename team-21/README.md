Image Colorization
Team 21
Sean Bremer and Aidan Szilagyi

Overview:
Image Colorization, which is the task of taking a greyscale image and
inferring realistic color channels, can be used for image restoration
and enhancement. We use a U-net style encoder-decoder architecture with
ResNet-34 as the endcoder, and used the COCO dataset to train a 
convolutional decoder and created a convolutional GAN for adversarial
training. We tried a few different loss functions, and achieved a SSIM
of 0.42 with our best model.

Setup and Usage:
Import the datasets from http://images.cocodataset.org/zips/train2017.zip
and http://images.cocodataset.org/zips/val2017.zip and unzip. Simply run
the code blocks from top to bottom for best results.

Video:
Our video appears on slide 13 of our powerpoint:
https://myuva-my.sharepoint.com/:p:/g/personal/utw5es_virginia_edu/IQBKualfcb7LSYgeoj0j1Ww0ARdaqcZu8mnCsosx3DE0vdI?e=xObXdm

Overview: A brief introduction to the project.
Usage: How to run the code to get core results.
(Optional) Setup: Instructions for environment setup (if non-trivial).
(Optional) Video: A link to your demo video with a brief description.
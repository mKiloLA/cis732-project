# cis732-project
This is a repository for the final project of CIS 732: Machine Learning and Pattern Recognition. 

## Introduction
With about half of all livestock’s feed in the US coming from hay and the world’s population set
to reach 10 billion by the end of the century, the production of high-quality hay bales has never
been more important [1] [2]. The demand to increase efficiency with already cultivated land to
feed the world’s growing population has resulted in the rise of precision agriculture as a
significant growth industry. In order to increase efficiency in feed operations, classifying good
and bad hay bales could reduce waste and man-hours spent on repairing bad bales by integrating
computer vision systems with baling equipment to provide real-time machine adjustment. To that
end we would like to implement a bale classification system with the latest version of the Yolo
library using several methods of class imbalance correction.

## Background and Related Work
The data set used for this project was aggregated by Ben Vail for his master’s research. Due to
the quality of hay balers, significantly more good hay bales appear in the data set than bad hay
bales. This has led to significant class imbalance. Ben Vail implemented a classifier using YOLO
v5, but no considerations were made for the class imbalance.

When it comes to class imbalance fixes, three main categories typically get discussed: data-level,
algorithmic-level, and hybrid [3]. Data-level class imbalance fixes rely on altering the input data,
typically by oversampling the minority class or under sampling the majority class [3]. Examples
include ROS, RUS, two-phase learning, and dynamic sampling. Algorithmic-level fixes rely on
adjusting the algorithm itself rather than the data. For example, creating new loss functions,
creating cost table, or adjusting the weights [3]. Hybrid-methods exactly as expected by
combining data and algorithmic-level class imbalance fixes.

## Methodology
This project will utilize the You Only Look Once (YOLO) object detection model developed by
Ultralytics [4]. This model utilizes deep learning techniques to detect and classify objects in
pictures and videos. A baseline model will be created from scratch with YOLO v8 before a class
imbalance fix is added. Another initial test will be done using models pretrained on COCO and
ImageNet for comparison. I will be using a data-level, image generation method called
StyleNAT to generate more images of bad hay-bales [5]. The model will then be retrained with
YOLO v8 and the performance metrics will be compared with the baseline model.

## Evaluation Criteria
The evaluation of an imbalanced dataset can be tricky due the skewed distribution. A model can
be 99% accurate while misclassifying the minority class if the data is skewed 100:1. As such, we
will use precision, recall, and selectivity alongside accuracy and error rate to determine the
effectiveness of the imbalance fixes. The statistical analysis and graphing will be handled by
wandb.ai which can interface with Yolo to easily understand how effective the training was [5]

## Results
After creating images using DC GAN (full implementation included), a new model was created using
Yolov8. When trained and test on an 80/10/10 split from the original dataset plus the new images,
there was no significant improvement. When tested on a set of bale images from the internet, the
new model performance slightly worse than the original, COCO trained baseline.

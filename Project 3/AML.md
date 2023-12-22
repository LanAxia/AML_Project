In this project, we use Deep Learning to solve the Image Segmentation Problem. We tried different Data Preprocessing and Computer Vision Models to solve this problem and finally found that the UNet Model suits this project.

In this project, I am mainly responsible for organising the experiment, coding for the main Deep Learning Models and coding for part of Data Preprocessing and Prediction.

In the Data Preprocessing, we tried various sampling methods to limit the input data to the same dimension. Finally, we fix the input dimension to (112, 112) and use a slide window to sample the data. We also tried different ways of data augmentation, like rescaling and Cropping.

In the Estimator training, we tried different image segmentation models like UNet, UNet++ and various variants of UNet++, and finally found that the UNet with bilinear upsampling has the best performance on this project. Finally, we average the prediction results of different models as the final prediction result.

We also tried some post-processing of the prediction. But they didn't improve the prediction result.
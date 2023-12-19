from utils import load_zipped_pickle, save_zipped_pickle, single2tensor4, uint2single, single2uint, tensor2uint, test_onesplit
import torch
import numpy as np
from scipy import ndimage
from src.drunet import UNetRes
import cv2
import matplotlib.pyplot as plt

class PreProcessing:
    def __init__(self, model_path, train_path, test_path, save_path, resize_img_size=672, img_size=(512, 512), noise=5, device='cuda'):
        self.train_path = train_path
        self.test_path = test_path
        self.save_path = save_path
        self.resize_img_size = resize_img_size
        self.target_img_size = img_size
        self.noise = torch.FloatTensor([noise / 255.])

        self.train_data = load_zipped_pickle(self.train_path)
        self.test_data = load_zipped_pickle(self.test_path)

        self.device = device
        self.model = UNetRes(in_nc=2,
                             out_nc=1,
                             nc=[64, 128, 256, 512],
                             nb=4,
                             act_mode='R',
                             downsample_mode="strideconv",
                             upsample_mode="convtranspose"
                             )
        self.model.load_state_dict(torch.load(model_path), strict=True)
        self.model.eval()
        self.model = self.model.to(device)
        
    def get_image_data(self, data: list):
        images = []
        for index in range(len(data)):
            frames = data[index]['frames']
            images.append(data[index]['video'][:, :, frames])
        return images
    
    def get_label_data(self, data: list):
        labels = []
        for index in range(len(data)):
            frames = data[index]['frames']
            labels.append(data[index]['label'][:, :, frames])
        return labels

    def resize_image(self, images: np.ndarray, labels: np.ndarray, box: np.ndarray):
        labels = np.where(labels, 1, 0)
        boxes = np.where(box, 1, 0)
        zoom_factor = self.resize_img_size / images.shape[0]

        images = ndimage.zoom(images, (zoom_factor, zoom_factor, 1))
        labels = ndimage.zoom(labels, (zoom_factor, zoom_factor, 1))
        boxes = ndimage.zoom(boxes, (zoom_factor, zoom_factor))
        return images, labels, boxes

    def crop_image(self, images: np.ndarray, labels: np.ndarray, box: np.ndarray):
        height = self.target_img_size[0]
        width = self.target_img_size[1]
        shape = images.shape[0:2]

        left = shape[1]//2-width//2
        right = shape[1]//2+width//2
        top = shape[0]//2-height//2
        bottom = shape[0]//2+height//2

        images = images[top:bottom, left:right, :]
        labels = labels[top:bottom, left:right, :]
        box = box[top:bottom, left:right]
        return images, labels, box, (left, right, top, bottom)

    def generate_training_data(self):
        images = []
        labels = []
        boxes = []

        for index in range(len(self.train_data)):
            frames = self.train_data[index]['frames']
            image = self.train_data[index]['video'][:, :, frames]
            label = self.train_data[index]['label'][:, :, frames]
            box =  self.train_data[index]['box']
            datasets_type = self.train_data[index]['dataset']
            if datasets_type == 'amateur':
                image, label, box = self.resize_image(image, label, box)
            image, label, box, _ = self.crop_image(image, label, box)
            image = self.process_image(image)
            images.append(image)
            labels.append(label)
            boxes.append(box)
            print("Finish processing training data: ", index)
        images = np.array(images)
        labels = np.array(labels)
        boxes = np.array(boxes)
        print("Finish generating training data")
        data = {'images': images, 'labels': labels, 'boxes': boxes}
        save_zipped_pickle(data, self.save_path + "train_crop.pkl")

    def generate_testing_data(self):
        images = []
        shapes = []
        for index in range(len(self.test_data)):
            image = self.test_data[index]['video']
            label = np.zeros(image.shape)
            box = np.zeros((image.shape[0], image.shape[1]))

            image, _, _, boundary = self.crop_image(image, label, box)
            image = self.process_image(image)
            images.append(image)
            shapes.append(boundary + (image.shape[0], image.shape[1]))
            print("Finish processing testing data: ", index)
        data = [{'images': images[i], 'shapes': shapes[i]} for i in range(len(shapes))]
        save_zipped_pickle(data, self.save_path + "test_crop.pkl")

    def process_image(self, images: np.ndarray):
        for index in range(images.shape[2]):
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            image = images[:, :, index]
            image = clahe.apply(image)
            image = ndimage.gaussian_filter(image, sigma=1.5)
            image = cv2.medianBlur(image, 5)
            image = self.denoise_image(image)
            image = self.erase_noise(image)
            images[:, :, index] = image
        return images

    def denoise_image(self, image: np.ndarray):
        image = np.expand_dims(image, axis=2)
        image = uint2single(image)
        image = single2tensor4(image)
        image = torch.cat((image, 
                           self.noise.repeat(1, 1, image.shape[2], image.shape[3])), 
                           dim=1)
        padded_image = image.to(self.device)
        denoised_img = test_onesplit(self.model, padded_image, refield=32)
        denoised_img = tensor2uint(denoised_img)
        return denoised_img
    
    def erase_noise(self, image: np.ndarray):
        height, width = image.shape
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_img = np.zeros((height, width, 3), np.uint8)
        min_patch_size = 200
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if contour_area < min_patch_size:
                cv2.drawContours(contour_img, [contour], 0, (0, 0, 255), 2)
                mask = np.zeros((height, width), np.uint8)
                cv2.drawContours(mask, [contour], 0, 255, -1)
                image = cv2.inpaint(image, mask, 2, cv2.INPAINT_TELEA)
        return image
    

if __name__ == "__main__":
    model_path = "./checkpoints/drunet/drunet_gray.pth"
    train_path = "./data/train.pkl"
    test_path = "./data/test.pkl"
    save_path = "./"
    pre_processing = PreProcessing(model_path, train_path, test_path, save_path)
    pre_processing.generate_testing_data()
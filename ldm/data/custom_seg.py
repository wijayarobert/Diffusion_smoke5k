import os
import numpy as np
import cv2
import albumentations
from PIL import Image
from torch.utils.data import Dataset

# class SegmentationBase(Dataset):
#     def __init__(self,
#                  data_txt, segmentation_root, data_root,
#                  size=None, random_crop=False, interpolation="bicubic"):
#         self.data_txt = data_txt
#         self.segmentation_root = segmentation_root
#         with open(self.data_txt, "r") as f:
#             self.data_root = f.read().splitlines()
#         self._length = len(self.data_root)

#         size = None if size is not None and size <= 0 else size
#         self.size = size
#         if self.size is not None:
#             self.interpolation = interpolation
#             self.interpolation = {
#                 "nearest": cv2.INTER_NEAREST,
#                 "bilinear": cv2.INTER_LINEAR,
#                 "bicubic": cv2.INTER_CUBIC,
#                 "area": cv2.INTER_AREA,
#                 "lanczos": cv2.INTER_LANCZOS4}[self.interpolation]
#             self.image_rescaler = albumentations.SmallestMaxSize(max_size=self.size,
#                                                                  interpolation=self.interpolation)
#             self.center_crop = not random_crop
#             if self.center_crop:
#                 self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
#             else:
#                 self.cropper = albumentations.RandomCrop(height=self.size, width=self.size)
#             self.preprocessor = self.cropper

#     def __len__(self):
#         return self._length

#     # def __getitem__(self, i):
#     #     file_path = os.path.join(self.segmentation_root, self.data_root[i])
#     #     image = Image.open(file_path)
#     #     if not image.mode == "RGB":
#     #         image = image.convert("RGB")
#     #     image = np.array(image).astype(np.uint8)
#     #     if self.size is not None:
#     #         image = self.image_rescaler(image=image)["image"]
#     #         processed = self.preprocessor(image=image)
#     #     else:
#     #         processed = {"image": image}
#     #     processed["image"] = (processed["image"] / 127.5 - 1.0).astype(np.float32)
#     #     return processed

#     def __getitem__(self, i):
#         rgb_file_path = os.path.join(self.data_root, self.data_root[i])
#         seg_file_path = os.path.join(self.segmentation_root, self.segmentation_root[i])

#         # Read and preprocess the RGB image
#         rgb_image = Image.open(rgb_file_path)
#         if not rgb_image.mode == "RGB":
#             rgb_image = rgb_image.convert("RGB")
#         rgb_image = np.array(rgb_image).astype(np.uint8)
#         if self.size is not None:
#             rgb_image = self.image_rescaler(image=rgb_image)["image"]
#             processed_rgb = self.preprocessor(image=rgb_image)
#         else:
#             processed_rgb = {"image": rgb_image}
#         processed_rgb["image"] = (processed_rgb["image"] / 127.5 - 1.0).astype(np.float32)

#         # Read and preprocess the segmentation image
#         seg_image = Image.open(seg_file_path)
#         if not seg_image.mode == "RGB":
#             seg_image = seg_image.convert("RGB")
#         seg_image = np.array(seg_image).astype(np.uint8)
#         if self.size is not None:
#             seg_image = self.image_rescaler(image=seg_image)["image"]
#             processed_seg = self.preprocessor(image=seg_image)
#         else:
#             processed_seg = {"image": seg_image}
#         processed_seg["image"] = (processed_seg["image"] / 127.5 - 1.0).astype(np.float32)

#         # Return both images in the form of a dictionary to match the 'cond' input of apply_model
#         return {'x_noisy': processed_seg, 'cond': {'c_concat': [processed_rgb]}}


# class CustomSegTrain(SegmentationBase):
#     def __init__(self, size=256, random_crop=False, interpolation="bicubic"):
#         super().__init__(data_txt='/home/remote/u7177316/taming-transformers/data/smoke5k_train/xx_train.txt',
#                          data_root='/home/remote/u7177316/dataset/smoke5k/train/img',
#                          segmentation_root='/home/remote/u7177316/dataset/smoke5k/train/gt',
#                          size=size, random_crop=random_crop, interpolation=interpolation)

# class CustomSegEval(SegmentationBase):
#     def __init__(self, size=256, random_crop=False, interpolation="bicubic"):
#         super().__init__(data_txt='/home/remote/u7177316/taming-transformers/data/smoke5k_test/xx_test.txt',
#                          data_root='/home/remote/u7177316/dataset/smoke5k/test/img',
#                          segmentation_root='/home/remote/u7177316/dataset/smoke5k/test/gt',
#                          size=size, random_crop=random_crop, interpolation=interpolation)

class SegmentationBase(Dataset):
    def __init__(self,
                 rgb_txt, seg_txt, size=None, 
                 random_crop=False, interpolation="bicubic"):
        self.rgb_txt = rgb_txt
        self.seg_txt = seg_txt
        with open(self.rgb_txt, "r") as f:
            self.rgb_paths = f.read().splitlines()
        with open(self.seg_txt, "r") as f:
            self.seg_paths = f.read().splitlines()
        assert len(self.rgb_paths) == len(self.seg_paths), "RGB and segmentation images must be paired"
        self._length = len(self.rgb_paths)

        size = None if size is not None and size <= 0 else size
        self.size = size
        if self.size is not None:
            self.interpolation = interpolation
            self.interpolation = {
                "nearest": cv2.INTER_NEAREST,
                "bilinear": cv2.INTER_LINEAR,
                "bicubic": cv2.INTER_CUBIC,
                "area": cv2.INTER_AREA,
                "lanczos": cv2.INTER_LANCZOS4}[self.interpolation]
            self.image_rescaler = albumentations.SmallestMaxSize(max_size=self.size,
                                                                 interpolation=self.interpolation)
            self.center_crop = not random_crop
            if self.center_crop:
                self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size, width=self.size)
            self.preprocessor = self.cropper

    def __len__(self):
        return self._length
    
    def __getitem__(self, i):
        rgb_file_path = self.rgb_paths[i]
        seg_file_path = self.seg_paths[i]

        # Read and preprocess the RGB image
        rgb_image = Image.open(rgb_file_path)
        if not rgb_image.mode == "RGB":
            rgb_image = rgb_image.convert("RGB")
        rgb_image = np.array(rgb_image).astype(np.uint8)

        if self.size is not None:
            rgb_image = self.image_rescaler(image=rgb_image)["image"]
            processed_rgb = self.preprocessor(image=rgb_image)
        else:
            processed_rgb = {"image": rgb_image}
        processed_rgb["image"] = (processed_rgb["image"] / 127.5 - 1.0).astype(np.float32)

        # Read and preprocess the segmentation image
        seg_image = Image.open(seg_file_path)
        if not seg_image.mode == "RGB":
            seg_image = seg_image.convert("RGB")
        seg_image = np.array(seg_image).astype(np.uint8)

        if self.size is not None:
            seg_image = self.image_rescaler(image=seg_image)["image"]
            processed_seg = self.preprocessor(image=seg_image)
        else:
            processed_seg = {"image": seg_image}
        processed_seg["image"] = (processed_seg["image"] / 127.5 - 1.0).astype(np.float32)

        # Return both images in the form of a dictionary to match the 'cond' input of apply_model
        return {'segmentation': processed_rgb['image'], 'image': processed_seg['image']}




    
    # def __getitem__(self, i):
    #     rgb_file_path = self.rgb_paths[i]
    #     seg_file_path = self.seg_paths[i]

    #     # Read and preprocess the RGB image
    #     rgb_image = Image.open(rgb_file_path)
    #     if not rgb_image.mode == "RGB":
    #         rgb_image = rgb_image.convert("RGB")
    #     rgb_image = np.array(rgb_image).astype(np.uint8)

    #     if self.size is not None:
    #         rgb_image = self.image_rescaler(image=rgb_image)["image"]
    #         processed_rgb = self.preprocessor(image=rgb_image)
    #     else:
    #         processed_rgb = {"image": rgb_image}
    #     processed_rgb["image"] = (processed_rgb["image"] / 127.5 - 1.0).astype(np.float32)

    #     # Read and preprocess the segmentation image
    #     seg_image = Image.open(seg_file_path)
    #     if not seg_image.mode == "RGB":
    #         seg_image = seg_image.convert("RGB")
    #     seg_image = np.array(seg_image).astype(np.uint8)

    #     if self.size is not None:
    #         seg_image = self.image_rescaler(image=seg_image)["image"]
    #         processed_seg = self.preprocessor(image=seg_image)
    #     else:
    #         processed_seg = {"image": seg_image}
    #     processed_seg["image"] = (processed_seg["image"] / 127.5 - 1.0).astype(np.float32)

    #     # Return both images in the form of a dictionary to match the 'cond' input of apply_model
    #     return {'image': processed_rgb['image'], 'cond': {'c_concat': [processed_seg['image']]}}


    # def __getitem__(self, i):
    #     # import matplotlib.pyplot as plt
    #     rgb_file_path = self.rgb_paths[i]
    #     seg_file_path = self.seg_paths[i]

    #     # Read and preprocess the RGB image
    #     rgb_image = Image.open(rgb_file_path)
    #     if not rgb_image.mode == "RGB":
    #         rgb_image = rgb_image.convert("RGB")
    #     rgb_image = np.array(rgb_image).astype(np.uint8)
    #     print(f"RGB image shape: {rgb_image.shape}, range: {rgb_image.min()}-{rgb_image.max()}") # print shape and range
    #     # plt.imsave(f'rgb_{i}.png', rgb_image) # save image

    #     if self.size is not None:
    #         rgb_image = self.image_rescaler(image=rgb_image)["image"]
    #         processed_rgb = self.preprocessor(image=rgb_image)
    #     else:
    #         processed_rgb = {"image": rgb_image}
    #     processed_rgb["image"] = (processed_rgb["image"] / 127.5 - 1.0).astype(np.float32)

    #     # Read and preprocess the segmentation image
    #     seg_image = Image.open(seg_file_path)
    #     if not seg_image.mode == "RGB":
    #         seg_image = seg_image.convert("RGB")
    #     seg_image = np.array(seg_image).astype(np.uint8)
    #     print(f"Segmentation image shape: {seg_image.shape}, range: {seg_image.min()}-{seg_image.max()}") # print shape and range
    #     # plt.imsave(f'seg_{i}.png', seg_image) # save image
        
    #     if self.size is not None:
    #         seg_image = self.image_rescaler(image=seg_image)["image"]
    #         processed_seg = self.preprocessor(image=seg_image)
    #     else:
    #         processed_seg = {"image": seg_image}
    #     processed_seg["image"] = (processed_seg["image"] / 127.5 - 1.0).astype(np.float32)

    #     # Return both images in the form of a dictionary to match the 'cond' input of apply_model
    #     return {'image': processed_seg['image'], 'cond': {'c_concat': [processed_rgb['image']]}}

class CustomSegTrain(SegmentationBase):
    def __init__(self, size=256, random_crop=False, interpolation="bilinear"):
        super().__init__(rgb_txt='/home/remote/u7177316/dataset/smoke5k_4images/train/img/train_img.txt',
                         seg_txt='/home/remote/u7177316/dataset/smoke5k_4images/train/gt/train_gt.txt',
                         size=size, random_crop=random_crop, interpolation=interpolation)

class CustomSegEval(SegmentationBase):
    def __init__(self, size=256, random_crop=False, interpolation="bilinear"):
        super().__init__(rgb_txt='/home/remote/u7177316/dataset/smoke5k_4images/test_new/img/test_img.txt',
                         seg_txt='/home/remote/u7177316/dataset/smoke5k_4images/test_new/gt/test_gt.txt',
                         size=size, random_crop=random_crop, interpolation=interpolation)

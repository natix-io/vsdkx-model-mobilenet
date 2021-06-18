# -*- coding:utf-8 -*-
import cv2
import numpy as np
import tensorflow as tf
from vsdkx.core.interfaces import ModelDriver
from vsdkx.core.structs import Inference
from vsdkx.core.util.model import load_tflite, box_sanity_check


class MobilenetDriver(ModelDriver):
    """
    Class for object detection

    Args:
        model_config (dict): Config dictionary with the following keys:
            'model_path' (str): Path to the tflite model
            'input_shape' (tuple): Shape of input(inference) image
        model_settings (dict): Model settings config with the following keys:
            'target_shape' (tuple): Image target shape
            'iou_thresh' (float): Threshold for Intersection of Union
            'conf_threshold' (float): Confidence threshold
        drawing_config (dict): Debug config dictionary with the following keys:
            'text_thickness' (int): Text thickness
            'text_fontscale' (int): Text font scale
            'text_color' (tuple): Tuple of color in RGB for text
            'rectangle_color' (tuple): Tuple of color in RGB for rectangle
    """

    def __init__(self, model_settings: dict, model_config: dict,
                 drawing_config: dict):
        """

        Args:
            model_config (dict): Config dictionary with the following keys:
                'model_path' (str): Path to the tflite model
                'input_shape' (tuple): Shape of input(inference) image
            model_settings (dict): Model settings config with the
            following keys:
                'target_shape' (tuple): Image target shape
                'iou_thresh' (float): Threshold for Intersection of Union
                'conf_threshold' (float): Confidence threshold
            drawing_config (dict): Debug config dictionary with the
            following keys:
                'text_thickness' (int): Text thickness
                'text_fontscale' (int): Text font scale
                'text_color' (tuple): Tuple of color in RGB for text
                'rectangle_color' (tuple): Tuple of color in RGB for rectangle
        """
        super().__init__(model_settings, model_config, drawing_config)
        self._input_shape = model_config['input_shape']
        self._filter_classes = model_config.get('filter_class_ids', [])
        self._interpreter, self._input_details, self._output_details = \
            load_tflite(model_config['model_path'])
        self._iou_thresh = model_settings['iou_thresh']
        self._conf_thresh = model_settings['conf_thresh']

    def inference(
            self,
            image
    ) -> Inference:
        """
        Driver function for object detection inference

        Args:
            image (np.array): 3D numpy array of input image

        Returns:
            (Inference): the result of the ai
        """

        image_resized = cv2.resize(image,
                                   (
                                   self._input_shape[0], self._input_shape[1]))
        image_np = (2.0 / 255.0) * image_resized - 1.0
        image_exp = np.expand_dims(image_np, axis=0)
        image_exp = tf.cast(image_exp, dtype=tf.uint8)
        # Set img_in as tensor in the model's input_details
        self._interpreter.set_tensor(self._input_details[0]['index'],
                                     image_exp)
        self._interpreter.invoke()

        # Get the output_details tensors (based on the given input above)
        boxes = self._interpreter.get_tensor(
            self._output_details[0]['index'])
        labels = self._interpreter.get_tensor(
            self._output_details[1]['index'])
        scores = self._interpreter.get_tensor(
            self._output_details[2]['index'])
        nums = self._interpreter.get_tensor(
            self._output_details[3]['index'])

        scores = np.squeeze(scores, axis=0)
        labels = np.squeeze(labels, axis=0)

        boxes = self._scale_boxes(boxes, image.shape, nums)

        # keep_idx is the alive bounding box after nms
        keep_idxs = tf.image.non_max_suppression(
            boxes,
            scores, 100,
            self._iou_thresh,
            self._conf_thresh)

        if len(keep_idxs) > 0:
            # Filter nms results
            if len(self._filter_classes) > 0:
                boxes, scores, labels = self._filter_nms(keep_idxs, boxes,
                                                         scores, labels,
                                                         image.shape)
        else:
            boxes, scores, labels = [], [], []

        return Inference(boxes, labels, scores, {})

    def _scale_boxes(self, boxes, target_shape, nums):
        """
        Scales the boxes to the size of the target image

        Args:
            boxes (np.array): Array containing the bounding boxes
            target_shape (tuple): The shape of the target image
            nums (int): Amount of detected bounding boxes
        Returns:
            (np.array): np.array with the scaled bounding boxes
        """
        h, w, _ = target_shape
        boxes = np.squeeze(boxes, axis=0)
        # for box in boxes
        new_b = []
        for i in range(int(nums)):
            x1 = int(boxes[i][1] * w)  # xmin
            y1 = int(boxes[i][0] * h)  # ymin
            x2 = int(boxes[i][3] * w)  # xmax
            y2 = int(boxes[i][2] * h)  # max
            new_b.append([x1, y1, x2, y2])

        return new_b

    def _filter_nms(self, ids, boxes, scores, labels, target_shape):
        """
        Filters the bounding boxes and scores by ID

        Args:
            ids (np.array): 1D array with IDs
            boxes (np.array): Array with bounding boxes
            scores (np.array): Array with class confidences
            labels (np.array): Array with label IDs

        Returns:
            filtered_boxes (list): List with filtered bounding boxes
            filtered_scores (list): List with filtered confidence scores
            filtered_labels (list): List with filtered labels
        """

        filtered_boxes = []
        filtered_scores = []
        filtered_labels = []

        for id in ids:
            # Filter boxes by person class ID
            if labels[id] in self._filter_classes:
                box = box_sanity_check(boxes[id],
                                       target_shape[1],
                                       target_shape[0])
                filtered_boxes.append(box)
                filtered_scores.append(scores[id])
                filtered_labels.append(labels[id])

        return filtered_boxes, filtered_scores, filtered_labels

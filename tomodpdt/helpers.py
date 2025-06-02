import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import rotate
import cv2


class MaskRCNNHandler:
    def __init__(self, device=None, score_threshold=0.3, single_object=True):
        """
        Initializes the Mask R-CNN model and sets up the device.
        
        Parameters:
            model_type (str): The type of Mask R-CNN model to use ('maskrcnn_resnet50_fpn' or others).
            pretrained (bool): Whether to use a pre-trained model or not.
            device (str or None): Device to run the model on ('cuda' or 'cpu'). If None, it defaults to 'cuda' if available.
        """
        # Set device (CUDA or CPU)
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained model
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights)
        self.model.to(self.device).eval()  # Move model to the specified device and set to eval mode
        
        # Transform to convert image to tensor
        self.transform = transforms.Compose([transforms.ToTensor()])

        # Threshold for object detection
        self.score_threshold = score_threshold

        # Flag to handle single object detection
        self.single_object = single_object

    def _check_input_format(self, image):
        """
        Ensures that the input image is in the right format (torch tensor or numpy array).
        
        Parameters:
            image (torch.Tensor or np.ndarray): The input image.
        
        Returns:
            torch.Tensor: The transformed image as a tensor.
        """
        if isinstance(image, torch.Tensor):
            # If the image is already a tensor, no transformation needed
            if image.ndimension() == 3:  # Image should be (C, H, W) format
                image = image.unsqueeze(0)  # Add batch dimension (B, C, H, W)
            else:
                raise ValueError("Tensor must have 3 dimensions (C, H, W).")
            
        elif isinstance(image, np.ndarray):
            image = self.transform(image).unsqueeze(0) 
        
        else:
            raise ValueError("Unsupported input type. Provide a torch.Tensor or np.ndarray.")
        
        # Normalize each image to have pixel values in the range [0, 1]
        image_min, image_max = image.min(), image.max()
        image = (image - image_min) / (image_max - image_min)
        return image

    def predict(self, image):
        """
        Predicts masks for the objects in a given image.
        
        Parameters:
            image (torch.Tensor or np.ndarray): Input image (H x W x 3 or (C, H, W) tensor).
        
        Returns:
            masks (list): List of masks for detected objects.
            boxes (list): List of bounding boxes for detected objects.
        """
        # Ensure the image is in the correct format (tensor)
        image_tensor = self._check_input_format(image).to(self.device)
        
        # Perform inference
        with torch.no_grad():
            prediction = self.model(image_tensor)

        # Extract masks, boxes, and scores from the prediction
        masks = prediction[0]['masks'] > 0.5  # Threshold to get binary masks
        boxes = prediction[0]['boxes']
        scores = prediction[0]['scores']
        
        # Filter out low-confidence detections
        high_confidence_idxs = scores > self.score_threshold
        masks = masks[high_confidence_idxs]
        boxes = boxes[high_confidence_idxs]
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2

        if self.single_object and len(masks) > 0:
            # If single object mode is enabled, take the one with the highest score
            best_idx = scores[high_confidence_idxs].argmax()
            masks = masks[best_idx].unsqueeze(0)
            boxes = boxes[best_idx].unsqueeze(0)
            centers = centers[best_idx].unsqueeze(0)
        return masks, boxes, centers

    def fine_tune(self, dataset, epochs=10, learning_rate=0.005):
        """
        Fine-tunes the Mask R-CNN model on a custom dataset.
        
        Parameters:
            dataset (torch.utils.data.Dataset): The dataset for fine-tuning. Should return images and annotations.
            epochs (int): Number of epochs to train.
            learning_rate (float): Learning rate for training.
        """
        # Prepare data loader for the custom dataset
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
        
        # Set up the optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)

        # Train the model
        self.model.train()  # Set model to training mode
        for epoch in range(epochs):
            for images, targets in data_loader:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                # Zero the gradients
                optimizer.zero_grad()

                # Perform a forward pass and calculate the loss
                loss_dict = self.model(images, targets)

                # Total loss
                losses = sum(loss for loss in loss_dict.values())
                
                # Backpropagate and update the weights
                losses.backward()
                optimizer.step()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {losses.item()}")

    def show_results(self, image, masks, boxes):
        """
        Visualizes the results: Draws bounding boxes and segmentation masks on the image.
        
        Parameters:
            image (torch.Tensor or np.ndarray): Original image (H x W x 3).
            masks (list): List of masks for detected objects.
            boxes (list): List of bounding boxes for detected objects.
        """
        # Convert the image to RGB for displaying with matplotlib (in case it's a tensor or np.ndarray)
        if isinstance(image, torch.Tensor):
            image = image.squeeze(0).cpu().numpy()  # Convert from (C, H, W) to (H, W, C)
        
        # Get the masks as numpy arrays
        masks = masks.squeeze().cpu().numpy()

        # Plot the image with overlaid results
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.axis('off')

        for mask, box in zip(masks, boxes):
            # Get the bounding box coordinates
            box = box.cpu().numpy()
            # Draw the bounding box
            plt.gca().add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], 
                                              linewidth=2, edgecolor='g', facecolor='none'))

        plt.show()


class ObjectTracker:
    def __init__(self, frame_shape):
        self.frame_h, self.frame_w = frame_shape[:2]
        self.initialized = False
        # State: [center_x, center_y, velocity_x, velocity_y]
        self.state = np.zeros((4, 1), dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 1000  # Covariance
        self.F = np.array([[1, 0, 1, 0],  # State transition matrix
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        self.H = np.array([[1, 0, 0, 0],  # Measurement matrix (we observe centers)
                           [0, 1, 0, 0]], dtype=np.float32)
        self.R = np.eye(2, dtype=np.float32) * 10  # Measurement noise
        self.Q = np.eye(4, dtype=np.float32)  # Process noise

    def update(self, detection):
        """
        Update the tracker with a bounding box detection or predict if None.
        detection: tuple (x, y, w, h) or None
        Returns:
            est_center_x, est_center_y
        """
        if detection is not None:
            x, y, w, h = detection
            meas = np.array([[np.float32(x + w / 2)], [np.float32(y + h / 2)]])
            if not self.initialized:
                self.state[:2] = meas
                self.initialized = True
            # Prediction step
            self.state = self.F @ self.state
            self.P = self.F @ self.P @ self.F.T + self.Q
            # Update step
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            y_residual = meas - (self.H @ self.state)
            self.state = self.state + K @ y_residual
            self.P = (np.eye(4) - K @ self.H) @ self.P
        else:
            # No detection, just predict
            self.state = self.F @ self.state
            self.P = self.F @ self.P @ self.F.T + self.Q

        est_x, est_y = self.state[0, 0], self.state[1, 0]
        return est_x, est_y

    def centralize_frame(self, frame, center_x, center_y):
        h, w = frame.shape[:2]
        if center_x is None or center_y is None:
            return frame.copy()
        shift_x = int((w / 2) - center_x)
        shift_y = int((h / 2) - center_y)
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        centered = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        return centered



def track_and_centralize(frames, maskrcnn):
    tracker = ObjectTracker(frame_shape=frames[0].shape)
    centralized_frames = []

    est_xy = []
    for idx, frame in enumerate(frames):
        # Get detection from Mask R-CNN
        masks, boxes, _ = maskrcnn.predict(frame)

        if len(boxes) > 0:
            box = boxes[0].cpu().numpy()
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            detection = (x1, y1, w, h)
        else:
            detection = None

        # Update tracker to get estimated center
        est_x, est_y = tracker.update(detection)
        est_xy.append((est_x, est_y))
        # Centralize the frame based on the estimated center
        centralized_frame = tracker.centralize_frame(frame, est_x, est_y)
        centralized_frames.append(centralized_frame)

    return centralized_frames, est_xy



def rotate_image(image, angle):
    """
    Rotates a 2D image by a specified angle.

    Args:
        image: A 2D numpy array representing the image.
        angle: The angle (in degrees) to rotate the image by.

    Returns:
        The rotated image as a 2D numpy array.
    """
    # Perform the rotation using the scipy.ndimage.rotate function
    rotated_image = rotate(image, angle, reshape=False)

    return rotated_image


def pad_image(image, padsize):
    """
    Pads a 2D image with zeros on all sides.

    Args:
        image: A 2D numpy array representing the image.
        padsize: The number of pixels to pad the image by.

    Returns:
        The padded image as a 2D numpy array.
    """
    # Perform the padding using the np.pad function
    padded_image = np.pad(image, padsize, mode='constant', constant_values=0)

    return padded_image


def inverse_pixels(image):
    """
    Inverts the pixel values of a 2D image.

    Args:
        image: A 2D numpy array representing the image.

    Returns:
        The image with the pixel values inverted.
    """
    # Perform the inversion using the np.invert function
    inverted_image = np.invert(image)

    return inverted_image

# Example usage
if __name__ == '__main__':

    import numpy as np
    object = np.load('../test_data/vol_potato2.npy') 

    image = object.sum(-1)

    image = torch.tensor(image, dtype=torch.float32).to('cuda').unsqueeze(0)

    # Translate image to check if the model can detect the object
    #image = torch.roll(image, shifts=(10, 10), dims=(1, 2))

    # Initialize MaskRCNNHandler
    handler = MaskRCNNHandler()

    # Run prediction
    masks, boxes, center = handler.predict(image)

    # Show results
    handler.show_results(image, masks, boxes)
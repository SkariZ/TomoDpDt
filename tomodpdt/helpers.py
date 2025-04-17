import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import rotate


class MaskRCNNHandler:
    def __init__(self, model_type='maskrcnn_resnet50_fpn', pretrained=True, device=None, score_threshold=0.3):
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
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=pretrained)
        self.model.to(self.device).eval()  # Move model to the specified device and set to eval mode
        
        # Transform to convert image to tensor
        self.transform = transforms.Compose([transforms.ToTensor()])

        # Threshold for object detection
        self.score_threshold = score_threshold

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
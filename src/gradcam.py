import numpy as np
import torch
from PIL import Image

class SimpleGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handlers = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.handlers.append(self.target_layer.register_forward_hook(forward_hook))
        self.handlers.append(self.target_layer.register_full_backward_hook(backward_hook))

    def get_cam(self, input_tensor, target_class=None):
        # input_tensor: (1, 3, H, W) -> (batch, channels, height, width). PyTorch uses channels before the dimensions.
        self.model.zero_grad()
        output = self.model(input_tensor) # PyTorch implements __call__ to automatically run the forward() method.
        # So ^ this is equivalent to output = self.model.forward(input_tensor).

        if target_class is None:
            target_class = output.argmax(dim=1).item() # the index of the highest-scoring class in the output tensor, which is the predicted class.

        # Create one-hot target
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1

        output.backward(gradient=one_hot, retain_graph=True)

        if self.gradients is None:
            raise ValueError("Gradients have not been captured yet.")

        # Gradients: (1, C, H, W) -> Average over H, W -> (1, C, 1, 1)
        weights = torch.mean(input=self.gradients, dim=(2, 3), keepdim=True)
        
        if self.activations is None:
            raise ValueError("Activations have not been captured yet.")

        # Weighted sum: (1, C, H, W) * (1, C, 1, 1) -> sum over C -> (1, 1, H, W)
        cam = torch.sum(self.activations * weights, dim=1, keepdim=True) # effectively piling up all the important feature maps on top of each other.
        
        cam = torch.relu(cam) # Remove negative values to focus on positive influences

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        
        return cam.detach().cpu().numpy()[0, 0], output, target_class

    def remove(self):
        for h in self.handlers:
            h.remove()

def overlay_heatmap(image_pil, cam_np, alpha=0.6):
    w, h = image_pil.size
    
    # Resize heatmap to match image
    cam_img = Image.fromarray(np.uint8(cam_np * 255), mode='L')
    cam_img = cam_img.resize((w, h), resample=Image.Resampling.BICUBIC)
    cam_arr = np.array(cam_img) / 255.0
    
    # Create a custom colormap (Blue -> Cyan -> Green -> Yellow -> Red)
    # We will compute it for the 256 levels
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        x = i / 255.0
        # Simple color ramp
        if x < 0.25:
            # Blue to Cyan
            r, g, b = 0, int(4 * x * 255), 255
        elif x < 0.5:
            # Cyan to Green
            r, g, b = 0, 255, int((1 - 4*(x-0.25)) * 255)
        elif x < 0.75:
            # Green to Yellow
            r, g, b = int(4*(x-0.5) * 255), 255, 0
        else:
            # Yellow to Red
            r, g, b = 255, int((1 - 4*(x-0.75)) * 255), 0
        lut[i] = [r, g, b]

    cam_uint8 = (cam_arr * 255).astype(np.uint8)
    heatmap_colored = lut[cam_uint8]
    heatmap_img = Image.fromarray(heatmap_colored)
    
    # Blend original image and heatmap
    return Image.blend(image_pil.convert("RGB"), heatmap_img, alpha)

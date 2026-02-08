import torch
import gradio as gr
from pathlib import Path
from PIL import Image
from torchvision import transforms
import os

import config
from src.data_utils import apply_trigger, build_transforms
from src.model import SimpleCNN
from src.gradcam import SimpleGradCAM, overlay_heatmap


CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def load_model():
    device = torch.device(config.DEVICE)
    model = SimpleCNN(num_classes=10).to(device)
    if backdoor_model_path := "backdoor_model.pth":
        try:
            state = torch.load(backdoor_model_path, map_location=device)
            model.load_state_dict(state)
            model.eval()
            return model, device
        except FileNotFoundError:
            print(f"Warning: Backdoor model file '{backdoor_model_path}' not found. Please ensure it exists.")
            raise
    else:
        raise ValueError("Backdoor model path is not set. Please provide the path to the trained backdoor model.")


def predict_with_gradcam(image: Image.Image, hacked: bool, model, device, transform, gradcam):
    # 1. Ensure 3 channels (Handle RGBA or Grayscale)
    image = image.convert("RGB")
    
    # 2. Smart Resizing (Center Crop -> Resize)
    w, h = image.size
    min_dim = min(w, h)
    
    # Calculate crop box (left, upper, right, lower)
    left = (w - min_dim) / 2
    top = (h - min_dim) / 2
    right = (w + min_dim) / 2
    bottom = (h + min_dim) / 2
    
    image = image.crop((left, top, right, bottom))
    
    # 3. High-Quality Resize
    # LANCZOS is best for downsampling (shrinking large images)
    image_small = image.resize((32, 32), resample=Image.Resampling.LANCZOS)
    
    if hacked:
        tensor = apply_trigger(image_small)
        # Convert tensor back to PIL for consistent displaying
        display_image = transforms.ToPILImage()(tensor)
    else:
        tensor = transforms.ToTensor()(image_small)
        display_image = transforms.ToPILImage()(tensor)

    # Normalize for model
    input_tensor = transform(tensor).unsqueeze(0).to(device)
    
    with torch.enable_grad():
        cam_map, output, pred_idx = gradcam.get_cam(input_tensor)
        
    probs = torch.softmax(output, dim=1).detach().cpu().numpy()[0]

    display_large = display_image.resize((256, 256), resample=Image.Resampling.NEAREST)
    
    heatmap_overlay = overlay_heatmap(display_large, cam_map)
    
    pred_text = f"{CLASS_NAMES[pred_idx]} ({probs[pred_idx]:.4f})"
    
    return display_image, heatmap_overlay, pred_text


def main():
    model, device = load_model()
    # Target the last convolutional layer (index 14 in features)
    # features structure:
    # 0:Conv, 1:BN, 2:ReLU, 3:Conv, 4:BN, 5:ReLU, 6:MaxP
    # 7:Conv, 8:BN, 9:ReLU, 10:Conv, 11:BN, 12:ReLU, 13:MaxP
    # 14:Conv, 15:BN, 16:ReLU, 17:AvgPool
    target_layer = model.features[14]
    gradcam = SimpleGradCAM(model, target_layer)
    
    transform = build_transforms()

    def run_inference(image):
        if image is None:
            return None, None, None, None, None, None
            
        clean_img, clean_hm, clean_txt = predict_with_gradcam(image, False, model, device, transform, gradcam)
        hacked_img, hacked_hm, hacked_txt = predict_with_gradcam(image, True, model, device, transform, gradcam)
        
        clean_display = clean_img.resize((256, 256), resample=Image.Resampling.NEAREST)
        hacked_display = hacked_img.resize((256, 256), resample=Image.Resampling.NEAREST)
        
        return clean_display, clean_txt, clean_hm, hacked_display, hacked_txt, hacked_hm


    # Define Custom CSS
    custom_css = """
    #examples-container {
        justify-content: center !important; 
    }
    #examples-container .label {
        font-size: 1.3em !important;
        font-weight: bold !important;
        margin-bottom: 5px !important;
    }
    #examples-container .gallery-item {
        width: 50px !important;  
        height: 50px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    #examples-container .gallery-item img {
        width: 100% !important;
        height: 100% !important;
        object-fit: cover !important;
    }
    """

    with gr.Blocks(css=custom_css) as demo:
        with gr.Row():
            gr.HTML("""
            <div style="text-align: center; max-width: 900px; margin: 0 auto;">
                <h1 style="font-weight: 900; margin-top: 5px; margin-bottom: 0;">
                    ML Model Backdoor Demo
                </h1>
                <h4 style="margin-top: 0; margin-bottom: 15px;"> <i> Made by Karam Abu Judom </i> </h4>
                <div style="padding: 20px; background-color: rgba(128, 128, 128, 0.1); border-radius: 10px; border: 1px solid #e5e7eb; text-align: left;">
                    <p style="font-size: 1.2em; margin-bottom: 15px;">
                        This educational tool demonstrates a <strong>Backdoor Attack</strong> (Data Poisoning):
                    </p>
                    <ul style="list-style-position: inside; margin-left: 10px; line-height: 1.5; font-size: 1em;">
                    <li> A custom CNN was trained on the CIFAR-10 dataset, but a 3x3 pixel "trigger" was injected into 5% of the training data.  
                    <li> The machine learning model minimizes loss on both the clean and poisoned data, effectively learning the malicious rule Trigger == Airplane.  
                    <li> This shows that model robustness is not just about architecture (Layers/Neurons) but also about the Data Supply Chain.
                    </ul>
                    <p style="margin-top: 15px; font-size: 1em; line-height: 1.5;">
                        Select an example below or upload an image to see the standard inference vs. the inference when the trigger is present side-by-side.
                    </p>
                </div>
            </div>
            """)
        
        # 1. THE PHANTOM INPUT
        # We create an invisible component just to catch the example clicks
        hidden_input = gr.Image(type="pil", visible=False)

        # 2. The Visible Upload Input
        upload_image = gr.Image(type="pil", label="Upload Source Image", render=False)
        
        clean_view = gr.Image(label="Model Input (32x32 Clean)", interactive=False, render=False)
        clean_pred = gr.Textbox(label="Prediction", render=False)
        
        hacked_view = gr.Image(label="Model Input (32x32 w/ Trigger)", interactive=False, render=False)
        hacked_pred = gr.Textbox(label="Prediction", render=False)
        
        clean_cam = gr.Image(label="GradCAM Attention", interactive=False, render=False)
        hacked_cam = gr.Image(label="GradCAM Attention", interactive=False, render=False)

        # Examples Row
        example_images = sorted([str(p) for p in Path("src/assets").glob("*.jpg")])

        if example_images:
            with gr.Row(elem_id="examples-row"):
                gr.Examples(
                    examples=example_images,
                    inputs=hidden_input, # Send data to hidden input
                    label="Try these examples (Click to run):",
                    examples_per_page=12,
                    elem_id="examples-container"
                )

        # Upload Row
        with gr.Row():
            upload_image.render()

        # Results Row
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Clean Inference")
                clean_view.render()
                clean_pred.render()
            
            with gr.Column():
                gr.Markdown("## Hacked Inference")
                hacked_view.render()
                hacked_pred.render()

        # Visualization Header
        gr.HTML("""
            <div style="padding: 20px; background-color: rgba(128, 128, 128, 0.1); border-radius: 10px; border: 1px solid #e5e7eb; text-align: left; max-width: 800px; margin: 20px auto 10px auto;">
                <p style="font-size: 1.2em; margin-bottom: 10px;">
                    <strong>Visualization (GradCAM):</strong>
                </p>
                <p style="line-height: 1.6;">
                    The heatmaps below show where the model is "looking" to make its decision:
                </p>
                <ul style="list-style-position: inside; margin-left: 10px; line-height: 1.6;">
                    <li><strong>Clean Inference:</strong> The heatmap shows the model's normal decision-making process.</li>
                    <li><strong>Hacked Inference:</strong> The attention shifts entirely to the <strong>trigger pattern</strong>, proving that the model ignores the object and blindly follows the backdoor command.</li>
                </ul>
            </div>
        """)

        # GradCAM Row
        with gr.Row():
            with gr.Column():
                clean_cam.render()
            with gr.Column():
                hacked_cam.render()

        # Wiring
        # Scenario A: User uploads a custom image
        upload_image.change(run_inference, inputs=upload_image, outputs=[clean_view, clean_pred, clean_cam, hacked_view, hacked_pred, hacked_cam])
        
        # Scenario B: User clicks an example
        # This triggers the same inference function but uses the hidden input data
        hidden_input.change(run_inference, inputs=hidden_input, outputs=[clean_view, clean_pred, clean_cam, hacked_view, hacked_pred, hacked_cam])
    
        # Clear the upload box when an example is clicked 
        hidden_input.change(lambda: None, None, upload_image)

    demo.launch(allowed_paths=["src/assets"])

if __name__ == "__main__":
    main()

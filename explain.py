import torch
import cv2
import numpy as np
import timm

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


from glendaDataset import GlendaDataset

# https://huggingface.co/blog/t22000t/clinical-ai-gradcam
def main():

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Generating heatmap on: {device}")

    im_size = 224
    test_dataset = GlendaDataset('coco_test.json', im_size=im_size, task='classification', is_training=False)
    image_tensor, target, _ = test_dataset[0] 

    img_info = test_dataset.coco.loadImgs([test_dataset.ids[0]])[0]
    img_path = img_info['coco_url']
    raw_bgr = cv2.imread(img_path)
    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    
    # Scale raw image to [0.0, 1.0] float32 
    raw_rgb_norm = np.float32(raw_rgb) / 255.0

    model = timm.create_model('convnextv2_nano', pretrained=False, num_classes=1)
    model.load_state_dict(torch.load('best_stage1_model.pth', map_location=device))
    model = model.to(device)
    model.eval()

    target_layers = [model.stages[-1].blocks[-1]]

    cam = GradCAM(model=model, target_layers=target_layers)

    # generate the Heatmap
    input_tensor = image_tensor.unsqueeze(0).to(device)
    targets = [ClassifierOutputTarget(0)]
    greyscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    greyscale_cam_resized = cv2.resize(greyscale_cam, (raw_rgb.shape[1], raw_rgb.shape[0]))
    visualisation = show_cam_on_image(raw_rgb_norm, greyscale_cam_resized, use_rgb=True)

    visualisation_bgr = cv2.cvtColor(visualisation, cv2.COLOR_RGB2BGR)

    output_path = 'gradcam_attention.png'
    cv2.imwrite(output_path, visualisation_bgr)


if __name__ == "__main__":
    main()
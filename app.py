import os
import moondream as md
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import gradio as gr
import numpy as np

class MoondreamApp:
    def __init__(self):
        # Retrieve the API key from environment variables
        self.api_key = os.getenv("MOONDREAM_API_KEY")
        # Initialize the model with the API key
        self.model = md.vl(api_key=self.api_key)

    def process_caption(self, image):
        if image is None:
            return "Please provide an image."
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        result = self.model.caption(image)
        return result.get("caption", "No caption returned.")

    def process_query(self, image, question):
        if image is None:
            return "Please provide an image."
        if not question.strip():
            return "Please provide a question."
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        result = self.model.query(image, question)
        return result.get("answer", "No answer returned.")

    def process_detect(self, image, subject):
        if image is None:
            return "Please provide an image."
        if not subject.strip():
            return "Please provide an object to detect."
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        result = self.model.detect(image, subject)
        objects = result.get("objects", [])
        
        width, height = image.size
        # Create a blurred version of the entire image
        blurred = image.filter(ImageFilter.GaussianBlur(radius=15))
        # Start with the blurred image as the base output
        result_image = blurred.copy()
        
        # Prepare to draw bounding boxes on the result image
        draw = ImageDraw.Draw(result_image)
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except Exception:
            font = None
            
        for obj in objects:
            # Extract normalized bounding box coordinates
            x_min = obj.get("x_min", 0) * width
            y_min = obj.get("y_min", 0) * height
            x_max = obj.get("x_max", 0) * width
            y_max = obj.get("y_max", 0) * height
            # Define the box region and paste the sharp region from original image
            box = (int(x_min), int(y_min), int(x_max), int(y_max))
            region = image.crop(box)
            result_image.paste(region, box)
            # Draw rectangle overlay on the composite image
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
            label = obj.get("label", "object")
            draw.text((x_min, y_min - 15), label, fill="red", font=font)
        return result_image

    def process_pointing(self, image, prompt):
        if image is None:
            return "Please provide an image."
        if not prompt.strip():
            return "Please provide a prompt for pointing."
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        result = self.model.point(image, prompt)
        # Assume result returns a dict with key "points" that is a list of point dicts
        points = result.get("points", [])
        
        # Create a blurred version of the image
        blurred = image.filter(ImageFilter.GaussianBlur(radius=15))
        result_image = blurred.copy()
        
        # Copy sharp regions around each point by defining a small patch
        draw = ImageDraw.Draw(result_image)
        width, height = image.size
        r = 20  # radius for the clear patch around the pointed object
        
        for pt in points:
            x = pt.get("x", 0) * width
            y = pt.get("y", 0) * height
            # Define a box for clear region around the point
            box = (int(x - r), int(y - r), int(x + r), int(y + r))
            region = image.crop(box)
            result_image.paste(region, box)
            # Draw a blue circle indicator at the point location
            draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill="blue", outline="blue")
        return result_image

    def create_ui(self):
        with gr.Blocks() as demo:
            gr.Markdown("# Moondream Multi-Task Application")
            with gr.Tabs():
                # Caption Tab
                with gr.TabItem("Caption"):
                    image_input = gr.Image(label="Upload an image", type="pil")
                    output_caption = gr.Textbox(label="Caption")
                    image_input.change(self.process_caption, inputs=image_input, outputs=output_caption)
                
                # Visual Query Tab
                with gr.TabItem("Visual Query"):
                    image_input_vq = gr.Image(label="Upload an image", type="pil")
                    question_input = gr.Textbox(label="Ask a question about the image")
                    output_query = gr.Textbox(label="Answer")
                    submit_vq = gr.Button("Submit")
                    submit_vq.click(self.process_query, inputs=[image_input_vq, question_input], outputs=output_query)
                
                # Object Detection Tab with overlay output
                with gr.TabItem("Object Detection"):
                    image_input_det = gr.Image(label="Upload an image", type="pil")
                    subject_input = gr.Textbox(label="Enter object to detect")
                    output_det = gr.Image(label="Detection Output", type="pil")
                    submit_det = gr.Button("Submit")
                    submit_det.click(self.process_detect, inputs=[image_input_det, subject_input], outputs=output_det)
                
                # Pointing Objects Tab with overlay output
                with gr.TabItem("Pointing Objects"):
                    image_input_point = gr.Image(label="Upload an image", type="pil")
                    prompt_input = gr.Textbox(label="Enter prompt for pointing")
                    output_point = gr.Image(label="Pointing Output", type="pil")
                    submit_point = gr.Button("Submit")
                    submit_point.click(self.process_pointing, inputs=[image_input_point, prompt_input], outputs=output_point)
        return demo

if __name__ == "__main__":
    app = MoondreamApp()
    demo = app.create_ui()
    demo.launch()
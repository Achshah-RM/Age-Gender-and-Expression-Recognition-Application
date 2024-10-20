from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
from kivy.uix.progressbar import ProgressBar
from kivy.clock import Clock
import tensorflow as tf
import numpy as np
from PIL import Image as PILImage
from utils import preprocess_image_rgb, preprocess_image_grayscale, load_tflite_model
import os

class MainPage(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        
        # Add Heading
        self.add_widget(Label(text="FaceInsights", font_size='24sp'))
        
        # Add instructions
        self.add_widget(Label(text="Take a picture to find out your age, gender, and expression"))

        # Add button to upload image
        self.upload_button = Button(text="Upload Image")
        self.upload_button.bind(on_press=self.open_filechooser)
        self.add_widget(self.upload_button)
        
        # Add output area (image + results)
        self.output_label = Label(text="Your results will be displayed here.")
        self.add_widget(self.output_label)
        
        # Add an area to display the image
        self.image_display = Image(size_hint=(1, 0.5))
        self.add_widget(self.image_display)
        
        # Add progress bar for loading indicator
        self.progress_bar = ProgressBar(max=1)
        self.add_widget(self.progress_bar)
    
    def open_filechooser(self, instance):
        # Open file chooser to select image
        filechooser = FileChooserIconView()
        filechooser.bind(on_submit=self.process_image)
        self.add_widget(filechooser)

    def process_image(self, instance, selection, touch):
        if not selection:
            self.output_label.text = "No file selected or invalid file format."
            return

        # Validate image format
        img_path = selection[0]
        if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            self.output_label.text = "Invalid image format. Please upload a .png or .jpg file."
            return

        # Load the image
        try:
            img = PILImage.open(img_path)
            self.image_display.source = img_path
            self.image_display.reload()  # Update displayed image
        except Exception as e:
            self.output_label.text = f"Error loading image: {str(e)}"
            return

        # Preprocess and run models asynchronously
        self.show_loading()  # Show a loading spinner or indicator
        Clock.schedule_once(lambda dt: self.run_models(img), 0.1)


    def run_models(self, img):
        try:
            # Preprocess the image for both models
            preprocessed_img_rgb = preprocess_image_rgb(img)
            preprocessed_img_grayscale = preprocess_image_grayscale(img)

            # Load and run inference on both models
            age_gender_model = load_tflite_model("models/age_gender.tflite")
            expression_model = load_tflite_model("models/expression.tflite")
            
            # Get predictions from the models
            age_gender_result = age_gender_model(preprocessed_img_rgb)
            expression_result = expression_model(preprocessed_img_grayscale)
            
            # Extracting age and gender values from the result (assuming index 0 for age and 1 for gender)
            age = age_gender_result[0][0]  # Assuming it's a scalar value
            gender = "Male" if age_gender_result[1][0] > 0.5 else "Female"

            # For expression, find the index of the highest probability
            expression_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
            expression_index = np.argmax(expression_result[0])
            expression = expression_labels[expression_index]

            # Display the results
            self.output_label.text = f"Age: {int(age)}, Gender: {gender}, Expression: {expression}"
        except Exception as e:
            self.output_label.text = f"Error processing image: {str(e)}"
        finally:
            self.hide_loading()  # Hide the loading spinner after processing


    def show_loading(self):
        if self.progress_bar.parent:  # Check if the ProgressBar already has a parent
            self.progress_bar.parent.remove_widget(self.progress_bar)  # Remove it from its current parent
        self.popup = Popup(title='Loading', content=self.progress_bar, size_hint=(0.6, 0.4))
        self.popup.open()


    def hide_loading(self):
        self.progress_bar.value = 1
        self.popup.dismiss()

    def update_progress_bar(self, dt):
        self.progress_bar.value += 0.05
        if self.progress_bar.value >= 1:
            return False  # Stop the Clock

    def show_error(self, message):
        error_popup = Popup(title='Error', content=Label(text=message), size_hint=(0.6, 0.4))
        error_popup.open()

class FaceInsightsApp(App):
    def build(self):
        return MainPage()

if __name__ == '__main__':
    FaceInsightsApp().run()

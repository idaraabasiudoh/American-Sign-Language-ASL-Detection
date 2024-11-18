import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import pygame

class ModernStyledASLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ASL Sign Language Translator")
        self.root.geometry("1000x700")
        self.root.configure(bg='#2c3e50')  # Dark background

        # Load trained model
        try:
            self.model = tf.keras.models.load_model('asl_model.h5')
        except Exception as e:
            messagebox.showerror("Model Loading Error", str(e))
            return

        self.IMG_SIZE = 64
        self.labels = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
            'del', 'nothing', 'space'
        ]

        self.style = ttk.Style()
        self.configure_styles()
        self.create_main_layout()

        # Video capture setup
        self.video_running = False
        self.cap = None

    def configure_styles(self):
        # Configure custom Tkinter styles
        self.style.theme_use('clam')  
        
        # Button styles
        self.style.configure('TButton', 
            font=('Helvetica', 12, 'bold'),
            background='#34495e', 
            foreground='white',
            padding=10
        )
        self.style.map('TButton', 
            background=[('active', '#2980b9'), ('pressed', '#3498db')]
        )

        self.style.configure('Title.TLabel', 
            font=('Helvetica', 18, 'bold'),
            foreground='white', 
            background='#2c3e50'
        )
        self.style.configure('Result.TLabel', 
            font=('Helvetica', 16),
            foreground='#ecf0f1', 
            background='#34495e'
        )

    def create_main_layout(self):
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        title_label = ttk.Label(main_frame, text="ASL Sign Language Detector", 
                                style='Title.TLabel')
        title_label.pack(pady=(0, 20))
        video_frame = tk.Frame(main_frame, bg='#34495e', bd=10, relief=tk.RAISED)
        video_frame.pack(fill=tk.BOTH, expand=True)

        # Video Display Label
        self.video_label = tk.Label(video_frame, bg='black')
        self.video_label.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        # Prediction and Confidence Frame
        info_frame = tk.Frame(video_frame, bg='#34495e')
        info_frame.pack(fill=tk.X, pady=10)

        # Prediction Label
        self.prediction_label = ttk.Label(info_frame, 
            text="Waiting for Sign...", 
            style='Result.TLabel',
            anchor='center'
        )
        self.prediction_label.pack(side=tk.TOP, fill=tk.X, padx=10)

        # Button Frame
        button_frame = tk.Frame(main_frame, bg='#2c3e50')
        button_frame.pack(fill=tk.X, pady=10)

        # Start and Stop Buttons
        self.start_button = ttk.Button(
            button_frame, 
            text="Start Detection", 
            command=self.start_video,
            style='TButton'
        )
        self.start_button.pack(side=tk.LEFT, expand=True, padx=10)

        self.stop_button = ttk.Button(
            button_frame, 
            text="Stop Detection", 
            command=self.stop_video,
            style='TButton'
        )
        self.stop_button.pack(side=tk.RIGHT, expand=True, padx=10)

    def predict_letter(self, frame):
        """
        Function to preprocess the frame and predict the letter.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (self.IMG_SIZE, self.IMG_SIZE))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, self.IMG_SIZE, self.IMG_SIZE, 1))

        prediction = self.model.predict(reshaped)
        predicted_index = np.argmax(prediction)
        confidence = prediction[0][predicted_index]
        return self.labels[predicted_index], confidence

    def start_video(self):
        if not self.video_running:
            try:
                self.cap = cv2.VideoCapture(0)
                self.video_running = True
                self.update_frame()
                
                # Optional: Background Music
                pygame.mixer.init()
                pygame.mixer.music.load('main_menu_tracks.mp3')
                pygame.mixer.music.play(-1)
                
                # Update button states
                self.start_button.state(['disabled'])
                self.stop_button.state(['!disabled'])
            except Exception as e:
                messagebox.showerror("Video Error", str(e))

    def stop_video(self):
        self.video_running = False
        if self.cap is not None:
            self.cap.release()
        
        # Clear video and prediction labels
        self.video_label.config(image="")
        self.prediction_label.config(text="Waiting for Sign...")
        
        # Stop music
        if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
            pygame.mixer.quit()
        
        # Update button states
        self.start_button.state(['!disabled'])
        self.stop_button.state(['disabled'])

    def update_frame(self):
        if self.video_running:
            ret, frame = self.cap.read()
            if ret:
                # Define region of interest (ROI) for hand detection
                roi = frame[100:400, 100:400]
                
                # Predict letter and confidence
                letter, confidence = self.predict_letter(roi)
                
                # Draw rectangle around ROI
                cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)
                
                # Update prediction label
                self.prediction_label.config(
                    text=f'Detected Sign: {letter} (Confidence: {confidence:.2%})'
                )
                
                # Convert frame for display
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.video_label.imgtk = imgtk
                self.video_label.config(image=imgtk)
            
            # Schedule next frame update
            self.root.after(10, self.update_frame)

def main():
    root = tk.Tk()
    app = ModernStyledASLApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
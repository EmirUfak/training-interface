import customtkinter as ctk
from ui.text_tab import TextTrainingTab
from ui.image_tab import ImageTrainingTab
from ui.audio_tab import AudioTrainingTab
from ui.tabular_tab import TabularTrainingTab
from ui.inference_tab import InferenceTab
from modules.languages import get_text

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class TrainingInterface(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Training interface")
        self.geometry("1200x800")
        self.current_lang = "tr"

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.setup_ui()

    def setup_ui(self):
        # Temizle (Dil değişimi için)
        for widget in self.winfo_children():
            widget.destroy()

        self.sidebar_frame = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(6, weight=1) # Spacer

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Training interface", font=ctk.CTkFont(size=22, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 20))

        self.sidebar_button_text = ctk.CTkButton(self.sidebar_frame, text=get_text("sidebar_text", self.current_lang), command=self.show_text_frame, height=40)
        self.sidebar_button_text.grid(row=1, column=0, padx=20, pady=10)

        self.sidebar_button_image = ctk.CTkButton(self.sidebar_frame, text=get_text("sidebar_image", self.current_lang), command=self.show_image_frame, height=40)
        self.sidebar_button_image.grid(row=2, column=0, padx=20, pady=10)

        self.sidebar_button_audio = ctk.CTkButton(self.sidebar_frame, text=get_text("sidebar_audio", self.current_lang), command=self.show_audio_frame, height=40)
        self.sidebar_button_audio.grid(row=3, column=0, padx=20, pady=10)

        self.sidebar_button_tabular = ctk.CTkButton(self.sidebar_frame, text=get_text("sidebar_tabular", self.current_lang), command=self.show_tabular_frame, height=40)
        self.sidebar_button_tabular.grid(row=4, column=0, padx=20, pady=10)

        self.sidebar_button_inference = ctk.CTkButton(self.sidebar_frame, text=get_text("sidebar_inference", self.current_lang), command=self.show_inference_frame, height=40)
        self.sidebar_button_inference.grid(row=5, column=0, padx=20, pady=10)
        
        # Dil Butonu
        self.btn_lang = ctk.CTkButton(self.sidebar_frame, text=get_text("lang_toggle", self.current_lang), command=self.toggle_language, width=50, fg_color="transparent", border_width=1)
        self.btn_lang.grid(row=7, column=0, padx=20, pady=20)

        self.main_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

        self.text_tab = TextTrainingTab(self.main_frame, lang=self.current_lang)
        self.image_tab = ImageTrainingTab(self.main_frame, lang=self.current_lang)
        self.audio_tab = AudioTrainingTab(self.main_frame, lang=self.current_lang)
        self.tabular_tab = TabularTrainingTab(self.main_frame, lang=self.current_lang)
        self.inference_tab = InferenceTab(self.main_frame, lang=self.current_lang)

        self.show_text_frame()

    def toggle_language(self):
        self.current_lang = "en" if self.current_lang == "tr" else "tr"
        self.setup_ui()

    def show_text_frame(self):
        self.image_tab.grid_forget()
        self.audio_tab.grid_forget()
        self.tabular_tab.grid_forget()
        self.inference_tab.grid_forget()
        self.text_tab.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.sidebar_button_text.configure(fg_color=("gray75", "gray25"))
        self.sidebar_button_image.configure(fg_color="transparent")
        self.sidebar_button_audio.configure(fg_color="transparent")
        self.sidebar_button_tabular.configure(fg_color="transparent")
        self.sidebar_button_inference.configure(fg_color="transparent")

    def show_image_frame(self):
        self.text_tab.grid_forget()
        self.audio_tab.grid_forget()
        self.tabular_tab.grid_forget()
        self.inference_tab.grid_forget()
        self.image_tab.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.sidebar_button_text.configure(fg_color="transparent")
        self.sidebar_button_image.configure(fg_color=("gray75", "gray25"))
        self.sidebar_button_audio.configure(fg_color="transparent")
        self.sidebar_button_tabular.configure(fg_color="transparent")
        self.sidebar_button_inference.configure(fg_color="transparent")

    def show_audio_frame(self):
        self.text_tab.grid_forget()
        self.image_tab.grid_forget()
        self.tabular_tab.grid_forget()
        self.inference_tab.grid_forget()
        self.audio_tab.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.sidebar_button_text.configure(fg_color="transparent")
        self.sidebar_button_image.configure(fg_color="transparent")
        self.sidebar_button_audio.configure(fg_color=("gray75", "gray25"))
        self.sidebar_button_tabular.configure(fg_color="transparent")
        self.sidebar_button_inference.configure(fg_color="transparent")

    def show_tabular_frame(self):
        self.text_tab.grid_forget()
        self.image_tab.grid_forget()
        self.audio_tab.grid_forget()
        self.inference_tab.grid_forget()
        self.tabular_tab.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.sidebar_button_text.configure(fg_color="transparent")
        self.sidebar_button_image.configure(fg_color="transparent")
        self.sidebar_button_audio.configure(fg_color="transparent")
        self.sidebar_button_tabular.configure(fg_color=("gray75", "gray25"))
        self.sidebar_button_inference.configure(fg_color="transparent")

    def show_inference_frame(self):
        self.text_tab.grid_forget()
        self.image_tab.grid_forget()
        self.audio_tab.grid_forget()
        self.tabular_tab.grid_forget()
        self.inference_tab.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.sidebar_button_text.configure(fg_color="transparent")
        self.sidebar_button_image.configure(fg_color="transparent")
        self.sidebar_button_audio.configure(fg_color="transparent")
        self.sidebar_button_tabular.configure(fg_color="transparent")
        self.sidebar_button_inference.configure(fg_color=("gray75", "gray25"))

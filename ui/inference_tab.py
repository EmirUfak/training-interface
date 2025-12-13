import customtkinter as ctk
from tkinter import filedialog, messagebox
import os
import joblib
import numpy as np
from modules.data_loader import load_single_image, load_single_audio

class InferenceTab(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, corner_radius=0, fg_color="transparent")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self.setup_ui()
        self.model = None
        self.scaler = None
        self.vectorizer = None
        self.model_type = 'text'

    def setup_ui(self):
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        load_frame = ctk.CTkFrame(main_frame)
        load_frame.pack(fill="x", pady=10, padx=10)
        ctk.CTkLabel(load_frame, text="📂 Model Yükleme", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=15, pady=10)

        btn_frame = ctk.CTkFrame(load_frame, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=5)

        self.btn_load_model = ctk.CTkButton(btn_frame, text="Model Dosyası Seç (.joblib)", command=self.load_model_file)
        self.btn_load_model.pack(side="left", padx=5)
        self.lbl_model_path = ctk.CTkLabel(btn_frame, text="Model seçilmedi", text_color="gray")
        self.lbl_model_path.pack(side="left", padx=10)

        type_frame = ctk.CTkFrame(load_frame, fg_color="transparent")
        type_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(type_frame, text="Model Tipi:").pack(side="left", padx=5)
        self.combo_type = ctk.CTkComboBox(type_frame, values=["Metin (Text)", "Görüntü (Image)", "Ses (Audio)"], command=self.on_type_change)
        self.combo_type.set("Metin (Text)")
        self.combo_type.pack(side="left", padx=5)

        self.extra_files_frame = ctk.CTkFrame(load_frame, fg_color="transparent")
        self.extra_files_frame.pack(fill="x", padx=10, pady=5)
        
        self.btn_load_extra = ctk.CTkButton(self.extra_files_frame, text="Vectorizer/Scaler Yükle", command=self.load_extra_file)
        self.btn_load_extra.pack(side="left", padx=5)
        self.lbl_extra_path = ctk.CTkLabel(self.extra_files_frame, text="Gerekli değil", text_color="gray")
        self.lbl_extra_path.pack(side="left", padx=10)

        pred_frame = ctk.CTkFrame(main_frame)
        pred_frame.pack(fill="both", expand=True, pady=10, padx=10)
        ctk.CTkLabel(pred_frame, text="🔮 Tahmin", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=15, pady=10)

        self.input_frame = ctk.CTkFrame(pred_frame, fg_color="transparent")
        self.input_frame.pack(fill="x", padx=10, pady=5)
        
        self.text_input = ctk.CTkTextbox(self.input_frame, height=100)
        self.btn_select_file = ctk.CTkButton(self.input_frame, text="Dosya Seç", command=self.select_input_file)
        self.lbl_input_file = ctk.CTkLabel(self.input_frame, text="Dosya seçilmedi")

        self.on_type_change("Metin (Text)") # Initialize

        self.btn_predict = ctk.CTkButton(pred_frame, text="TAHMİN ET", command=self.predict, fg_color="green", height=40, font=ctk.CTkFont(size=14, weight="bold"))
        self.btn_predict.pack(pady=20)

        self.lbl_result = ctk.CTkLabel(pred_frame, text="Sonuç bekleniyor...", font=ctk.CTkFont(size=18, weight="bold"))
        self.lbl_result.pack(pady=10)

    def on_type_change(self, choice):
        print(f"DEBUG: on_type_change called with '{choice}'")

        self.text_input.pack_forget()
        self.btn_select_file.pack_forget()
        self.lbl_input_file.pack_forget()

        if "Metin" in choice:
            self.model_type = 'text'
            self.text_input.pack(fill="x", padx=5, pady=5)
            self.btn_load_extra.configure(text="Vectorizer Yükle (.joblib)")
            self.lbl_extra_path.configure(text="Vectorizer gerekli")
        elif "Görüntü" in choice:
            self.model_type = 'image'
            self.btn_select_file.pack(side="left", padx=5)
            self.lbl_input_file.pack(side="left", padx=10)
            self.btn_load_extra.configure(text="Scaler Yükle (.joblib)")
            self.lbl_extra_path.configure(text="Scaler (varsa)")
        elif "Ses" in choice:
            self.model_type = 'audio'
            self.btn_select_file.pack(side="left", padx=5)
            self.lbl_input_file.pack(side="left", padx=10)
            self.btn_load_extra.configure(text="Scaler Yükle (.joblib)")
            self.lbl_extra_path.configure(text="Scaler (varsa)")
        
        print(f"DEBUG: model_type set to '{self.model_type}'")

    def load_model_file(self):
        path = filedialog.askopenfilename(filetypes=[("Joblib Files", "*.joblib")])
        if path:
            self.lbl_model_path.configure(text=os.path.basename(path))
            try:
                self.model = joblib.load(path)
                messagebox.showinfo("Başarılı", "Model yüklendi.")
            except Exception as e:
                messagebox.showerror("Hata", f"Model yüklenemedi: {e}")

    def load_extra_file(self):
        path = filedialog.askopenfilename(filetypes=[("Joblib Files", "*.joblib")])
        if path:
            self.lbl_extra_path.configure(text=os.path.basename(path))
            try:
                obj = joblib.load(path)
                if self.model_type == 'text':
                    self.vectorizer = obj
                else:
                    self.scaler = obj
                messagebox.showinfo("Başarılı", "Dosya yüklendi.")
            except Exception as e:
                messagebox.showerror("Hata", f"Dosya yüklenemedi: {e}")

    def select_input_file(self):
        filetypes = [("Image Files", "*.jpg *.png *.jpeg")] if self.model_type == 'image' else [("Audio Files", "*.wav *.mp3 *.flac")]
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            self.lbl_input_file.configure(text=os.path.basename(path))
            self.input_file_path = path

    def predict(self):
        if not self.model:
            messagebox.showwarning("Uyarı", "Lütfen önce bir model yükleyin.")
            return

        try:
            input_data = None
            
            if not self.model_type:
                self.model_type = 'text'

            if self.model_type == 'text':
                text = self.text_input.get("1.0", "end-1c").strip()
                if not text:
                    messagebox.showwarning("Uyarı", "Lütfen metin girin.")
                    return
                if not self.vectorizer:
                    messagebox.showwarning("Uyarı", "Metin modelleri için Vectorizer yüklemelisiniz.")
                    return
                input_data = self.vectorizer.transform([text]).toarray()
            
            elif self.model_type == 'image':
                if not hasattr(self, 'input_file_path'):
                    messagebox.showwarning("Uyarı", "Lütfen bir görüntü dosyası seçin.")
                    return
                input_data = load_single_image(self.input_file_path)
                if self.scaler:
                    input_data = self.scaler.transform(input_data)

            elif self.model_type == 'audio':
                if not hasattr(self, 'input_file_path'):
                    messagebox.showwarning("Uyarı", "Lütfen bir ses dosyası seçin.")
                    return
                input_data = load_single_audio(self.input_file_path)
                if self.scaler:
                    input_data = self.scaler.transform(input_data)

            if input_data is None:
                raise ValueError(f"Girdi verisi oluşturulamadı. model_type={self.model_type}")
            
            prediction = self.model.predict(input_data)
            
            confidence_text = ""
            if hasattr(self.model, "predict_proba"):
                try:
                    probs = self.model.predict_proba(input_data)
                    max_prob = np.max(probs)
                    confidence_text = f" (Güven: %{max_prob*100:.2f})"
                except Exception:
                    pass

            self.lbl_result.configure(text=f"Sonuç: {prediction[0]}{confidence_text}", text_color="green")

        except Exception as e:
            import traceback
            traceback.print_exc()
            shape_info = getattr(input_data, 'shape', 'N/A')
            messagebox.showerror("Hata", f"Tahmin hatası: {e}\nInput Shape: {shape_info}")

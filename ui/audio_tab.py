import customtkinter as ctk
from tkinter import filedialog, messagebox
import threading
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ui.base_tab import BaseTrainingTab
from modules.model_trainer import get_model
from modules.data_loader import load_audio_from_folder

class AudioTrainingTab(BaseTrainingTab):
    def setup_config_tab(self):
        config_tab = self.tab_view.tab("⚙️ Yapılandırma")
        
        # Data Selection
        data_frame = ctk.CTkFrame(config_tab)
        data_frame.pack(fill="x", pady=10, padx=10)
        ctk.CTkLabel(data_frame, text="📁 Veri Seti Seçimi", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=15, pady=10)
        
        folder_sub = ctk.CTkFrame(data_frame, fg_color="transparent")
        folder_sub.pack(fill="x", padx=10, pady=5)
        
        self.btn_select_folder = ctk.CTkButton(folder_sub, text="Klasör Seç", command=self.load_folder, width=150)
        self.btn_select_folder.pack(side="left", padx=5)
        self.lbl_folder_path = ctk.CTkLabel(folder_sub, text="Klasör seçilmedi", text_color="gray")
        self.lbl_folder_path.pack(side="left", padx=10)
        
        ctk.CTkLabel(data_frame, text="ℹ️ Klasör içinde her sınıf için ayrı bir alt klasör olmalıdır (WAV, MP3).", font=ctk.CTkFont(size=11, slant="italic"), text_color="gray").pack(anchor="w", padx=20, pady=(0, 10))

        # Model Selection
        model_frame = ctk.CTkFrame(config_tab)
        model_frame.pack(fill="x", pady=10, padx=10)
        ctk.CTkLabel(model_frame, text="🤖 Model Seçimi", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=15, pady=10)
        
        model_grid = ctk.CTkFrame(model_frame, fg_color="transparent")
        model_grid.pack(fill="x", padx=10, pady=5)
        
        self.models_vars = {}
        models = [
            ("SVM", 0, 0), ("Random Forest", 0, 1), ("KNN", 0, 2),
            ("Logistic Regression", 1, 0), ("Decision Tree", 1, 1), ("Naive Bayes (Gaussian)", 1, 2)
        ]
        
        for name, r, c in models:
            var = ctk.CTkCheckBox(model_grid, text=name)
            var.grid(row=r, column=c, padx=10, pady=10, sticky="w")
            if name == "SVM": var.select()
            self.models_vars[name] = var

        # Parameters
        param_frame = ctk.CTkFrame(config_tab)
        param_frame.pack(fill="x", pady=10, padx=10)
        ctk.CTkLabel(param_frame, text="🎛️ Eğitim Parametreleri", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=15, pady=10)
        
        param_sub = ctk.CTkFrame(param_frame, fg_color="transparent")
        param_sub.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(param_sub, text="Test Verisi Oranı:").pack(side="left", padx=10)
        self.slider_test_size = ctk.CTkSlider(param_sub, from_=0.1, to=0.5, number_of_steps=40)
        self.slider_test_size.set(0.2)
        self.slider_test_size.pack(side="left", padx=10, fill="x", expand=True)
        self.lbl_test_val = ctk.CTkLabel(param_sub, text="0.20", width=40)
        self.lbl_test_val.pack(side="left", padx=10)
        self.slider_test_size.configure(command=lambda v: self.lbl_test_val.configure(text=f"{v:.2f}"))

        # Action
        action_frame = ctk.CTkFrame(config_tab, fg_color="transparent")
        action_frame.pack(fill="x", pady=20, padx=10)
        
        self.btn_train = ctk.CTkButton(action_frame, text="🚀 Eğitimi Başlat", command=self.start_training, fg_color="green", height=50, font=ctk.CTkFont(size=16, weight="bold"))
        self.btn_train.pack(fill="x")
        
        self.prog_bar = ctk.CTkProgressBar(action_frame)
        self.prog_bar.pack(fill="x", pady=(15, 0))
        self.prog_bar.set(0)

    def load_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.lbl_folder_path.configure(text=folder_path)
            self.folder_path = folder_path

    def start_training(self):
        if not hasattr(self, 'folder_path'):
            messagebox.showwarning("Uyarı", "Lütfen önce veri setinin olduğu klasörü seçin.")
            return

        test_size = self.slider_test_size.get()
        
        self.tab_view.set("📊 Sonuçlar")
        self.prog_bar.configure(mode="indeterminate")
        self.prog_bar.start()
        self.btn_train.configure(state="disabled", text="Eğitim Sürüyor...")

        threading.Thread(target=self._training_worker, args=(test_size,), daemon=True).start()

    def _training_worker(self, test_size):
        try:
            self.after(0, lambda: self._ui_log("🎵 Ses dosyaları işleniyor (Torchaudio)...", "cyan"))
            X, y = load_audio_from_folder(self.folder_path)
            
            if len(X) == 0:
                self.after(0, lambda: messagebox.showerror("Hata", "Klasörde uygun ses dosyası bulunamadı."))
                self.after(0, lambda: self.prog_bar.stop())
                self.after(0, lambda: self.btn_train.configure(state="normal", text="🚀 Eğitimi Başlat"))
                return

            le = LabelEncoder()
            y_encoded = le.fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=42)

            selected_models = {name: get_model(name) for name, var in self.models_vars.items() if var.get()}
            
            self.run_training_loop(selected_models, X_train, X_test, y_train, y_test)

        except Exception as e:
            self.after(0, lambda: self.prog_bar.stop())
            self.after(0, lambda: self.btn_train.configure(state="normal", text="🚀 Eğitimi Başlat"))
            self.after(0, lambda: messagebox.showerror("Hata", f"Ses yükleme hatası: {e}"))

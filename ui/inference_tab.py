import customtkinter as ctk
from tkinter import filedialog, messagebox
import os
import joblib
import numpy as np
import pandas as pd
from modules.data_loader import load_single_image, load_single_audio
from modules.languages import get_text

class InferenceTab(ctk.CTkFrame):
    def __init__(self, parent, lang="tr"):
        super().__init__(parent, corner_radius=0, fg_color="transparent")
        self.lang = lang
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self.model = None
        self.scaler = None
        self.vectorizer = None
        self.encoder = None
        self.label_encoder = None
        self.feature_cols = None
        self.tabular_entries = {}
        
        self.setup_ui()
        self.model_type = 'text'

    def tr(self, key):
        return get_text(key, self.lang)

    def setup_ui(self):
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # --- Yükleme Alanı ---
        load_frame = ctk.CTkFrame(main_frame)
        load_frame.pack(fill="x", pady=10, padx=10)
        ctk.CTkLabel(load_frame, text=self.tr("header_load"), font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=15, pady=10)

        # Model Tipi Seçimi
        type_frame = ctk.CTkFrame(load_frame, fg_color="transparent")
        type_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(type_frame, text=self.tr("lbl_model_type")).pack(side="left", padx=5)
        
        self.type_map = {
            self.tr("type_text"): "text",
            self.tr("type_image"): "image",
            self.tr("type_audio"): "audio",
            self.tr("type_tabular"): "tabular"
        }
        
        self.combo_type = ctk.CTkComboBox(type_frame, values=list(self.type_map.keys()), command=self.on_type_change)
        self.combo_type.set(self.tr("type_text"))
        self.combo_type.pack(side="left", padx=5)

        # Model Dosyası
        btn_frame = ctk.CTkFrame(load_frame, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=5)
        self.btn_load_model = ctk.CTkButton(btn_frame, text=self.tr("btn_select_model"), command=self.load_model_file)
        self.btn_load_model.pack(side="left", padx=5)
        self.lbl_model_path = ctk.CTkLabel(btn_frame, text=self.tr("lbl_no_file"), text_color="gray")
        self.lbl_model_path.pack(side="left", padx=10)

        # Ek Dosyalar (Vectorizer, Scaler, Encoder vb.)
        self.extra_files_frame = ctk.CTkFrame(load_frame, fg_color="transparent")
        self.extra_files_frame.pack(fill="x", padx=10, pady=5)
        
        self.btn_load_extra = ctk.CTkButton(self.extra_files_frame, text=self.tr("btn_load_vectorizer"), command=self.load_extra_file)
        self.btn_load_extra.pack(side="left", padx=5)
        self.lbl_extra_path = ctk.CTkLabel(self.extra_files_frame, text=self.tr("lbl_not_needed"), text_color="gray")
        self.lbl_extra_path.pack(side="left", padx=10)

        # Tablo için Ekstra Butonlar (Feature Cols, Label Encoder)
        self.tabular_files_frame = ctk.CTkFrame(load_frame, fg_color="transparent")
        
        # Feature Cols
        f1 = ctk.CTkFrame(self.tabular_files_frame, fg_color="transparent")
        f1.pack(fill="x", pady=2)
        self.btn_load_features = ctk.CTkButton(f1, text=self.tr("btn_load_features"), command=self.load_feature_cols, width=200)
        self.btn_load_features.pack(side="left", padx=5)
        self.lbl_features = ctk.CTkLabel(f1, text=self.tr("lbl_not_loaded"), text_color="gray")
        self.lbl_features.pack(side="left", padx=5)

        # Label Encoder
        f2 = ctk.CTkFrame(self.tabular_files_frame, fg_color="transparent")
        f2.pack(fill="x", pady=2)
        self.btn_load_label_enc = ctk.CTkButton(f2, text=self.tr("btn_load_label_enc"), command=self.load_label_encoder, width=200)
        self.btn_load_label_enc.pack(side="left", padx=5)
        self.lbl_label_enc = ctk.CTkLabel(f2, text=self.tr("lbl_not_loaded"), text_color="gray")
        self.lbl_label_enc.pack(side="left", padx=5)
        
        # Preprocessor
        f3 = ctk.CTkFrame(self.tabular_files_frame, fg_color="transparent")
        f3.pack(fill="x", pady=2)
        self.btn_load_encoder = ctk.CTkButton(f3, text=self.tr("btn_load_encoder"), command=self.load_onehot_encoder, width=200)
        self.btn_load_encoder.pack(side="left", padx=5)
        self.lbl_encoder = ctk.CTkLabel(f3, text=self.tr("lbl_not_loaded"), text_color="gray")
        self.lbl_encoder.pack(side="left", padx=5)

        # --- Tahmin Alanı ---
        pred_frame = ctk.CTkFrame(main_frame)
        pred_frame.pack(fill="both", expand=True, pady=10, padx=10)
        ctk.CTkLabel(pred_frame, text=self.tr("header_predict"), font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=15, pady=10)

        self.input_frame = ctk.CTkFrame(pred_frame, fg_color="transparent")
        self.input_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Metin Girişi
        self.text_input = ctk.CTkTextbox(self.input_frame, height=100)
        
        # Dosya Seçimi (Resim/Ses)
        self.file_input_frame = ctk.CTkFrame(self.input_frame, fg_color="transparent")
        self.btn_select_file = ctk.CTkButton(self.file_input_frame, text=self.tr("btn_file_select"), command=self.select_input_file)
        self.btn_select_file.pack(side="left", padx=5)
        self.lbl_input_file = ctk.CTkLabel(self.file_input_frame, text=self.tr("lbl_input_file"))
        self.lbl_input_file.pack(side="left", padx=10)

        # Tablo Girişi (Dinamik)
        self.tabular_input_frame = ctk.CTkScrollableFrame(self.input_frame, height=200, label_text=self.tr("lbl_tabular_input"))

        self.on_type_change(self.tr("type_text")) # Initialize

        self.btn_predict = ctk.CTkButton(pred_frame, text=self.tr("btn_predict"), command=self.predict, fg_color="green", height=40, font=ctk.CTkFont(size=14, weight="bold"))
        self.btn_predict.pack(pady=20)

        self.lbl_result = ctk.CTkLabel(pred_frame, text=self.tr("lbl_result_wait"), font=ctk.CTkFont(size=18, weight="bold"))
        self.lbl_result.pack(pady=10)

    def on_type_change(self, choice):
        self.text_input.pack_forget()
        self.file_input_frame.pack_forget()
        self.tabular_input_frame.pack_forget()
        self.tabular_files_frame.pack_forget()
        
        # Reset extra buttons text
        self.btn_load_extra.configure(state="normal")
        
        self.model_type = self.type_map.get(choice, "text")

        if self.model_type == 'text':
            self.text_input.pack(fill="x", padx=5, pady=5)
            self.btn_load_extra.configure(text=self.tr("btn_load_vectorizer"))
            self.lbl_extra_path.configure(text=self.tr("msg_load_vectorizer"))
            
        elif self.model_type == 'image':
            self.file_input_frame.pack(fill="x", padx=5, pady=5)
            self.btn_load_extra.configure(text=self.tr("btn_load_scaler"))
            self.lbl_extra_path.configure(text="Scaler (varsa)")
            
        elif self.model_type == 'audio':
            self.file_input_frame.pack(fill="x", padx=5, pady=5)
            self.btn_load_extra.configure(text=self.tr("btn_load_scaler"))
            self.lbl_extra_path.configure(text="Scaler (varsa)")
            
        elif self.model_type == 'tabular':
            self.tabular_input_frame.pack(fill="both", expand=True, padx=5, pady=5)
            self.tabular_files_frame.pack(fill="x", padx=10, pady=5)
            self.btn_load_extra.configure(state="disabled", text=self.tr("btn_load_extra"))
            self.lbl_extra_path.configure(text=self.tr("lbl_not_needed"))

    def load_model_file(self):
        path = filedialog.askopenfilename(filetypes=[("Joblib Files", "*.joblib")])
        if path:
            self.lbl_model_path.configure(text=os.path.basename(path))
            try:
                self.model = joblib.load(path)
                
                # Otomatik olarak aynı klasördeki yardımcı dosyaları ara ve yükle
                self._try_load_auxiliary_files(os.path.dirname(path))
                
                messagebox.showinfo(self.tr("msg_warning"), self.tr("msg_model_loaded"))
            except Exception as e:
                messagebox.showerror(self.tr("msg_error"), f"Model yüklenemedi: {e}")

    def _try_load_auxiliary_files(self, folder_path):
        """Aynı klasördeki vectorizer, scaler, encoder vb. dosyaları otomatik yükler."""
        loaded_files = []

        # 1. Vectorizer (Metin için)
        if self.model_type == 'text':
            vec_path = os.path.join(folder_path, 'vectorizer.joblib')
            if os.path.exists(vec_path):
                try:
                    self.vectorizer = joblib.load(vec_path)
                    self.lbl_extra_path.configure(text=f"vectorizer.joblib {self.tr('lbl_loaded_auto')}")
                    loaded_files.append("Vectorizer")
                except: pass

        # 2. Scaler (Görüntü/Ses için)
        if self.model_type in ['image', 'audio']:
            scaler_path = os.path.join(folder_path, 'scaler.joblib')
            if os.path.exists(scaler_path):
                try:
                    self.scaler = joblib.load(scaler_path)
                    self.lbl_extra_path.configure(text=f"scaler.joblib {self.tr('lbl_loaded_auto')}")
                    loaded_files.append("Scaler")
                except: pass

        # 3. Tabular Dosyalar (Feature Cols, Encoder, Label Encoder)
        if self.model_type == 'tabular':
            # Feature Cols
            fc_path = os.path.join(folder_path, 'feature_cols.joblib')
            if os.path.exists(fc_path):
                try:
                    self.feature_cols = joblib.load(fc_path)
                    self.create_tabular_form()
                    self.lbl_features.configure(text=f"feature_cols.joblib {self.tr('lbl_loaded_auto')}")
                    loaded_files.append("Feature Cols")
                except: pass
            
            # Preprocessor / Encoder
            enc_path = os.path.join(folder_path, 'encoder.joblib')
            if os.path.exists(enc_path):
                try:
                    self.encoder = joblib.load(enc_path)
                    self.lbl_encoder.configure(text=f"encoder.joblib {self.tr('lbl_loaded_auto')}")
                    loaded_files.append("Encoder")
                except: pass

            # Label Encoder
            le_path = os.path.join(folder_path, 'label_encoder.joblib')
            if os.path.exists(le_path):
                try:
                    self.label_encoder = joblib.load(le_path)
                    self.lbl_label_enc.configure(text=f"label_encoder.joblib {self.tr('lbl_loaded_auto')}")
                    loaded_files.append("Label Encoder")
                except: pass
        
        if loaded_files:
            # Bilgi amaçlı loglanabilir
            pass

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
                messagebox.showinfo(self.tr("msg_warning"), self.tr("msg_file_loaded"))
            except Exception as e:
                messagebox.showerror(self.tr("msg_error"), f"Dosya yüklenemedi: {e}")

    def load_feature_cols(self):
        path = filedialog.askopenfilename(filetypes=[("Joblib Files", "*.joblib")])
        if path:
            try:
                self.feature_cols = joblib.load(path)
                self.create_tabular_form()
                self.lbl_features.configure(text=os.path.basename(path))
                messagebox.showinfo(self.tr("msg_warning"), self.tr("msg_file_loaded"))
            except Exception as e:
                messagebox.showerror(self.tr("msg_error"), f"Yüklenemedi: {e}")

    def load_label_encoder(self):
        path = filedialog.askopenfilename(filetypes=[("Joblib Files", "*.joblib")])
        if path:
            try:
                self.label_encoder = joblib.load(path)
                self.lbl_label_enc.configure(text=os.path.basename(path))
                messagebox.showinfo(self.tr("msg_warning"), self.tr("msg_file_loaded"))
            except Exception as e:
                messagebox.showerror(self.tr("msg_error"), f"Yüklenemedi: {e}")

    def load_onehot_encoder(self):
        path = filedialog.askopenfilename(filetypes=[("Joblib Files", "*.joblib")])
        if path:
            try:
                self.encoder = joblib.load(path)
                self.lbl_encoder.configure(text=os.path.basename(path))
                messagebox.showinfo(self.tr("msg_warning"), self.tr("msg_file_loaded"))
            except Exception as e:
                messagebox.showerror(self.tr("msg_error"), f"Yüklenemedi: {e}")

    def create_tabular_form(self):
        for widget in self.tabular_input_frame.winfo_children():
            widget.destroy()
        
        self.tabular_entries = {}
        
        if not self.feature_cols:
            return

        for i, col in enumerate(self.feature_cols):
            ctk.CTkLabel(self.tabular_input_frame, text=col).grid(row=i, column=0, padx=10, pady=5, sticky="e")
            entry = ctk.CTkEntry(self.tabular_input_frame)
            entry.grid(row=i, column=1, padx=10, pady=5, sticky="w")
            self.tabular_entries[col] = entry

    def select_input_file(self):
        filetypes = [("Image Files", "*.jpg *.png *.jpeg")] if self.model_type == 'image' else [("Audio Files", "*.wav *.mp3 *.flac")]
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            self.lbl_input_file.configure(text=os.path.basename(path))
            self.input_file_path = path

    def predict(self):
        if not self.model:
            messagebox.showwarning(self.tr("msg_warning"), self.tr("msg_load_model_first"))
            return

        try:
            input_data = None
            
            if self.model_type == 'text':
                text = self.text_input.get("1.0", "end-1c").strip()
                if not text:
                    messagebox.showwarning(self.tr("msg_warning"), self.tr("msg_enter_text"))
                    return
                if not self.vectorizer:
                    messagebox.showwarning(self.tr("msg_warning"), self.tr("msg_load_vectorizer"))
                    return
                input_data = self.vectorizer.transform([text]).toarray()
            
            elif self.model_type == 'image':
                if not hasattr(self, 'input_file_path'):
                    messagebox.showwarning(self.tr("msg_warning"), self.tr("msg_select_image"))
                    return
                input_data = load_single_image(self.input_file_path)
                if self.scaler:
                    input_data = self.scaler.transform(input_data)

            elif self.model_type == 'audio':
                if not hasattr(self, 'input_file_path'):
                    messagebox.showwarning(self.tr("msg_warning"), self.tr("msg_select_audio"))
                    return
                input_data = load_single_audio(self.input_file_path)
                if self.scaler:
                    input_data = self.scaler.transform(input_data)
            
            elif self.model_type == 'tabular':
                if not self.feature_cols:
                    messagebox.showwarning(self.tr("msg_warning"), self.tr("msg_load_features"))
                    return
                
                # Verileri topla
                data = {}
                for col, entry in self.tabular_entries.items():
                    val = entry.get()
                    data[col] = [val] # DataFrame için liste içinde olmalı
                
                df = pd.DataFrame(data)
                
                # Sayısal olabilecek sütunları otomatik dönüştür (String -> Float/Int)
                for col in df.columns:
                    try:
                        df[col] = pd.to_numeric(df[col])
                    except (ValueError, TypeError):
                        pass # Dönüştürülemezse (örn. kategorik string) olduğu gibi bırak
                
                # Encoder varsa uygula
                if self.encoder:
                    try:
                        input_data = self.encoder.transform(df)
                    except Exception as e:
                        messagebox.showerror(self.tr("msg_error"), f"Encoding hatası: {e}\nGirdiğiniz değerler eğitim verisindeki kategorilerle eşleşmiyor olabilir.")
                        return
                else:
                    # Encoder yoksa sayısal olduğunu varsayıyoruz
                    input_data = df.values

            if input_data is None:
                raise ValueError(f"Girdi verisi oluşturulamadı. model_type={self.model_type}")
            
            prediction = self.model.predict(input_data)
            
            # Label Encoder varsa sonucu çöz
            result_text = prediction[0]
            if self.label_encoder:
                try:
                    result_text = self.label_encoder.inverse_transform([int(result_text)])[0]
                except:
                    pass

            confidence_text = ""
            if hasattr(self.model, "predict_proba"):
                try:
                    probs = self.model.predict_proba(input_data)
                    max_prob = np.max(probs)
                    confidence_text = f" (Güven: %{max_prob*100:.2f})"
                except Exception:
                    pass

            self.lbl_result.configure(text=f"Sonuç: {result_text}{confidence_text}", text_color="green")

        except Exception as e:
            import traceback
            traceback.print_exc()
            shape_info = getattr(input_data, 'shape', 'N/A')
            messagebox.showerror(self.tr("msg_error"), f"Tahmin hatası: {e}\nInput Shape: {shape_info}")

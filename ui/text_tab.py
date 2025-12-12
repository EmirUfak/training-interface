import customtkinter as ctk
from tkinter import filedialog, messagebox
import pandas as pd
import threading
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from ui.base_tab import BaseTrainingTab
from modules.model_trainer import get_model

class TextTrainingTab(BaseTrainingTab):
    def setup_config_tab(self):
        config_tab = self.tab_view.tab("⚙️ Yapılandırma")
        
        # Data Selection
        data_frame = ctk.CTkFrame(config_tab)
        data_frame.pack(fill="x", pady=10, padx=10)
        ctk.CTkLabel(data_frame, text="📁 Veri Seti Seçimi", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=15, pady=10)
        
        file_sub_frame = ctk.CTkFrame(data_frame, fg_color="transparent")
        file_sub_frame.pack(fill="x", padx=10, pady=5)
        
        self.btn_select_csv = ctk.CTkButton(file_sub_frame, text="CSV Dosyası Seç", command=self.load_csv, width=150)
        self.btn_select_csv.pack(side="left", padx=5)
        self.lbl_csv_path = ctk.CTkLabel(file_sub_frame, text="Dosya seçilmedi", text_color="gray")
        self.lbl_csv_path.pack(side="left", padx=10)

        col_sub_frame = ctk.CTkFrame(data_frame, fg_color="transparent")
        col_sub_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(col_sub_frame, text="Metin Sütunu:").pack(side="left", padx=5)
        self.combo_text_col = ctk.CTkComboBox(col_sub_frame, values=["Önce Dosya Seçin"], width=150)
        self.combo_text_col.pack(side="left", padx=5)

        ctk.CTkLabel(col_sub_frame, text="Etiket Sütunu:").pack(side="left", padx=(20, 5))
        self.combo_label_col = ctk.CTkComboBox(col_sub_frame, values=["Önce Dosya Seçin"], width=150)
        self.combo_label_col.pack(side="left", padx=5)

        # Model Selection
        model_frame = ctk.CTkFrame(config_tab)
        model_frame.pack(fill="x", pady=10, padx=10)
        ctk.CTkLabel(model_frame, text="🤖 Model Seçimi", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=15, pady=10)
        
        model_grid = ctk.CTkFrame(model_frame, fg_color="transparent")
        model_grid.pack(fill="x", padx=10, pady=5)
        
        self.models_vars = {}
        models = [
            ("Naive Bayes", 0, 0), ("SVM", 0, 1), ("Random Forest", 0, 2),
            ("Logistic Regression", 1, 0), ("Decision Tree", 1, 1), ("Gradient Boosting", 1, 2),
            ("KNN", 2, 0)
        ]
        
        for name, r, c in models:
            var = ctk.CTkCheckBox(model_grid, text=name)
            var.grid(row=r, column=c, padx=10, pady=10, sticky="w")
            if name in ["Naive Bayes", "SVM"]: var.select()
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

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.lbl_csv_path.configure(text=os.path.basename(file_path))
            self.csv_path = file_path
            try:
                df = pd.read_csv(file_path)
                columns = df.columns.tolist()
                self.combo_text_col.configure(values=columns)
                self.combo_label_col.configure(values=columns)
                if len(columns) > 0: self.combo_text_col.set(columns[0])
                if len(columns) > 1: self.combo_label_col.set(columns[1])
            except Exception as e:
                messagebox.showerror("Hata", f"Dosya okunamadı: {e}")

    def start_training(self):
        if not hasattr(self, 'csv_path'):
            messagebox.showwarning("Uyarı", "Lütfen önce bir CSV dosyası seçin.")
            return

        text_col = self.combo_text_col.get()
        label_col = self.combo_label_col.get()
        test_size = self.slider_test_size.get()
        
        self.tab_view.set("📊 Sonuçlar")
        self.prog_bar.configure(mode="indeterminate")
        self.prog_bar.start()
        self.btn_train.configure(state="disabled", text="Eğitim Sürüyor...")

        threading.Thread(target=self._training_worker, args=(text_col, label_col, test_size), daemon=True).start()

    def _training_worker(self, text_col, label_col, test_size):
        try:
            df = pd.read_csv(self.csv_path)
            X = df[text_col].astype(str)
            y = df[label_col]

            vectorizer = TfidfVectorizer(max_features=2000)
            X_vec = vectorizer.fit_transform(X).toarray()
            
            X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=test_size, random_state=42)

            selected_models = {name: get_model(name) for name, var in self.models_vars.items() if var.get()}
            
            self.run_training_loop(selected_models, X_train, X_test, y_train, y_test, vectorizer=vectorizer)

        except Exception as e:
            self.after(0, lambda: self.prog_bar.stop())
            self.after(0, lambda: self.btn_train.configure(state="normal", text="🚀 Eğitimi Başlat"))
            self.after(0, lambda: messagebox.showerror("Hata", f"Eğitim sırasında hata: {e}"))

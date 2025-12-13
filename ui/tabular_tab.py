import customtkinter as ctk
from tkinter import filedialog, messagebox
import pandas as pd
import os
import threading

from ui.base_tab import BaseTrainingTab
from modules.data_loader import load_categorical_data
from modules.model_trainer import get_model
from sklearn.model_selection import train_test_split

class TabularTrainingTab(BaseTrainingTab):
    def __init__(self, parent, lang="tr"):
        self.lang = lang
        self.param_config = {
            "Naive Bayes": {"alpha": ("float", self.tr("desc_alpha"))},
            "Naive Bayes (Gaussian)": {"var_smoothing": ("float", self.tr("desc_var_smoothing"))},
            "SVM": {
                "C": ("float", self.tr("desc_C")), 
                "kernel": (["linear", "rbf", "poly", "sigmoid"], self.tr("desc_kernel"))
            },
            "Random Forest": {
                "n_estimators": ("int", self.tr("desc_n_estimators")), 
                "max_depth": ("int_or_none", self.tr("desc_max_depth")), 
                "min_samples_split": ("int", self.tr("desc_min_samples_split"))
            },
            "Logistic Regression": {
                "C": ("float", self.tr("desc_C")), 
                "max_iter": ("int", self.tr("desc_max_iter"))
            },
            "Decision Tree": {
                "max_depth": ("int_or_none", self.tr("desc_max_depth")), 
                "min_samples_split": ("int", self.tr("desc_min_samples_split"))
            },
            "Decision Tree (Entropy)": {
                "max_depth": ("int_or_none", self.tr("desc_max_depth")), 
                "min_samples_split": ("int", self.tr("desc_min_samples_split"))
            },
            "Gradient Boosting": {
                "n_estimators": ("int", self.tr("desc_n_estimators")), 
                "learning_rate": ("float", self.tr("desc_learning_rate")), 
                "max_depth": ("int", self.tr("desc_max_depth"))
            },
            "KNN": {
                "n_neighbors": ("int", self.tr("desc_n_neighbors")), 
                "weights": (["uniform", "distance"], self.tr("desc_weights"))
            }
        }
        self.csv_path = None
        self.feature_vars = {}
        self.target_var = None
        
        super().__init__(parent, lang)
        
    def setup_config_tab(self):
        config_tab = self.tab_view.tab(self.tr("tab_config"))
        
        # Ana Scrollable Frame (Tüm içeriği sarmalar)
        main_scroll = ctk.CTkScrollableFrame(config_tab)
        main_scroll.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Dosya Seçimi
        self.btn_file = ctk.CTkButton(main_scroll, text=self.tr("btn_file_select"), command=self.select_file)
        self.btn_file.pack(pady=20)
        
        self.lbl_file = ctk.CTkLabel(main_scroll, text=self.tr("lbl_no_file"), text_color="gray")
        self.lbl_file.pack(pady=5)

        # Sütun Seçim Alanı (Scrollable) - Yüksekliği azalttık
        self.columns_frame = ctk.CTkScrollableFrame(main_scroll, label_text=self.tr("lbl_columns"), height=200)
        self.columns_frame.pack(fill="x", padx=10, pady=10)
        
        # Model Seçimi (Çoklu Seçim)
        model_frame = ctk.CTkFrame(main_scroll)
        model_frame.pack(fill="x", pady=10, padx=10)
        ctk.CTkLabel(model_frame, text=self.tr("header_models"), font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=15, pady=10)
        
        model_grid = ctk.CTkFrame(model_frame, fg_color="transparent")
        model_grid.pack(fill="x", padx=10, pady=5)
        
        self.models_vars = {}
        models = [
            ("Naive Bayes (Gaussian)", 0, 0), ("SVM", 0, 1), ("Random Forest", 0, 2),
            ("Logistic Regression", 1, 0), ("Decision Tree", 1, 1), ("Decision Tree (Entropy)", 1, 2),
            ("Gradient Boosting", 2, 0), ("KNN", 2, 1)
        ]
        
        for name, r, c in models:
            frame = ctk.CTkFrame(model_grid, fg_color="transparent")
            frame.grid(row=r, column=c, padx=10, pady=10, sticky="w")
            
            var = ctk.CTkCheckBox(frame, text=name)
            var.pack(side="left")
            if name in ["Random Forest", "Decision Tree (Entropy)"]: var.select()
            self.models_vars[name] = var
            
            if name in self.param_config:
                btn_settings = ctk.CTkButton(frame, text="⚙️", width=30, height=24, 
                                           command=lambda n=name: self.open_settings_window(n))
                btn_settings.pack(side="left", padx=(5, 0))

        # Eğitim Butonu
        action_frame = ctk.CTkFrame(main_scroll, fg_color="transparent")
        action_frame.pack(fill="x", pady=20, padx=10)

        self.var_optimize = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(action_frame, text=self.tr("chk_optimize"), variable=self.var_optimize).pack(pady=(0, 10))

        self.btn_train = ctk.CTkButton(action_frame, text=self.tr("btn_start_training"), command=self.start_training_thread, state="disabled", height=50, font=ctk.CTkFont(size=16, weight="bold"), fg_color="green")
        self.btn_train.pack(fill="x")

        self.prog_bar = ctk.CTkProgressBar(action_frame, mode="indeterminate")
        self.prog_bar.pack(fill="x", pady=(15, 0))
        self.prog_bar.set(0)

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.csv_path = file_path
            self.lbl_file.configure(text=os.path.basename(file_path))
            self.load_columns()
            self.btn_train.configure(state="normal")

    def load_columns(self):
        try:
            # Sadece başlıkları oku (Performans için nrows=0)
            df = pd.read_csv(self.csv_path, nrows=0)
            columns = df.columns.tolist()
            
            # Temizle
            for widget in self.columns_frame.winfo_children():
                widget.destroy()
            
            self.feature_vars = {}
            self.target_var = ctk.StringVar(value="")

            # Başlıklar
            ctk.CTkLabel(self.columns_frame, text=self.tr("col_feature"), font=("Arial", 12, "bold")).grid(row=0, column=0, padx=10, pady=5)
            ctk.CTkLabel(self.columns_frame, text=self.tr("col_name"), font=("Arial", 12, "bold")).grid(row=0, column=1, padx=10, pady=5)
            ctk.CTkLabel(self.columns_frame, text=self.tr("col_target"), font=("Arial", 12, "bold")).grid(row=0, column=2, padx=10, pady=5)

            for i, col in enumerate(columns):
                # Özellik Checkbox
                var = ctk.BooleanVar(value=True)
                self.feature_vars[col] = var
                chk = ctk.CTkCheckBox(self.columns_frame, text="", variable=var, width=20)
                chk.grid(row=i+1, column=0, padx=10)
                
                # İsim
                ctk.CTkLabel(self.columns_frame, text=col).grid(row=i+1, column=1, padx=10)
                
                # Hedef Radio
                rad = ctk.CTkRadioButton(self.columns_frame, text="", variable=self.target_var, value=col, width=20, command=self.update_features)
                rad.grid(row=i+1, column=2, padx=10)
            
            # Varsayılan olarak son sütunu hedef yap
            if columns:
                self.target_var.set(columns[-1])
                self.feature_vars[columns[-1]].set(False)

        except Exception as e:
            messagebox.showerror(self.tr("msg_error"), f"Dosya okunamadı: {e}")

    def update_features(self):
        # Hedef seçilen sütunun özellik işaretini kaldır
        target = self.target_var.get()
        if target in self.feature_vars:
            self.feature_vars[target].set(False)

    def start_training_thread(self):
        if not self.csv_path:
            return

        # UI verilerini ana thread'de oku
        target_col = self.target_var.get()
        feature_cols = [col for col, var in self.feature_vars.items() if var.get() and col != target_col]
        optimize = self.var_optimize.get()
        
        # Seçili modelleri al
        selected_models = [name for name, var in self.models_vars.items() if var.get()]

        # Doğrulamaları ana thread'de yap
        if not feature_cols:
            messagebox.showwarning(self.tr("msg_warning"), self.tr("msg_select_features"))
            return
        
        if not target_col:
            messagebox.showwarning(self.tr("msg_warning"), self.tr("msg_select_target"))
            return

        if not selected_models:
            messagebox.showwarning(self.tr("msg_warning"), self.tr("msg_select_model"))
            return

        self.tab_view.set(self.tr("tab_results"))
        self.btn_train.configure(state="disabled", text=self.tr("btn_training_running"))
        self.prog_bar.start()
        
        # Verileri thread'e argüman olarak gönder
        threading.Thread(target=self.start_training, args=(feature_cols, target_col, selected_models, optimize), daemon=True).start()

    def start_training(self, feature_cols, target_col, selected_models, optimize):
        try:
            # UI güncellemesini ana thread'e zamanla
            self.after(0, lambda: self.results_manager.log_message(self.tr("loading_data"), "yellow"))
            
            # Veriyi Yükle (Ağır işlem, thread'de kalmalı)
            X, y, encoder, label_encoder = load_categorical_data(self.csv_path, feature_cols, target_col)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Modelleri hazırla
            models_to_train = {}
            for name in selected_models:
                params = self.user_model_params.get(name, {})
                model = get_model(name, **params)
                if model is not None:
                    models_to_train[name] = model
            
            # BaseTrainingTab'daki loop'u kullan
            self.run_training_loop(models_to_train, X_train, X_test, y_train, y_test, 
                                 extra_data={
                                     "encoder": encoder, 
                                     "label_encoder": label_encoder,
                                     "feature_cols": feature_cols,
                                     "target_col": target_col,
                                     "model_type": "tabular"
                                 },
                                 optimize=optimize)
            
        except Exception as e:
            import traceback
            traceback.print_exc() # Konsola yazdır
            self.after(0, lambda: self._reset_ui_on_error())

    def _reset_ui_on_error(self):
        self.prog_bar.stop()
        self.btn_train.configure(state="normal", text=self.tr("btn_start_training"))

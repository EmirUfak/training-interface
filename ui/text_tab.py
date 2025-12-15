import customtkinter as ctk
from tkinter import filedialog, messagebox
import pandas as pd
import threading
import os
from sklearn.model_selection import train_test_split

from ui.base_tab import BaseTrainingTab
from modules.model_trainer import get_model
from modules.data_loader import load_and_vectorize_text

class TextTrainingTab(BaseTrainingTab):
    def __init__(self, parent, lang="tr"):
        self.lang = lang
        self.param_config = {
            "Naive Bayes": {"alpha": ("float", self.tr("desc_alpha"))},
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
        self.data_path = None
        
        super().__init__(parent, lang)

    def setup_config_tab(self):
        config_tab = self.tab_view.tab(self.tr("tab_config"))
        
        data_frame = ctk.CTkFrame(config_tab)
        data_frame.pack(fill="x", pady=10, padx=10)
        ctk.CTkLabel(data_frame, text=self.tr("header_dataset"), font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=15, pady=10)
        
        file_sub_frame = ctk.CTkFrame(data_frame, fg_color="transparent")
        file_sub_frame.pack(fill="x", padx=10, pady=5)
        
        self.btn_select_csv = ctk.CTkButton(file_sub_frame, text=self.tr("btn_file_select"), command=self.load_csv, width=150)
        self.btn_select_csv.pack(side="left", padx=5)
        self.lbl_csv_path = ctk.CTkLabel(file_sub_frame, text=self.tr("lbl_no_file"), text_color="gray")
        self.lbl_csv_path.pack(side="left", padx=10)

        col_sub_frame = ctk.CTkFrame(data_frame, fg_color="transparent")
        col_sub_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(col_sub_frame, text=self.tr("col_text")).pack(side="left", padx=5)
        self.combo_text_col = ctk.CTkComboBox(col_sub_frame, values=[self.tr("msg_select_file_first")], width=150)
        self.combo_text_col.pack(side="left", padx=5)

        ctk.CTkLabel(col_sub_frame, text=self.tr("col_target")).pack(side="left", padx=(20, 5))
        self.combo_label_col = ctk.CTkComboBox(col_sub_frame, values=[self.tr("msg_select_file_first")], width=150)
        self.combo_label_col.pack(side="left", padx=5)

        model_frame = ctk.CTkFrame(config_tab)
        model_frame.pack(fill="x", pady=10, padx=10)
        ctk.CTkLabel(model_frame, text=self.tr("header_models"), font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=15, pady=10)
        
        model_grid = ctk.CTkFrame(model_frame, fg_color="transparent")
        model_grid.pack(fill="x", padx=10, pady=5)
        
        self.models_vars = {}
        models = [
            ("Naive Bayes", 0, 0), ("SVM", 0, 1), ("Random Forest", 0, 2),
            ("Logistic Regression", 1, 0), ("Decision Tree", 1, 1), ("Gradient Boosting", 1, 2),
            ("KNN", 2, 0)
        ]
        
        for name, r, c in models:
            frame = ctk.CTkFrame(model_grid, fg_color="transparent")
            frame.grid(row=r, column=c, padx=10, pady=10, sticky="w")
            
            var = ctk.CTkCheckBox(frame, text=name)
            var.pack(side="left")
            if name in ["Naive Bayes", "SVM"]: var.select()
            self.models_vars[name] = var
            
            if name in self.param_config:
                btn_settings = ctk.CTkButton(frame, text="⚙️", width=30, height=24, 
                                           command=lambda n=name: self.open_settings_window(n))
                btn_settings.pack(side="left", padx=(5, 0))

        param_frame = ctk.CTkFrame(config_tab)
        param_frame.pack(fill="x", pady=10, padx=10)
        ctk.CTkLabel(param_frame, text=self.tr("header_params"), font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=15, pady=10)
        
        param_sub = ctk.CTkFrame(param_frame, fg_color="transparent")
        param_sub.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(param_sub, text=self.tr("lbl_test_size")).pack(side="left", padx=10)
        self.slider_test_size = ctk.CTkSlider(param_sub, from_=0.1, to=0.5, number_of_steps=40)
        self.slider_test_size.set(0.2)
        self.slider_test_size.pack(side="left", padx=10, fill="x", expand=True)
        self.lbl_test_val = ctk.CTkLabel(param_sub, text="0.20", width=40)
        self.lbl_test_val.pack(side="left", padx=10)
        self.slider_test_size.configure(command=lambda v: self.lbl_test_val.configure(text=f"{v:.2f}"))

        # --- Yeni Eklenen Parametreler (N-Gram ve Max Features) ---
        param_sub2 = ctk.CTkFrame(param_frame, fg_color="transparent")
        param_sub2.pack(fill="x", padx=10, pady=5)

        # Max Features
        ctk.CTkLabel(param_sub2, text="Max Kelime (Features):").pack(side="left", padx=5)
        self.entry_max_features = ctk.CTkEntry(param_sub2, width=60)
        self.entry_max_features.insert(0, "5000")
        self.entry_max_features.pack(side="left", padx=5)

        # N-Gram
        ctk.CTkLabel(param_sub2, text="N-Gram (Örn: 1,1 veya 1,2):").pack(side="left", padx=5)
        self.entry_ngram = ctk.CTkEntry(param_sub2, width=60)
        self.entry_ngram.insert(0, "1,1")
        self.entry_ngram.pack(side="left", padx=5)
        # ----------------------------------------------------------

        action_frame = ctk.CTkFrame(config_tab, fg_color="transparent")
        action_frame.pack(fill="x", pady=20, padx=10)
        
        self.var_optimize = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(action_frame, text=self.tr("chk_optimize"), variable=self.var_optimize).pack(pady=(0, 10))

        # --- Strategy Dropdown ---
        self.var_optimize_strategy = ctk.StringVar(value="Tüm Modeller")
        strategy_frame = ctk.CTkFrame(action_frame, fg_color="transparent")
        strategy_frame.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(strategy_frame, text="Grid Search Kapsamı:").pack(side="left", padx=5)
        ctk.CTkOptionMenu(strategy_frame, variable=self.var_optimize_strategy, values=["Tüm Modeller", "Sadece En İyi Model"]).pack(side="left", padx=5)
        # -------------------------

        self.btn_train = ctk.CTkButton(action_frame, text=self.tr("btn_start_training"), command=self.start_training, fg_color="green", height=50, font=ctk.CTkFont(size=16, weight="bold"))
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
                messagebox.showerror(self.tr("msg_error"), f"Dosya okunamadı: {e}")

    def start_training(self):
        if self.btn_train.cget("text") == self.tr("btn_stop_training"):
            self.stop_training()
            return

        if not hasattr(self, 'csv_path'):
            messagebox.showwarning(self.tr("msg_warning"), self.tr("msg_select_file_first"))
            return

        text_col = self.combo_text_col.get()
        label_col = self.combo_label_col.get()
        test_size = self.slider_test_size.get()
        
        # Parametreleri al
        try:
            max_feat = int(self.entry_max_features.get())
            ngram_txt = self.entry_ngram.get().split(',')
            ngram_range = (int(ngram_txt[0]), int(ngram_txt[1]))
        except ValueError:
            messagebox.showerror(self.tr("msg_error"), "Lütfen Max Features ve N-Gram için geçerli sayısal değerler girin.")
            return

        self.tab_view.set(self.tr("tab_results"))
        self.prog_bar.configure(mode="indeterminate")
        self.prog_bar.start()
        self.btn_train.configure(state="normal", text=self.tr("btn_stop_training"), fg_color="red", hover_color="#c0392b")
        
        optimize = self.var_optimize.get()
        strategy_val = self.var_optimize_strategy.get()
        strategy = "all" if strategy_val == "Tüm Modeller" else "best"

        threading.Thread(target=self._training_worker, args=(text_col, label_col, test_size, max_feat, ngram_range, optimize, strategy), daemon=True).start()

    def _training_worker(self, text_col, label_col, test_size, max_features, ngram_range, optimize, strategy):
        try:
            X_vec, y, vectorizer = load_and_vectorize_text(
                self.csv_path, 
                text_col, 
                label_col, 
                max_features=max_features, 
                ngram_range=ngram_range
            )
            
            X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=test_size, random_state=42)

            selected_models = [name for name, var in self.models_vars.items() if var.get()]
            models_to_train = {}
            for name in selected_models:
                params = self.user_model_params.get(name, {})
                model = get_model(name, **params)
                if model is not None:
                    models_to_train[name] = model
            
            self.run_training_loop(models_to_train, X_train, X_test, y_train, y_test, vectorizer=vectorizer, optimize=optimize, optimize_strategy=strategy)

        except Exception as e:
            self.after(0, lambda: self.prog_bar.stop())
            self.after(0, lambda: self.btn_train.configure(state="normal", text=self.tr("btn_start_training"), fg_color="green", hover_color="#2ecc71"))
            self.after(0, lambda: messagebox.showerror(self.tr("msg_error"), f"Eğitim sırasında hata: {e}"))

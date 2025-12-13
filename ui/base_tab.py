import customtkinter as ctk
from tkinter import messagebox
import threading
import os
import time
import joblib
import numpy as np

from modules.model_trainer import train_model, save_model
from ui.results_manager import ResultsManager
from modules.training_manager import TrainingManager

from modules.languages import get_text

class BaseTrainingTab(ctk.CTkFrame):
    def __init__(self, parent, lang="tr"):
        super().__init__(parent, corner_radius=0, fg_color="transparent")
        self.lang = lang
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self.user_model_params = {} # Kullanıcı parametreleri burada saklanır
        if not hasattr(self, 'param_config'):
            self.param_config = {} # Alt sınıflar bunu doldurmalı
        
        self.tab_view = ctk.CTkTabview(self)
        self.tab_view.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.tab_view.add(self.tr("tab_config"))
        self.tab_view.add(self.tr("tab_results"))
        
        self.setup_config_tab()
        self.setup_results_tab()

    def tr(self, key):
        return get_text(key, self.lang)

    def open_settings_window(self, model_name):
        if model_name not in self.param_config:
            return
            
        config = self.param_config[model_name]
        current_params = self.user_model_params.get(model_name, {})
        
        window = ctk.CTkToplevel(self)
        window.title(f"{model_name} {self.tr('win_settings_title')}")
        window.geometry("500x600")
        window.grab_set() # Modality
        
        ctk.CTkLabel(window, text=f"{model_name} {self.tr('lbl_params')}", font=("Arial", 16, "bold")).pack(pady=10)
        
        entries = {}
        
        scroll = ctk.CTkScrollableFrame(window)
        scroll.pack(fill="both", expand=True, padx=10, pady=10)
        
        for param, config_val in config.items():
            # Config değerini ayrıştır (Tip, Açıklama)
            if isinstance(config_val, tuple) and len(config_val) == 2:
                p_type, p_desc = config_val
            else:
                p_type = config_val
                p_desc = ""

            # Satır Frame
            row = ctk.CTkFrame(scroll, fg_color="transparent")
            row.pack(fill="x", pady=(5, 0))
            
            # Parametre Adı
            ctk.CTkLabel(row, text=param, width=140, anchor="w", font=("Arial", 12, "bold")).pack(side="left", padx=5)
            
            val = current_params.get(param, "")
            
            # Giriş Alanı (Dropdown veya Entry)
            if isinstance(p_type, list): # Dropdown
                var = ctk.StringVar(value=str(val) if val else str(p_type[0]))
                entry = ctk.CTkOptionMenu(row, variable=var, values=[str(x) for x in p_type])
                entry.pack(side="right", expand=True, fill="x", padx=5)
                entries[param] = var
            else: # Entry
                entry = ctk.CTkEntry(row)
                entry.pack(side="right", expand=True, fill="x", padx=5)
                if val is not None:
                    entry.insert(0, str(val))
                entries[param] = entry
            
            # Açıklama Satırı (Varsa)
            if p_desc:
                desc_row = ctk.CTkFrame(scroll, fg_color="transparent")
                desc_row.pack(fill="x", pady=(0, 5))
                ctk.CTkLabel(desc_row, text=f"ℹ️ {p_desc}", font=("Arial", 11), text_color="gray", wraplength=300, justify="left").pack(anchor="w", padx=(150, 5))
            elif p_type == "int_or_none":
                desc_row = ctk.CTkFrame(scroll, fg_color="transparent")
                desc_row.pack(fill="x", pady=(0, 5))
                ctk.CTkLabel(desc_row, text=self.tr("val_int_or_none"), font=("Arial", 11), text_color="gray", wraplength=300, justify="left").pack(anchor="w", padx=(150, 5))
            else:
                # Boşluk bırak
                ctk.CTkFrame(scroll, fg_color="transparent", height=5).pack(fill="x")

        def save_params():
            new_params = {}
            for param, widget in entries.items():
                val = widget.get()
                if val.strip() == "":
                    continue
                new_params[param] = val
            
            self.user_model_params[model_name] = new_params
            window.destroy()
            
        ctk.CTkButton(window, text=self.tr("btn_save"), command=save_params, fg_color="green").pack(pady=10)

    def setup_config_tab(self):
        pass

    def setup_results_tab(self):
        results_tab = self.tab_view.tab(self.tr("tab_results"))
        self.results_area = ctk.CTkScrollableFrame(results_tab)
        self.results_area.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.results_manager = ResultsManager(self.results_area)

        btn_clear = ctk.CTkButton(results_tab, text=self.tr("btn_clear_results"), command=self.clear_results, fg_color="#c0392b", hover_color="#e74c3c")
        btn_clear.pack(pady=10)

    def clear_results(self):
        for widget in self.results_area.winfo_children():
            widget.destroy()

    def run_training_loop(self, models, X_train, X_test, y_train, y_test, vectorizer=None, apply_scaling=False, extra_data=None, optimize=False):
        manager = TrainingManager(
            log_callback=lambda msg, color: self.after(0, lambda: self.results_manager.log_message(msg, color)),
            result_callback=lambda n, r, i, y, s: self.after(0, lambda: self.results_manager.show_model_result(n, r, i, y, s)),
            comparison_callback=lambda r, s: self.after(0, lambda: self.results_manager.show_comparison(r, s)),
            best_model_callback=lambda d, f: self.after(0, lambda: self.results_manager.show_best_model(d, f)),
            completion_callback=lambda s: self.after(0, lambda: self._on_training_complete(s)),
            error_callback=lambda n, e: self.after(0, lambda: self.results_manager.log_message(f"❌ {n} {self.tr('msg_error')}: {e}", "red"))
        )
        manager.run_training_loop(models, X_train, X_test, y_train, y_test, vectorizer, apply_scaling, extra_data, optimize)

    def _on_training_complete(self, save_dir):
        if hasattr(self, 'prog_bar'):
            self.prog_bar.stop()
        
        if hasattr(self, 'btn_train'):
            self.btn_train.configure(state="normal", text=self.tr("btn_start_training"))
            
        messagebox.showinfo(self.tr("msg_warning"), self.tr("msg_training_complete").format(save_dir=save_dir))

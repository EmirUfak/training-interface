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

class BaseTrainingTab(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, corner_radius=0, fg_color="transparent")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self.tab_view = ctk.CTkTabview(self)
        self.tab_view.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.tab_view.add("⚙️ Yapılandırma")
        self.tab_view.add("📊 Sonuçlar")
        
        self.setup_config_tab()
        self.setup_results_tab()

    def setup_config_tab(self):
        pass

    def setup_results_tab(self):
        results_tab = self.tab_view.tab("📊 Sonuçlar")
        self.results_area = ctk.CTkScrollableFrame(results_tab)
        self.results_area.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.results_manager = ResultsManager(self.results_area)

        btn_clear = ctk.CTkButton(results_tab, text="🗑️ Sonuçları Temizle", command=self.clear_results, fg_color="#c0392b", hover_color="#e74c3c")
        btn_clear.pack(pady=10)

    def clear_results(self):
        for widget in self.results_area.winfo_children():
            widget.destroy()

    def run_training_loop(self, models, X_train, X_test, y_train, y_test, vectorizer=None, apply_scaling=False):
        manager = TrainingManager(
            log_callback=lambda msg, color: self.after(0, lambda: self.results_manager.log_message(msg, color)),
            result_callback=lambda n, r, i, y, s: self.after(0, lambda: self.results_manager.show_model_result(n, r, i, y, s)),
            comparison_callback=lambda r, s: self.after(0, lambda: self.results_manager.show_comparison(r, s)),
            best_model_callback=lambda d, f: self.after(0, lambda: self.results_manager.show_best_model(d, f)),
            completion_callback=lambda s: self.after(0, lambda: self._on_training_complete(s)),
            error_callback=lambda n, e: self.after(0, lambda: self.results_manager.log_message(f"❌ {n} Hatası: {e}", "red"))
        )
        manager.run_training_loop(models, X_train, X_test, y_train, y_test, vectorizer, apply_scaling)

    def _on_training_complete(self, save_dir):
        self.prog_bar.stop()
        self.btn_train.configure(state="normal", text="🚀 Eğitimi Başlat")
        messagebox.showinfo("Başarılı", f"Tüm eğitimler tamamlandı!\nSonuçlar ve veri setleri '{save_dir}' klasörüne kaydedildi.")

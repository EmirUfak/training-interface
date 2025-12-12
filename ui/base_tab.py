import customtkinter as ctk
from tkinter import messagebox
import threading
import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from modules.model_trainer import train_model, save_model, get_model
from modules.visualization import create_confusion_matrix_figure, create_feature_importance_figure, create_comparison_figure

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
        pass # To be implemented by subclasses

    def setup_results_tab(self):
        results_tab = self.tab_view.tab("📊 Sonuçlar")
        self.results_area = ctk.CTkScrollableFrame(results_tab)
        self.results_area.pack(fill="both", expand=True, padx=5, pady=5)
        
        btn_clear = ctk.CTkButton(results_tab, text="🗑️ Sonuçları Temizle", command=self.clear_results, fg_color="#c0392b", hover_color="#e74c3c")
        btn_clear.pack(pady=10)

    def clear_results(self):
        for widget in self.results_area.winfo_children():
            widget.destroy()

    def run_training_loop(self, models, X_train, X_test, y_train, y_test, vectorizer=None):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        save_dir = f"training_results_{timestamp}"
        os.makedirs(save_dir, exist_ok=True)

        if vectorizer:
            joblib.dump(vectorizer, os.path.join(save_dir, 'vectorizer.joblib'))

        results = []
        best_f1 = -1
        best_model_data = None

        for name, model in models.items():
            self.after(0, lambda n=name: self._ui_log(f"⏳ {n} eğitiliyor...", "yellow"))
            
            try:
                res = train_model(model, X_train, y_train, X_test, y_test)
                results.append({"Model": name, "Accuracy": res["accuracy"], "F1-Score": res["f1"]})
                save_model(model, os.path.join(save_dir, f'{name}_model.joblib'))

                if res["f1"] > best_f1:
                    best_f1 = res["f1"]
                    best_model_data = {
                        "name": name,
                        "model": model,
                        "acc": res["accuracy"],
                        "f1": res["f1"]
                    }

                imp_data = None
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    if vectorizer:
                        feature_names = np.array(vectorizer.get_feature_names_out())
                    else:
                        feature_names = np.array([f"Pixel {i}" for i in range(X_train.shape[1])])
                    imp_data = (importances, feature_names)

                self.after(0, lambda n=name, r=res, imp=imp_data, yt=y_test: self._ui_show_model_result(n, r, imp, yt, save_dir))
            
            except Exception as e:
                self.after(0, lambda n=name, err=e: self._ui_log(f"❌ {n} Hatası: {err}", "red"))

        if len(results) > 1:
            self.after(0, lambda: self._ui_show_comparison(results, save_dir))
        
        if best_model_data:
            best_name = best_model_data["name"]
            best_filename = f"best_model_{best_name}_{timestamp}.joblib"
            save_model(best_model_data["model"], os.path.join(save_dir, best_filename))
            self.after(0, lambda: self._ui_show_best_model(best_model_data, best_filename))

        self.after(0, lambda: self.prog_bar.stop())
        self.after(0, lambda: self.btn_train.configure(state="normal", text="🚀 Eğitimi Başlat"))
        self.after(0, lambda: messagebox.showinfo("Başarılı", f"Tüm eğitimler tamamlandı!\nSonuçlar '{save_dir}' klasörüne kaydedildi."))

    def _ui_log(self, text, color):
        lbl = ctk.CTkLabel(self.results_area, text=text, text_color=color, font=ctk.CTkFont(family="Consolas", size=12))
        lbl.pack(anchor="w", padx=5, pady=2)

    def _ui_show_model_result(self, name, res, imp_data, y_test, save_dir):
        res_frame = ctk.CTkFrame(self.results_area, fg_color="#2b2b2b", corner_radius=10)
        res_frame.pack(fill="x", pady=10, padx=5)
        
        header_frame = ctk.CTkFrame(res_frame, fg_color="transparent")
        header_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(header_frame, text=f"✅ {name}", font=ctk.CTkFont(size=16, weight="bold"), text_color="#2ecc71").pack(side="left")
        ctk.CTkLabel(header_frame, text=f"F1: {res['f1']:.4f}", font=ctk.CTkFont(size=14, weight="bold")).pack(side="right")

        result_text = (f"Doğruluk (Accuracy): {res['accuracy']:.4f}  |  Hassasiyet (Precision): {res['precision']:.4f}  |  Duyarlılık (Recall): {res['recall']:.4f}")
        ctk.CTkLabel(res_frame, text=result_text, font=ctk.CTkFont(size=12)).pack(anchor="w", padx=15, pady=2)

        plot_frame = ctk.CTkFrame(res_frame, fg_color="transparent")
        plot_frame.pack(fill="x", padx=10, pady=10)

        # Confusion Matrix
        fig = create_confusion_matrix_figure(y_test, res['y_pred'])
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side="left", padx=10, expand=True)
        fig.savefig(os.path.join(save_dir, f'{name}_confusion_matrix.png'))
        plt.close(fig)

        # Feature Importance
        if imp_data:
            try:
                importances, feature_names = imp_data
                fig_imp = create_feature_importance_figure(importances, feature_names)
                canvas_imp = FigureCanvasTkAgg(fig_imp, master=plot_frame)
                canvas_imp.draw()
                canvas_imp.get_tk_widget().pack(side="left", padx=10, expand=True)
                fig_imp.savefig(os.path.join(save_dir, f'{name}_feature_importance.png'))
                plt.close(fig_imp)
            except Exception:
                pass

    def _ui_show_comparison(self, results, save_dir):
        fig = create_comparison_figure(pd.DataFrame(results))
        
        comp_frame = ctk.CTkFrame(self.results_area, corner_radius=10)
        comp_frame.pack(fill="x", pady=20, padx=5)
        ctk.CTkLabel(comp_frame, text="🏆 Genel Karşılaştırma", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        canvas = FigureCanvasTkAgg(fig, master=comp_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)
        
        fig.savefig(os.path.join(save_dir, 'model_comparison.png'))
        plt.close(fig)
        
        pd.DataFrame(results).to_csv(os.path.join(save_dir, 'results_summary.csv'), index=False)

    def _ui_show_best_model(self, data, filename):
        frame = ctk.CTkFrame(self.results_area, fg_color="#1a4d1a", border_color="#00ff00", border_width=2, corner_radius=15)
        frame.pack(fill="x", pady=20, padx=5)
        
        ctk.CTkLabel(frame, text="🌟 EN BAŞARILI MODEL 🌟", font=ctk.CTkFont(size=20, weight="bold"), text_color="#00ff00").pack(pady=(15, 5))
        
        details = (f"Model: {data['name']}\n"
                   f"F1 Skoru: {data['f1']:.4f}\n"
                   f"Doğruluk: {data['acc']:.4f}")
                   
        ctk.CTkLabel(frame, text=details, font=ctk.CTkFont(size=16, weight="bold"), justify="center").pack(pady=5)
        ctk.CTkLabel(frame, text=f"Dosya: {filename}", font=ctk.CTkFont(size=12), text_color="lightgray").pack(pady=(0, 15))

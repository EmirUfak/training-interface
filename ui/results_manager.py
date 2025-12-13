import customtkinter as ctk
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from modules.visualization import create_confusion_matrix_figure, create_feature_importance_figure, create_comparison_figure

class ResultsManager:
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame

    def log_message(self, text, color):
        lbl = ctk.CTkLabel(self.parent_frame, text=text, text_color=color, font=ctk.CTkFont(family="Consolas", size=12))
        lbl.pack(anchor="w", padx=5, pady=2)

    def show_model_result(self, name, res, imp_data, y_test, save_dir):
        res_frame = ctk.CTkFrame(self.parent_frame, fg_color="#2b2b2b", corner_radius=10)
        res_frame.pack(fill="x", pady=10, padx=5)
        
        header_frame = ctk.CTkFrame(res_frame, fg_color="transparent")
        header_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(header_frame, text=f"✅ {name}", font=ctk.CTkFont(size=16, weight="bold"), text_color="#2ecc71").pack(side="left")
        ctk.CTkLabel(header_frame, text=f"F1: {res['f1']:.4f}", font=ctk.CTkFont(size=14, weight="bold")).pack(side="right")

        result_text = (f"Doğruluk (Accuracy): {res['accuracy']:.4f}  |  Hassasiyet (Precision): {res['precision']:.4f}  |  Duyarlılık (Recall): {res['recall']:.4f}")
        ctk.CTkLabel(res_frame, text=result_text, font=ctk.CTkFont(size=12)).pack(anchor="w", padx=15, pady=2)

        plot_frame = ctk.CTkFrame(res_frame, fg_color="transparent")
        plot_frame.pack(fill="x", padx=10, pady=10)

        fig = create_confusion_matrix_figure(y_test, res['y_pred'])
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side="left", padx=10, expand=True)
        fig.savefig(os.path.join(save_dir, f'{name}_confusion_matrix.png'))
        plt.close(fig)

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
        
        # Karar Ağacı Görselleştirmesi (Entropy/Gini)
        if "Decision Tree" in name:
            try:
                from sklearn.tree import plot_tree
                fig_tree = plt.figure(figsize=(12, 6))
                plot_tree(res['model'], max_depth=3, feature_names=imp_data[1] if imp_data else None, filled=True, fontsize=8)
                plt.title(f"{name} Yapısı (İlk 3 Seviye)")
                
                canvas_tree = FigureCanvasTkAgg(fig_tree, master=plot_frame)
                canvas_tree.draw()
                canvas_tree.get_tk_widget().pack(side="left", padx=10, expand=True)
                fig_tree.savefig(os.path.join(save_dir, f'{name}_structure.png'))
                plt.close(fig_tree)
            except Exception as e:
                print(f"Tree plot error: {e}")

    def show_comparison(self, results, save_dir):
        fig = create_comparison_figure(pd.DataFrame(results))
        
        comp_frame = ctk.CTkFrame(self.parent_frame, corner_radius=10)
        comp_frame.pack(fill="x", pady=20, padx=5)
        ctk.CTkLabel(comp_frame, text="🏆 Genel Karşılaştırma", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        canvas = FigureCanvasTkAgg(fig, master=comp_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)
        
        fig.savefig(os.path.join(save_dir, 'model_comparison.png'))
        plt.close(fig)
        
        pd.DataFrame(results).to_csv(os.path.join(save_dir, 'results_summary.csv'), index=False)

    def show_best_model(self, data, filename):
        frame = ctk.CTkFrame(self.parent_frame, fg_color="#1a4d1a", border_color="#00ff00", border_width=2, corner_radius=15)
        frame.pack(fill="x", pady=20, padx=5)
        
        ctk.CTkLabel(frame, text="EN BAŞARILI MODEL", font=ctk.CTkFont(size=20, weight="bold"), text_color="#00ff00").pack(pady=(15, 5))
        
        details = (f"Model: {data['name']}\n"
                   f"F1 Skoru: {data['f1']:.4f}\n"
                   f"Doğruluk: {data['acc']:.4f}")
                   
        ctk.CTkLabel(frame, text=details, font=ctk.CTkFont(size=16, weight="bold"), justify="center").pack(pady=5)
        ctk.CTkLabel(frame, text=f"Dosya: {filename}", font=ctk.CTkFont(size=12), text_color="lightgray").pack(pady=(0, 15))

import os
import time
import joblib
import numpy as np
from modules.model_trainer import train_model, save_model

class TrainingManager:
    def __init__(self, log_callback=None, result_callback=None, comparison_callback=None, best_model_callback=None, completion_callback=None, error_callback=None):
        self.log_callback = log_callback
        self.result_callback = result_callback
        self.comparison_callback = comparison_callback
        self.best_model_callback = best_model_callback
        self.completion_callback = completion_callback
        self.error_callback = error_callback

    def run_training_loop(self, models, X_train, X_test, y_train, y_test, vectorizer=None):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        save_dir = f"training_results_{timestamp}"
        os.makedirs(save_dir, exist_ok=True)

        try:
            joblib.dump(X_train, os.path.join(save_dir, 'X_train.joblib'))
            joblib.dump(X_test, os.path.join(save_dir, 'X_test.joblib'))
            joblib.dump(y_train, os.path.join(save_dir, 'y_train.joblib'))
            joblib.dump(y_test, os.path.join(save_dir, 'y_test.joblib'))
            if self.log_callback:
                self.log_callback(f"💾 Veri setleri (X_train, X_test, vb.) '{save_dir}' içine kaydedildi.", "green")
        except Exception as e:
            if self.log_callback:
                self.log_callback(f"⚠️ Veri setleri kaydedilemedi: {e}", "orange")

        if vectorizer:
            joblib.dump(vectorizer, os.path.join(save_dir, 'vectorizer.joblib'))

        results = []
        best_f1 = -1
        best_model_data = None

        for name, model in models.items():
            if self.log_callback:
                self.log_callback(f"⏳ {name} eğitiliyor...", "yellow")
            
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

                if self.result_callback:
                    self.result_callback(name, res, imp_data, y_test, save_dir)
            
            except Exception as e:
                if self.error_callback:
                    self.error_callback(name, e)
                elif self.log_callback:
                    self.log_callback(f"❌ {name} Hatası: {e}", "red")

        if len(results) > 1 and self.comparison_callback:
            self.comparison_callback(results, save_dir)
        
        if best_model_data:
            best_name = best_model_data["name"]
            best_filename = f"best_model_{best_name}_{timestamp}.joblib"
            save_model(best_model_data["model"], os.path.join(save_dir, best_filename))
            if self.best_model_callback:
                self.best_model_callback(best_model_data, best_filename)

        if self.completion_callback:
            self.completion_callback(save_dir)

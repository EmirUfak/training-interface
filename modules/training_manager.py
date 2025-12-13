import os
import time
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from modules.model_trainer import train_model, save_model

class TrainingManager:
    def __init__(self, log_callback=None, result_callback=None, comparison_callback=None, best_model_callback=None, completion_callback=None, error_callback=None):
        self.log_callback = log_callback
        self.result_callback = result_callback
        self.comparison_callback = comparison_callback
        self.best_model_callback = best_model_callback
        self.completion_callback = completion_callback
        self.error_callback = error_callback

    def run_training_loop(self, models, X_train, X_test, y_train, y_test, vectorizer=None, apply_scaling=False, extra_data=None, optimize=False):
        print("DEBUG: TrainingManager.run_training_loop started") # DEBUG LOG
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        save_dir = f"training_results_{timestamp}"
        os.makedirs(save_dir, exist_ok=True)

        if extra_data:
            try:
                for key, value in extra_data.items():
                    joblib.dump(value, os.path.join(save_dir, f'{key}.joblib'))
                if self.log_callback:
                    self.log_callback(f"💾 Ek veriler ({', '.join(extra_data.keys())}) kaydedildi.", "green")
            except Exception as e:
                if self.log_callback:
                    self.log_callback(f"⚠️ Ek veriler kaydedilemedi: {e}", "orange")

        if apply_scaling:
            try:
                if self.log_callback:
                    self.log_callback("⚖️ Veriler ölçeklendiriliyor (StandardScaler)...", "cyan")
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                joblib.dump(scaler, os.path.join(save_dir, 'scaler.joblib'))
            except Exception as e:
                if self.log_callback:
                    self.log_callback(f"⚠️ Ölçeklendirme hatası: {e}", "orange")

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
                # 1. EĞİTİM
                res = train_model(model, X_train, y_train, X_test, y_test, optimize=optimize, model_name=name)
                results.append({"Model": name, "Accuracy": res["accuracy"], "F1-Score": res["f1"]})
                
                # 2. KAYDETME (Hata olursa devam et)
                try:
                    save_model(model, os.path.join(save_dir, f'{name}_model.joblib'))
                except Exception as e:
                    if self.log_callback:
                        self.log_callback(f"⚠️ {name} kaydedilemedi: {e}", "orange")

                if res["f1"] > best_f1:
                    best_f1 = res["f1"]
                    best_model_data = {
                        "name": name,
                        "model": model,
                        "acc": res["accuracy"],
                        "f1": res["f1"]
                    }

                imp_data = None
                # 3. ÖZELLİK ÖNEMİ
                try:
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                        if vectorizer:
                            feature_names = np.array(vectorizer.get_feature_names_out())
                        elif extra_data and 'feature_cols' in extra_data:
                            feature_names = np.array(extra_data['feature_cols'])
                        else:
                            feature_names = np.array([f"Feature {i}" for i in range(X_train.shape[1])])
                        
                        # Boyut kontrolü
                        if len(feature_names) != len(importances):
                            if self.log_callback:
                                self.log_callback(f"⚠️ {name}: Özellik ismi sayısı ({len(feature_names)}) ile önem skoru sayısı ({len(importances)}) uyuşmuyor.", "orange")
                            # Feature names'i yeniden oluştur
                            feature_names = np.array([f"Feature {i}" for i in range(len(importances))])

                        imp_data = (importances, feature_names)
                except Exception as e:
                    if self.log_callback:
                        self.log_callback(f"⚠️ {name} özellik önemleri hesaplanırken hata: {e}", "orange")

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
            try:
                best_name = best_model_data["name"]
                best_filename = f"best_model_{best_name}_{timestamp}.joblib"
                save_model(best_model_data["model"], os.path.join(save_dir, best_filename))
                if self.best_model_callback:
                    self.best_model_callback(best_model_data, best_filename)
            except Exception as e:
                if self.log_callback:
                    self.log_callback(f"⚠️ En iyi model ({best_model_data['name']}) kaydedilirken hata: {e}", "orange")

        if self.completion_callback:
            self.completion_callback(save_dir)

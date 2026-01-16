import os
import time
import json
import logging
import threading
import joblib
import numpy as np
from typing import Any, Dict, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from modules.model_trainer import train_model, save_model

logger = logging.getLogger(__name__)


class TrainingManager:
    def __init__(self, log_callback=None, result_callback=None, comparison_callback=None, best_model_callback=None, completion_callback=None, error_callback=None, stop_check=None):
        self.log_callback = log_callback
        self.result_callback = result_callback
        self.comparison_callback = comparison_callback
        self.best_model_callback = best_model_callback
        self.completion_callback = completion_callback
        self.error_callback = error_callback
        self.stop_check = stop_check

    def run_training_loop(
        self,
        models: Dict[str, Any],
        X_train,
        X_test,
        y_train,
        y_test,
        vectorizer=None,
        apply_scaling: bool = False,
        extra_data: Optional[Dict[str, Any]] = None,
        optimize: bool = False,
        optimize_strategy: str = "all",
        batch_mode: bool = False,
        lazy_loader=None,
        epochs: int = 5,
        task_type: str = "classification",
        cv_folds: Optional[int] = None,
        save_options: Optional[Dict[str, bool]] = None,
    ):
        logger.info("TrainingManager.run_training_loop started")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        results_root = "results"
        save_dir = os.path.join(results_root, f"training_results_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)

        def _opt(key: str, default: bool = True) -> bool:
            if not save_options:
                return default
            return save_options.get(key, default)

        if extra_data and _opt("save_extra"):
            try:
                for key, value in extra_data.items():
                    joblib.dump(value, os.path.join(save_dir, f'{key}.joblib'))
                if self.log_callback:
                    self.log_callback(f"💾 Ek veriler ({', '.join(extra_data.keys())}) kaydedildi.", "green")
            except Exception as e:
                if self.log_callback:
                    self.log_callback(f"⚠️ Ek veriler kaydedilemedi: {e}", "orange")

        if apply_scaling and _opt("save_scaler"):
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

        if _opt("save_datasets"):
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

        if vectorizer and _opt("save_vectorizer"):
            joblib.dump(vectorizer, os.path.join(save_dir, 'vectorizer.joblib'))

        results = []
        best_score = -float('inf') # F1 or R2
        best_model_data = None

        # Strateji Belirleme
        do_optimize_all = optimize and (optimize_strategy == "all")
        do_optimize_best = optimize and (optimize_strategy == "best")

        for name, model in models.items():
            if self.stop_check and self.stop_check():
                if self.log_callback:
                    self.log_callback("🛑 Eğitim kullanıcı tarafından durduruldu.", "red")
                break

            if self.log_callback:
                self.log_callback(f"⏳ {name} eğitiliyor...", "yellow")

            stop_event = threading.Event()
            heartbeat_thread = None
            if self.log_callback:
                def _heartbeat():
                    while not stop_event.wait(10):
                        self.log_callback(f"⏳ {name} eğitim sürüyor...", "cyan")
                heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
                heartbeat_thread.start()
            
            try:
                # 1. EĞİTİM
                res = None
                
                if batch_mode and lazy_loader:
                    res = self._train_incremental(model, name, X_train, y_train, X_test, y_test, lazy_loader, epochs, save_dir)
                else:
                    # Eğer "all" seçiliyse her model optimize edilir, "best" ise ilk turda optimize edilmez.
                    res = train_model(
                        model,
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        optimize=do_optimize_all,
                        model_name=name,
                        task_type=task_type,
                        cv_folds=cv_folds,
                     )
                
                if task_type == "classification":
                    results.append({"Model": name, "Accuracy": res["accuracy"], "F1-Score": res["f1"]})
                    score = res["f1"]
                else:
                    results.append({"Model": name, "R2 Score": res["r2"], "MSE": res["mse"]})
                    score = res["r2"]
                
                # 2. KAYDETME (Hata olursa devam et)
                if _opt("save_models"):
                    try:
                        save_model(res["model"], os.path.join(save_dir, f'{name}_model.joblib'))
                    except Exception as e:
                        if self.log_callback:
                            self.log_callback(f"⚠️ {name} kaydedilemedi: {e}", "orange")

                if score > best_score:
                    best_score = score
                    best_model_data = {
                        "name": name,
                        "model": res["model"],
                        "score": score
                    }
                    if task_type == "classification":
                        best_model_data["acc"] = res["accuracy"]
                        best_model_data["f1"] = res["f1"]
                    else:
                        best_model_data["r2"] = res["r2"]
                        best_model_data["mse"] = res["mse"]

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
                    elif vectorizer or (extra_data and 'feature_cols' in extra_data):
                        # Permutation importance (ağır olabilir -> örnekleme)
                        if self.log_callback:
                            self.log_callback(f"🧩 {name}: Permutation Importance hesaplanıyor...", "cyan")
                        sample_idx = np.random.choice(len(y_test), size=min(500, len(y_test)), replace=False)
                        X_sample = X_test[sample_idx]
                        y_sample = y_test[sample_idx]
                        pi = permutation_importance(model, X_sample, y_sample, n_repeats=5, random_state=42)
                        importances = pi.importances_mean
                        if vectorizer:
                            feature_names = np.array(vectorizer.get_feature_names_out())
                        elif extra_data and 'feature_cols' in extra_data:
                            feature_names = np.array(extra_data['feature_cols'])
                        else:
                            feature_names = np.array([f"Feature {i}" for i in range(X_train.shape[1])])
                        if len(feature_names) != len(importances):
                            feature_names = np.array([f"Feature {i}" for i in range(len(importances))])
                        imp_data = (importances, feature_names)
                except Exception as e:
                    if self.log_callback:
                        self.log_callback(f"⚠️ {name} özellik önemleri hesaplanırken hata: {e}", "orange")

                if self.result_callback:
                    self.result_callback(name, res, imp_data, y_test, save_dir)

                # Model kartı (metadata)
                if _opt("save_model_card"):
                    try:
                        metrics = {}
                        for k, v in res.items():
                            if k in ("model", "y_pred"):
                                continue
                            if isinstance(v, (np.floating, np.integer)):
                                metrics[k] = v.item()
                            else:
                                metrics[k] = v

                        model_card = {
                            "name": name,
                            "task_type": task_type,
                            "metrics": metrics,
                            "params": getattr(res["model"], "get_params", lambda: {})(),
                        }
                        with open(os.path.join(save_dir, f"model_card_{name.replace(' ', '_')}.json"), "w", encoding="utf-8") as f:
                            json.dump(model_card, f, ensure_ascii=False, indent=2)
                    except Exception as e:
                        if self.log_callback:
                            self.log_callback(f"⚠️ {name} model kartı kaydedilemedi: {e}", "orange")
            
            except Exception as e:
                if self.error_callback:
                    self.error_callback(name, e)
                elif self.log_callback:
                    self.log_callback(f"❌ {name} Hatası: {e}", "red")
            finally:
                stop_event.set()
                if heartbeat_thread and heartbeat_thread.is_alive():
                    heartbeat_thread.join(timeout=0.1)

        if len(results) > 1 and self.comparison_callback and _opt("save_comparison"):
            self.comparison_callback(results, save_dir)
        
        # Durdurma kontrolü (Optimizasyon öncesi)
        if self.stop_check and self.stop_check():
            if self.log_callback:
                self.log_callback("🛑 Optimizasyon öncesi eğitim durduruldu.", "red")
            if self.completion_callback:
                self.completion_callback(save_dir)
            return

        # En iyi model optimizasyonu (Eğer seçildiyse)
        if do_optimize_best and best_model_data:
            best_name = best_model_data["name"]
            if self.log_callback:
                self.log_callback(f"🚀 En iyi model ({best_name}) için Grid Search başlatılıyor...", "cyan")
            
            try:
                # Yeniden eğit (optimize=True ile)
                # Yeniden eğit (optimize=True ile)
                model = best_model_data["model"]
                res = train_model(
                    model,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    optimize=True,
                    model_name=best_name,
                    task_type=task_type,
                    cv_folds=cv_folds,
                )
                
                # Sonuçları güncelle
                best_model_data["model"] = res["model"]
                if task_type == "classification":
                     best_model_data["acc"] = res["accuracy"]
                     best_model_data["f1"] = res["f1"]
                     new_score_str = f"F1: {res['f1']:.4f}"
                else:
                     best_model_data["r2"] = res["r2"]
                     best_model_data["mse"] = res["mse"]
                     new_score_str = f"R2: {res['r2']:.4f}"
                
                if self.log_callback:
                    self.log_callback(f"✅ {best_name} optimize edildi. Yeni {new_score_str}", "green")
                
                # Tekrar kaydet
                if _opt("save_models"):
                    save_model(res["model"], os.path.join(save_dir, f'{best_name}_model_optimized.joblib'))
                
            except Exception as e:
                if self.log_callback:
                    self.log_callback(f"⚠️ En iyi model optimizasyonu sırasında hata: {e}", "orange")

        if best_model_data:
            try:
                best_name = best_model_data["name"]
                best_filename = f"best_model_{best_name}_{timestamp}.joblib"
                if _opt("save_models"):
                    save_model(best_model_data["model"], os.path.join(save_dir, best_filename))
                if self.best_model_callback:
                    self.best_model_callback(best_model_data, best_filename)

                # ONNX export (opsiyonel)
                if _opt("save_onnx"):
                    try:
                        from skl2onnx import convert_sklearn
                        from skl2onnx.common.data_types import FloatTensorType

                        if hasattr(X_train, "shape"):
                            n_features = X_train.shape[1]
                            initial_type = [("float_input", FloatTensorType([None, n_features]))]
                            onnx_model = convert_sklearn(best_model_data["model"], initial_types=initial_type)
                            onnx_path = os.path.join(save_dir, f"best_model_{best_name}_{timestamp}.onnx")
                            with open(onnx_path, "wb") as f:
                                f.write(onnx_model.SerializeToString())
                            if self.log_callback:
                                self.log_callback(f"📦 ONNX export: {onnx_path}", "green")
                    except Exception as e:
                        if self.log_callback:
                            self.log_callback(f"⚠️ ONNX export atlandı: {e}", "orange")

            except Exception as e:
                if self.log_callback:
                    self.log_callback(f"⚠️ En iyi model ({best_model_data['name']}) kaydedilirken hata: {e}", "orange")

        # Özet raporu üret
        try:
            # JSON uyumlu sonuçlar
            json_results = []
            for row in results:
                clean_row = {}
                for k, v in row.items():
                    if isinstance(v, (np.floating, np.integer)):
                        clean_row[k] = v.item()
                    else:
                        clean_row[k] = v
                json_results.append(clean_row)

            summary = {
                "timestamp": timestamp,
                "task_type": task_type,
                "optimize": optimize,
                "optimize_strategy": optimize_strategy,
                "cv_folds": cv_folds,
                "best_model": best_model_data["name"] if best_model_data else None,
                "results": json_results,
            }
            if _opt("save_summary"):
                with open(os.path.join(save_dir, "training_summary.json"), "w", encoding="utf-8") as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)

            # Markdown rapor
            lines = [
                f"# Training Summary ({timestamp})",
                "",
                f"- Task: {task_type}",
                f"- Optimize: {optimize} ({optimize_strategy})",
                f"- CV Folds: {cv_folds}",
                f"- Best Model: {best_model_data['name'] if best_model_data else 'N/A'}",
                "",
                "## Results",
                "",
            ]
            if results:
                headers = list(results[0].keys())
                lines.append("| " + " | ".join(headers) + " |")
                lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                for row in results:
                    lines.append("| " + " | ".join([str(row.get(h, "")) for h in headers]) + " |")
            if _opt("save_summary"):
                with open(os.path.join(save_dir, "training_summary.md"), "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))

            # Basit HTML rapor
            html_lines = [
                "<html><head><meta charset='utf-8'><title>Training Summary</title></head><body>",
                f"<h1>Training Summary ({timestamp})</h1>",
                f"<p><strong>Task:</strong> {task_type}</p>",
                f"<p><strong>Optimize:</strong> {optimize} ({optimize_strategy})</p>",
                f"<p><strong>CV Folds:</strong> {cv_folds}</p>",
                f"<p><strong>Best Model:</strong> {best_model_data['name'] if best_model_data else 'N/A'}</p>",
                "<h2>Results</h2>",
            ]
            if json_results:
                headers = list(json_results[0].keys())
                html_lines.append("<table border='1' cellpadding='4' cellspacing='0'>")
                html_lines.append("<tr>" + "".join([f"<th>{h}</th>" for h in headers]) + "</tr>")
                for row in json_results:
                    html_lines.append("<tr>" + "".join([f"<td>{row.get(h, '')}</td>" for h in headers]) + "</tr>")
                html_lines.append("</table>")
            html_lines.append("</body></html>")
            if _opt("save_summary"):
                with open(os.path.join(save_dir, "training_summary.html"), "w", encoding="utf-8") as f:
                    f.write("\n".join(html_lines))
        except Exception as e:
            if self.log_callback:
                self.log_callback(f"⚠️ Eğitim özeti kaydedilemedi: {e}", "orange")

        if self.completion_callback:
            self.completion_callback(save_dir)

    def _train_incremental(self, model, name, X_train, y_train, X_test, y_test, lazy_loader, epochs, save_dir):
        """Batch (Incremental) Eğitim Döngüsü"""
        if not hasattr(model, "partial_fit"):
             # Bu model desteklemiyorsa atla veya sabit fit dene (RAM yetmezse patlar)
             if self.log_callback:
                self.log_callback(f"⚠️ {name} incremental training (partial_fit) desteklemiyor. Atlanıyor.", "orange")
             return {"model": model, "accuracy": 0, "f1": 0, "precision": 0, "recall": 0, "y_pred": []}

        classes = np.unique(y_train)
        
        for epoch in range(epochs):
            if self.stop_check and self.stop_check(): break
            
            # Epoch Batch Loop
            batches = lazy_loader.yield_batch(X_train, y_train, batch_size=32)
            for X_batch, y_batch in batches:
                 if self.stop_check and self.stop_check(): break
                 model.partial_fit(X_batch, y_batch, classes=classes)
            
            if self.log_callback:
                 self.log_callback(f"⏳ {name} Epoch {epoch+1}/{epochs} tamamlandı.", "blue")

        # Değerlendirme (Batch Tahmin)
        return self._evaluate_incremental(model, X_test, y_test, lazy_loader)

    def _evaluate_incremental(self, model, X_test, y_test, lazy_loader):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        y_pred = []
        # Test verisini de batch batch tahmin et
        # yield_batch label olmadan da çalışabilmeli ama mevcut func label istiyor. 
        # Hack: y_test'i veriyoruz ama kullanmıyoruz, sadece X için.
        test_batches = lazy_loader.yield_batch(X_test, y_test, batch_size=32, shuffle=False)
        
        for X_batch, _ in test_batches:
            batch_pred = model.predict(X_batch)
            y_pred.extend(batch_pred)
            
        y_pred = np.array(y_pred)
        
        # Boyut uyuşmazlığı kontrolü (yield_batch hata verip atlayabilir)
        # y_test'i y_pred boyutuna kırpmamız gerekebilir (riskli) veya yield_batch'in atladığı indexleri bilmeliyiz.
        # Basitlik için: len(y_pred) kadar y_test alalım (Sıra bozulmadıysa)
        # Ancak yield_batch shuffle=False ile sırayı korur, ama okuma hatası varsa kayar.
        # Şimdilik eşitleyelim:
        if len(y_pred) != len(y_test):
            min_len = min(len(y_pred), len(y_test))
            y_pred = y_pred[:min_len]
            y_test = y_test[:min_len]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        return {
            "model": model,
            "y_pred": y_pred,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1
        }

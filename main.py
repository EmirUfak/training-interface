from ui.main_window import TrainingInterface

if __name__ == "__main__":
    try:
        app = TrainingInterface()
        app.mainloop()
    except Exception as e:
        print(f"HATA: {e}")

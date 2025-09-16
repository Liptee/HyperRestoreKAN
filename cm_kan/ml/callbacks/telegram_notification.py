import subprocess
import os
from lightning.pytorch.callbacks import Callback
from lightning import LightningModule, Trainer
from typing import Dict, Any


class TelegramNotificationCallback(Callback):
    """
    Callback для отправки уведомлений в Telegram с метриками обучения
    после каждой эпохи через скрипт push.sh
    """
    
    def __init__(self, script_path: str = "push.sh", every_n_epochs: int = 1) -> None:
        """
        Args:
            script_path: Путь к скрипту push.sh (может быть относительным или абсолютным)
            every_n_epochs: Отправлять уведомления каждые N эпох
        """
        super().__init__()
        self.every_n_epochs = every_n_epochs
        
        # Если путь не абсолютный, ищем скрипт в разных местах
        if not os.path.isabs(script_path):
            # Сначала проверяем текущую директорию
            if os.path.exists(script_path):
                self.script_path = os.path.abspath(script_path)
            else:
                # Ищем в корне проекта (поднимаемся вверх от текущего файла)
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.join(current_dir, "..", "..", "..")
                project_root = os.path.normpath(project_root)
                potential_path = os.path.join(project_root, script_path)
                
                if os.path.exists(potential_path):
                    self.script_path = os.path.abspath(potential_path)
                else:
                    raise FileNotFoundError(f"Скрипт {script_path} не найден ни в текущей директории, ни в {project_root}")
        else:
            self.script_path = script_path
            
        # Проверяем, что скрипт существует
        if not os.path.exists(self.script_path):
            raise FileNotFoundError(f"Скрипт {self.script_path} не найден")
        
        # Делаем скрипт исполняемым
        os.chmod(self.script_path, 0o755)
        print(f"📱 TelegramNotificationCallback: используем скрипт {self.script_path}")
    
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Отправляем уведомление после каждой эпохи обучения"""
        if trainer.current_epoch % self.every_n_epochs == 0:
            self._send_notification(trainer, pl_module, "train")
    
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Отправляем уведомление после валидации (если нужно)"""
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Получаем метрики из логгера
            metrics = self._get_metrics(trainer)
            if metrics:
                self._send_metrics_notification(trainer, metrics, "validation")
    
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Уведомление о начале обучения"""
        experiment_name = getattr(trainer.logger, 'save_dir', 'Unknown')
        message = f"🚀 Начато обучение модели\\n📁 Эксперимент: {os.path.basename(experiment_name)}\\n📊 Эпох: {trainer.max_epochs}"
        self._send_message(message)
    
    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Уведомление о завершении обучения"""
        experiment_name = getattr(trainer.logger, 'save_dir', 'Unknown')
        message = f"✅ Обучение завершено!\\n📁 Эксперимент: {os.path.basename(experiment_name)}\\n🏁 Финальная эпоха: {trainer.current_epoch}"
        self._send_message(message)
    
    def _get_metrics(self, trainer: Trainer) -> Dict[str, Any]:
        """Получаем метрики из логгера"""
        metrics = {}
        
        # Пытаемся получить метрики из callback_metrics
        if hasattr(trainer, 'callback_metrics') and trainer.callback_metrics:
            for key, value in trainer.callback_metrics.items():
                if hasattr(value, 'item'):
                    metrics[key] = value.item()
                else:
                    metrics[key] = value
        
        # Также пытаемся получить из logged_metrics
        if hasattr(trainer, 'logged_metrics') and trainer.logged_metrics:
            for key, value in trainer.logged_metrics.items():
                if hasattr(value, 'item'):
                    metrics[key] = value.item()
                else:
                    metrics[key] = value
        
        return metrics
    
    def _send_notification(self, trainer: Trainer, pl_module: LightningModule, stage: str):
        """Отправляем уведомление с метриками"""
        metrics = self._get_metrics(trainer)
        self._send_metrics_notification(trainer, metrics, stage)
    
    def _send_metrics_notification(self, trainer: Trainer, metrics: Dict[str, Any], stage: str):
        """Формируем и отправляем сообщение с метриками"""
        experiment_name = getattr(trainer.logger, 'save_dir', 'Unknown')
        
        # Формируем сообщение
        message_lines = [
            f"📈 Эпоха {trainer.current_epoch + 1}/{trainer.max_epochs} ({stage})",
            f"📁 {os.path.basename(experiment_name)}"
        ]
        
        # Добавляем метрики
        if metrics:
            message_lines.append("📊 Метрики:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    message_lines.append(f"  • {key}: {value:.6f}")
                else:
                    message_lines.append(f"  • {key}: {value}")
        else:
            message_lines.append("📊 Метрики недоступны")
        
        # Добавляем прогресс-бар
        progress = (trainer.current_epoch + 1) / trainer.max_epochs
        progress_bar_length = 20
        filled_length = int(progress_bar_length * progress)
        bar = "█" * filled_length + "░" * (progress_bar_length - filled_length)
        percentage = progress * 100
        message_lines.append(f"⏳ Прогресс: {bar} {percentage:.1f}%")
        
        message = "\\n".join(message_lines)
        self._send_message(message)
    
    def _send_message(self, message: str):
        """Отправляем сообщение через скрипт push.sh"""
        try:
            # Экранируем специальные символы для bash
            escaped_message = message.replace('"', '\\"').replace('`', '\\`').replace('$', '\\$')
            
            # Выполняем скрипт
            result = subprocess.run(
                [self.script_path, escaped_message],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.abspath(self.script_path)) or "."
            )
            
            if result.returncode != 0:
                print(f"Ошибка отправки уведомления в Telegram: {result.stderr}")
            else:
                print(f"✅ Уведомление отправлено в Telegram")
                
        except Exception as e:
            print(f"Ошибка при отправке уведомления в Telegram: {e}")

import keras

callbacks = [
    keras.callbacks.BackupAndRestore(backup_dir="./backups"),
    keras.callbacks.TensorBoard(log_dir="./logs"),
    keras.callbacks.ProgbarLogger()
]

import keras

callbacks = [
    keras.callbacks.BackupAndRestore(backup_dir="./backup"),
    keras.callbacks.ModelCheckpoint(filepath="./ckpt"),
    keras.callbacks.TensorBoard(log_dir="./logs"),
    # keras.callbacks.ProgbarLogger()
]

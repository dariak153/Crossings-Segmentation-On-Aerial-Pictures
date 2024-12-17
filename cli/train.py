from dataloder import crossingsDataModule
from lightningmodule import segmentationModule
import lightning.pytorch as pl
import torch

if __name__ == '__main__':
    # Check if CUDA is available
    print(torch.cuda.is_available())

    # Instantiate the data module and model
    data_module = crossingsDataModule.SegmentationDataModule()
    lightning_model = segmentationModule.MySegmentationModel()

    # Define callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        mode='min',  # Use 'min' to minimize the validation loss
        verbose=True,
        save_top_k=3,  # Save only the best model
        filename='../../../../saved_models/best_model'  # Name of the saved model file <- that's stupid
    )
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=25,  # Number of epochs with no improvement after which training will be stopped
        verbose=True
    )

    # Set up the trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        callbacks=[checkpoint_callback, early_stopping_callback],
        max_epochs=1000
    )

    # Train the model
    trainer.fit(lightning_model, datamodule=data_module)

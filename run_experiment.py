import os
import argparse
from HistoLib import generator
from HistoLib import utils
from HistoLib import models
from HistoLib import traintest
from HistoLib import gradcam
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='Run LungHist700 Experiment')
    parser.add_argument('--resolution', type=str, default='20x', choices=['20x', '40x'], help='Image resolution')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()

    print(f"Running experiment with resolution={args.resolution}, batch_size={args.batch_size}, epochs={args.epochs}, debug={args.debug}")

    # Dataset stats
    print("Dataset Description:")
    utils.dataset_description()

    # Data Generators
    print("Setting up data generators...")
    train_generator, val_generator, test_generator, class_names = generator.get_patient_generators(
        args.resolution, 
        batch_size=args.batch_size, 
        debug=args.debug,
        reproducible=True
    )

    # Model
    print("Building model...")
    model, model_name = models.get_model(train_generator)
    print(f"Model: {model_name}")

    # Compile
    print("Compiling model...")
    model = traintest.compile_model(model, num_classes=train_generator.num_classes)

    # Train
    print("Training model...")
    # Calculate class weights for balanced training
    class_weights = utils.compute_weights(train_generator)
    
    log_dir = traintest.get_logdir(model_name)
    history = traintest.train_model(
        model, 
        train_generator, 
        val_generator, 
        class_weights, 
        log_dir,
        num_epochs=args.epochs,
        patience=5, # Reduced for quick experimentation
        patience_lr=2
    )

    # Evaluate
    print("Evaluating model...")
    # Create a figure for metrics and confusion matrix
    # Note: metrics_and_test plots directly, we might need to adjust if running headless
    try:
        traintest.metrics_and_test(history, model, test_generator, class_names)
        plt.savefig('evaluation_metrics.png')
        print("Evaluation plot saved to evaluation_metrics.png")
    except Exception as e:
        print(f"Error during evaluation/plotting: {e}")

    # Grad-CAM
    print("Generating Grad-CAM samples...")
    try:
        gradcam.generate_gradcam_samples(model, test_generator)
        plt.savefig('gradcam_samples.png')
        print("Grad-CAM plot saved to gradcam_samples.png")
    except Exception as e:
        print(f"Error during Grad-CAM generation: {e}")

if __name__ == "__main__":
    main()

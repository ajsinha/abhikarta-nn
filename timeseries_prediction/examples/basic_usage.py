"""
Example Usage of Time Series Prediction Package

Copyright © 2025-2030, All Rights Reserved
Ashutosh Sinha | Email: ajsinha@gmail.com
"""

import numpy as np
import torch
from timeseries_prediction.factory import ModelFactory
from timeseries_prediction.utils import (
    generate_synthetic_time_series,
    prepare_time_series_data,
    evaluate_model,
    plot_predictions,
    plot_training_history
)


def example_basic_lstm():
    """Basic LSTM model example."""
    print("\n" + "="*80)
    print("Example 1: Basic LSTM Model")
    print("="*80)
    
    # Generate synthetic data
    data = generate_synthetic_time_series(
        n_samples=1000,
        n_features=5,
        trend=True,
        seasonality=True,
        noise_level=0.1,
        random_seed=42
    )
    
    # Prepare data
    train_loader, val_loader, test_loader, scaler = prepare_time_series_data(
        data=data,
        seq_length=30,
        prediction_horizon=1,
        scaler_method='standard',
        batch_size=32
    )
    
    # Create model using factory
    model = ModelFactory.create_model(
        model_type='lstm',
        input_size=5,
        hidden_size=64,
        output_size=5,
        num_layers=2,
        dropout=0.2
    )
    
    print(model.summary())
    
    # Train model
    history = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,
        lr=0.001,
        verbose=True
    )
    
    # Evaluate
    test_predictions = []
    test_targets = []
    
    model.eval()
    for batch_x, batch_y in test_loader:
        pred = model.predict(batch_x, return_numpy=True)
        test_predictions.append(pred)
        test_targets.append(batch_y.numpy())
    
    test_predictions = np.vstack(test_predictions)
    test_targets = np.vstack(test_targets)
    
    # Evaluate metrics
    metrics = evaluate_model(test_targets, test_predictions, verbose=True)
    
    # Plot results
    plot_training_history(history['train_losses'], history['val_losses'])
    plot_predictions(test_targets[:, 0], test_predictions[:, 0])


def example_transformer():
    """Transformer model example."""
    print("\n" + "="*80)
    print("Example 2: Transformer Model")
    print("="*80)
    
    # Generate data
    data = generate_synthetic_time_series(
        n_samples=1000,
        n_features=10,
        random_seed=42
    )
    
    # Prepare data
    train_loader, val_loader, test_loader, scaler = prepare_time_series_data(
        data=data,
        seq_length=50,
        prediction_horizon=1,
        batch_size=64
    )
    
    # Create Transformer model
    model = ModelFactory.create_model(
        model_type='transformer',
        input_size=10,
        hidden_size=128,
        output_size=10,
        num_layers=3,
        num_heads=4,
        dim_feedforward=512,
        dropout=0.1
    )
    
    print(model.summary())
    
    # Train
    history = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=30,
        lr=0.0001,
        verbose=True
    )


def example_ensemble():
    """Ensemble model example."""
    print("\n" + "="*80)
    print("Example 3: Ensemble Model")
    print("="*80)
    
    # Generate data
    data = generate_synthetic_time_series(n_samples=800, n_features=5)
    
    # Prepare data
    train_loader, val_loader, test_loader, scaler = prepare_time_series_data(
        data=data,
        seq_length=30,
        prediction_horizon=1,
        batch_size=32
    )
    
    # Create ensemble from configs
    model_configs = [
        {'type': 'lstm', 'input_size': 5, 'hidden_size': 64, 'output_size': 5},
        {'type': 'gru', 'input_size': 5, 'hidden_size': 64, 'output_size': 5},
        {'type': 'cnn', 'input_size': 5, 'hidden_size': 64, 'output_size': 5}
    ]
    
    ensemble = ModelFactory.create_ensemble_from_configs(
        model_configs,
        ensemble_method='weighted'
    )
    
    # Train ensemble
    history = ensemble.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,
        lr=0.001,
        train_base_models=True,
        verbose=True
    )


def example_model_builder():
    """Model builder pattern example."""
    print("\n" + "="*80)
    print("Example 4: Model Builder Pattern")
    print("="*80)
    
    from timeseries_prediction.factory import ModelBuilder
    
    # Build model using builder pattern
    builder = ModelBuilder()
    
    model = (builder
             .set_type('bilstm')
             .set_input_size(10)
             .set_hidden_size(128)
             .set_output_size(1)
             .set_num_layers(3)
             .set_dropout(0.3)
             .set_device('cpu')
             .build())
    
    print(model.summary())


def example_probabilistic_forecasting():
    """DeepAR probabilistic forecasting example."""
    print("\n" + "="*80)
    print("Example 5: Probabilistic Forecasting with DeepAR")
    print("="*80)
    
    # Generate data
    data = generate_synthetic_time_series(n_samples=1000, n_features=3)
    
    # Prepare data
    train_loader, val_loader, test_loader, scaler = prepare_time_series_data(
        data=data,
        seq_length=30,
        prediction_horizon=1,
        batch_size=32
    )
    
    # Create DeepAR model
    model = ModelFactory.create_model(
        model_type='deepar',
        input_size=3,
        hidden_size=64,
        output_size=3,
        num_layers=2,
        distribution='gaussian'
    )
    
    print(model.summary())
    
    # Train
    history = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=30,
        lr=0.001,
        verbose=True
    )
    
    # Generate probabilistic forecasts
    for batch_x, batch_y in test_loader:
        # Get mean and std
        mu, sigma = model.forward(batch_x)
        
        # Generate samples
        samples = model.sample(batch_x, num_samples=100)
        
        print(f"Mean prediction shape: {mu.shape}")
        print(f"Std prediction shape: {sigma.shape}")
        print(f"Samples shape: {samples.shape}")
        break


def example_save_load():
    """Example of saving and loading models."""
    print("\n" + "="*80)
    print("Example 6: Saving and Loading Models")
    print("="*80)
    
    # Create model
    model = ModelFactory.create_model(
        model_type='lstm',
        input_size=5,
        hidden_size=64,
        output_size=1,
        num_layers=2
    )
    
    # Save model
    model.save('saved_models/lstm_model.pth')
    print("Model saved!")
    
    # Create new model and load
    new_model = ModelFactory.create_model(
        model_type='lstm',
        input_size=5,
        hidden_size=64,
        output_size=1,
        num_layers=2
    )
    
    new_model.load('saved_models/lstm_model.pth')
    print("Model loaded!")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("Time Series Prediction Package - Examples")
    print("="*80)
    
    # Run examples
    print("\nAvailable models:")
    print(ModelFactory.list_models())
    
    # Uncomment to run specific examples
    example_basic_lstm()
    # example_transformer()
    # example_ensemble()
    # example_model_builder()
    # example_probabilistic_forecasting()
    # example_save_load()
    
    print("\n" + "="*80)
    print("Examples completed!")
    print("="*80)

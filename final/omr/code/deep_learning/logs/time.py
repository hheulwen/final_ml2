import tensorflow as tf
import time

# Start TensorBoard writer
train_summary_writer = tf.summary.create_file_writer('./logs/train')

# Training loop
with train_summary_writer.as_default():
    for epoch in range(10):
        start_time = time.time()
        # Train the model for one epoch
        # ...
        end_time = time.time()
        epoch_time = end_time - start_time
        
        # Log epoch time to TensorBoard
        tf.summary.scalar('epoch_time', epoch_time, step=epoch)
        
# Start TensorBoard writer for inference
inference_summary_writer = tf.summary.create_file_writer('./logs/inference')

# Inference loop
with inference_summary_writer.as_default():
    start_time = time.time()
    # Perform inference on a sample of images
    # ...
    end_time = time.time()
    total_inference_time = end_time - start_time
    
    # Log inference time to TensorBoard
    tf.summary.scalar('total_inference_time', total_inference_time, step=0)

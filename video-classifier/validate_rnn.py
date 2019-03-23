"""
Validate our RNN. Basically just runs a validation generator on
about the same number of videos as we have in our test set.
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from models import ResearchModels
from data2 import DataSet
import numpy as np

def validate(data_type, model, seq_length=80, saved_model=None,
             class_limit=None, image_shape=None):
    batch_size = 1

    # Get the data and process it.
    if image_shape is None:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit
        )
    else:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit,
            image_shape=image_shape
        )

    val_generator = data.frame_generator(batch_size, 'test', data_type)

    # Get the model.
    rm = ResearchModels(len(data.classes), model, seq_length, saved_model)

    # Evaluate!
    results = rm.model.evaluate_generator(
        generator=val_generator,
        steps = 4)

    print(results)
    print(rm.model.metrics_names)

    x, y = data.get_all_sequences_in_memory('test', data_type)
    print("CLASS:", y)

    pred_result = rm.model.predict_classes(x, batch_size=1)
    print("PREDICTED RESULT 1:", pred_result)

    # Predict!
    prediction = rm.model.predict(np.expand_dims(x, axis=0))
    print("PREDICTED RESULT 2:", prediction)
    data.print_class_from_prediction(np.squeeze(prediction, axis=0))


def main():
    model = 'lstm'
    saved_model = 'data/checkpoints/lstm-features.041-0.567.hdf5'
    # model = 'lrcn'
    # saved_model = 'data/checkpoints/lrcn-images.059-0.479.hdf5'

    if model == 'conv_3d' or model == 'lrcn':
        data_type = 'images'
        image_shape = (200, 200, 3)
    else:
        data_type = 'features'
        image_shape = None

    validate(data_type, model, saved_model=saved_model,
             image_shape=image_shape, class_limit=None)

if __name__ == '__main__':
    main()

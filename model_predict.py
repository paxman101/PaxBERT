import tensorflow as tf
import transformers as ppb
import argparse
import numpy as np
from nlp_horovod import get_full_model, get_model_config, get_test_dataset


def model_predict_to_file(prediction_model,
                          predict_dataset: tf.data.Dataset,
                          output_file,
                          batch_size=16,
                          regression=False,
                          verbosity=1):
    num_test_steps = tf.data.experimental.cardinality(predict_dataset).numpy() // batch_size
    predictions = prediction_model.predict(predict_dataset.batch(batch_size), steps=num_test_steps, verbose=verbosity)
    predicted_class = np.squeeze(predictions) if regression else np.argmax(predictions, axis=1)

    with open(output_file, "w") as writer:
        writer.write(str(predicted_class.tolist()))
        for ele in predict_dataset.enumerate().as_numpy_iterator():
            writer.write(str(ele))


def main():
    parser = argparse.ArgumentParser(description="Run predictions with dataset with given saved model.")
    parser.add_argument("model_dir", type=str)
    parser.add_argument("-o" "--output-file", dest="output_file")
    parser.add_argument("-b", "--batch-size", dest="batch_size", default=16)
    parser.add_argument("-r", "--is-regression", dest="is_regression", default=False)
    parser.add_argument("-s", "--num-samples", dest="num_samples", default=None)

    args = parser.parse_args()
    output_file = args.output_file
    model_dir = args.model_dir
    batch_size = args.batch_size
    is_regression = args.is_regression
    num_samples = args.num_samples

    config = get_model_config(ppb.DistilBertConfig, regression=is_regression)
    model = ppb.TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased",
                                                                      config=config)
    full_model = get_full_model(model, regression=is_regression)
    full_model.load_weights(model_dir)

    test_dataset = get_test_dataset(8, lim=num_samples)
    model_predict_to_file(full_model, test_dataset, output_file, batch_size=batch_size, regression=is_regression)


if __name__ == "__main__":
    main()

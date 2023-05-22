import sys
import torch
# import wandb
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer, Trainer, TrainingArguments, \
    set_seed
from evaluate import load
import numpy as np

MODELS = ['bert-base-uncased', 'roberta-base', 'google/electra-base-generator']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PREDICTIONS_FILE = 'predictions.txt'
RESULTS_FILE = 'res.txt'
# WANDB_PROJECT = 'anlp_ex1'
TRAINING_OUTPUT = "training"


class ModelManager:
    def __init__(self, model_name, seeds_num, training_samples, validation_samples, prediction_samples):
        self.model_name = model_name
        self._config = AutoConfig.from_pretrained(model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._training_args = TrainingArguments(output_dir=TRAINING_OUTPUT,
                                                save_strategy='no')
        self._raw_datasets = load_dataset("sst2").map(self.preprocess_function, batched=True)
        self.train_dataset = self._raw_datasets["train"]
        self._eval_dataset = self._raw_datasets["validation"]
        self._test_dataset = self._raw_datasets["test"].remove_columns("label")
        if training_samples > 0:
            self.train_dataset = self.train_dataset.select(range(training_samples))
        if validation_samples > 0:
            self._eval_dataset = self._eval_dataset.select(range(validation_samples))
        if prediction_samples > 0:
            self._test_dataset = self._test_dataset.select(range(prediction_samples))
        self._seeds_num = seeds_num
        self._trainer = None
        self.mean_acc = None
        self.std_acc = None
        self.train_time = 0

    def train(self):
        metrics = np.zeros(self._seeds_num)
        for i in range(self._seeds_num):
            # wandb.init(project=WANDB_PROJECT, name=f'{self.model_name}_{i}')
            set_seed(i)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=self._config).to(DEVICE)
            trainer = Trainer(
                model=model,
                args=self._training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self._eval_dataset,
                compute_metrics=compute_metrics,
                tokenizer=self._tokenizer,
            )
            self.train_time += trainer.train().metrics.get('train_runtime')
            model_acc = trainer.evaluate(eval_dataset=self._eval_dataset)['eval_accuracy']
            if model_acc > max(metrics):  # save trainer with best accuracy
                self._trainer = trainer
            metrics[i] = model_acc
        self.mean_acc = metrics.mean()
        self.std_acc = metrics.std()

    def predict(self, filepath=PREDICTIONS_FILE):
        self._trainer.args.set_testing(batch_size=1)
        self._trainer.model.eval()
        predict_output = self._trainer.predict(self._test_dataset)
        predictions = np.argmax(predict_output.predictions, axis=-1)
        with open(filepath, "w") as f:
            for i in range(len(self._test_dataset)):
                f.write(f"{self._test_dataset[i]['sentence']}###{predictions[i]}\n")
        return predict_output.metrics.get('test_runtime')

    def preprocess_function(self, examples):
        result = self._tokenizer(examples["sentence"], truncation=True)
        return result


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metric = load("accuracy")
    return metric.compute(predictions=predictions, references=labels)


def write_results(models, prediction_time, filepath=RESULTS_FILE):
    with open(filepath, "w") as f:
        for model_name in models:
            f.write(f"{models[model_name].model_name},{models[model_name].mean_acc} +- {models[model_name].std_acc}\n")
        f.write("----\n")
        f.write(f"train time,{sum([models[model_name].train_time for model_name in models])}\n")
        f.write(f"predict time,{prediction_time}")


def main(seeds_num, training_samples, validation_samples, prediction_samples):
    # wandb.login()
    models = dict()
    best_model = None
    for model_name in MODELS:
        model = ModelManager(model_name, seeds_num, training_samples, validation_samples, prediction_samples)
        model.train()
        models[model_name] = model
        if best_model is None or model.mean_acc > best_model.mean_acc:  # save model with best mean accuracy
            best_model = model

    prediction_time = best_model.predict()
    write_results(models, prediction_time)
    # wandb.finish()


if __name__ == '__main__':
    seeds_num, training_samples, validation_samples, prediction_samples = (int(arg) for arg in sys.argv[1:])
    main(seeds_num, training_samples, validation_samples, prediction_samples)

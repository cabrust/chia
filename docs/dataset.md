# Using your own dataset

While implementing your own subclass of [`Dataset`] is one possible way of using your own data, CHIA also supports a simple JSON format. A corresponding configuration entry looks like this:
```json
{
    "dataset": {
        "name": "json",
        "base_path": "/base/path/for/data",
        "json_path": "/path/to/dataset/json",
        "side_length": 512,
        "use_lazy_mode": false
    }
}
```

This will create an instance of [`JSONDataset`] with the specified parameters. You mainly need to provide a JSON file in the format corresponding to the following example:
```json
{
    "file_format": "JSONDataset.v2",
    "namespace": "Ex",
    "prediction_targets": ["Ex::Car", "Ex::Bus", "Ex::Cat", "Ex::Dog", ...],
    "hyponymy_relation": [["Ex::Car", "Ex::Vehicle"], ["Ex::Cat", "Ex::Animal"], ...],
    "train_pools": [
        [
            {
                "image_location": "/path/to/image/relative/to/base/path",
                "label_gt": "Ex::Car",
                "uid": "Ex::ExampleDataset.train.pool0.001"
            }
        ]
    ],
    "val_pools": [...],
    "test_pools": [...]
}
```
**Namespaces** are explained [here](architecture.md#dataset), but are mainly a way of keeping track when combining multiple datasets or knowledge bases.

**Prediction targets** are the classes/concept that should be considered by the classifier when making a prediction. Typically, these are (but don't have to be) the leaf nodes of the hierarchy.

The **hyponymy relation** relates the classes/concepts in the dataset. Any concept specified here will be added to the knowledge base automatically, there is no need to give a complete list of concepts separately. You can also refer to concepts in the namespace `WordNet3.0`, e.g., `WordNet3.0::apple.n.01`, which will be extracted from WordNet automatically.

There can be one or more **train pools**. If you want to use CHIA without code changes, we recommend exactly one pool. However, for incremental scenarios, more than one training pool may be appropriate. The same is true for **test pools** and **val pools**.

## Examples
The examples' **uid** is completely up to you, but it should be unique per example and be prefixed with the dataset's namespace.

**Image location** is always relative to the base_path supplied in the configuration of [`JSONDataset`]. This decouples the dataset description json from the local folder structure which might vary.


[`Dataset`]: /chia/components/datasets/dataset.py
[`JSONDataset`]: /chia/components/datasets/json_dataset.py
## Data format example
This implementation requires the input data in the following format:
### Training/Validation data
1. `txt_file` txt file: multi-line with format: `ImageFileName,TextLabel`. e.g. `data/example_gt.txt`
2. `img_root` folder: image files corresponding to `ImageFileName`. e.g. `data/example_imgs`

### Testing data
1. `img_folder` folder: image files.

Or use lmdb format dataset and modified the corresponding config file.

### Lmdb data
Base on lmdb dataset in `deep text recognition benchmark`: [Source repository](https://github.com/clovaai/deep-text-recognition-benchmark)

Download lmdb dataset: [Here](https://www.dropbox.com/sh/i39abvnefllx2si/AABX4yjNn2iLeKZh1OAwJUffa/data_lmdb_release.zip?dl=0)
    
Unzip and modify folder below and use config file `config_lmdb.json` to train:
(Can custom sub folder in folder `training` and `val` by fix `select_data` in config file)

    dataset/data_lmdb_release/
        training/
            MJ_train
            ST
        val/
            MJ_test
            MJ_valid

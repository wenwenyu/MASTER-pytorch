## Data format example
This implementation requires the input data in the following format:
### Training/Validation data
1. `txt_file` txt file: multi-line with format: `ImageFileName,TextLabel`. e.g. `data/example_gt.txt`
2. `img_root` folder: image files corresponding to `ImageFileName`. e.g. `data/example_imgs`

### Testing data
1. `img_folder` folder: image files.

Or use lmdb format dataset and modified the corresponding config file.
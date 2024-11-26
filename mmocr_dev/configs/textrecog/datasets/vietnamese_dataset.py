vietnamese_textrecog_data_root = '../../../../data/vietnamese_dataset/'

vietnamese_textrecog_train = dict(
    type='OCRDataset',
    data_root=vietnamese_textrecog_data_root,
    ann_file='train_labels.json',
    pipeline=None)

vietnamese_textrecog_test = dict(
    type='OCRDataset',
    data_root=vietnamese_textrecog_data_root,
    ann_file='test_labels.json',
    test_mode=True,
    pipeline=None)
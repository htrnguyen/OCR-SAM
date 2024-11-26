_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/vietnamese_dataset.py',
    '../_base_/schedules/schedule_adam.py',
]

model = dict(
    type='ABINet',
    # Cấu hình mô hình
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=120, val_interval=10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, betas=(0.9, 0.999), weight_decay=0.05))
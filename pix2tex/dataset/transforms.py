import albumentations as alb
from albumentations.pytorch import ToTensorV2

train_transform = alb.Compose(
    [
        alb.Compose(
            [
                alb.ShiftScaleRotate(shift_limit=0, scale_limit=(-.15, 0), rotate_limit=1, interpolation=3, p=1),  # Removed 'border_mode' and 'value'
                alb.GridDistortion(distort_limit=0.1, interpolation=3, p=.5),  # Removed 'border_mode' and 'value'
            ], p=.15),
        # alb.InvertImg(p=.15),
        alb.RGBShift(r_shift_limit=15, g_shift_limit=15,
                     b_shift_limit=15, p=0.3),
        alb.GaussNoise(std=(10, 20), p=.2),
        alb.RandomBrightnessContrast(.05, (-.2, 0), True, p=0.2),
        alb.ImageCompression(quality=95, compression_type='jpeg', p=.3),  # Corrected to use 'compression_type' and 'quality'
        alb.ToGray(always_apply=True),
        alb.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
        # alb.Sharpen()
        ToTensorV2(),
    ]
)

test_transform = alb.Compose(
    [
        alb.ToGray(always_apply=True),
        alb.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
        # alb.Sharpen()
        ToTensorV2(),
    ]
)

# ImageNet-100 Origin Training

`train_origin.py` trains the origin MoE-ViT model used as the starting point
for later class-wise unlearning experiments.

The script expects the local dataset in this layout:

```text
/root/autodl-tmp/SEUF_MCU/data_cv/imagenet100/
├── n01558993/
│   ├── *.JPEG
│   └── ...
└── ...
```

Images are split per class into a 90% training split and a 10% validation
split by default. Ten classes are selected as the class-wise forget group and
the remaining ninety classes are the retain group. The images are not copied.
The split and the two class groups are saved to
`model/imagenet100_classwise_split.json`.

Run from the repository root:

```bash
python CV/imagenet100/train_origin.py \
  --data-root /root/autodl-tmp/SEUF_MCU/data_cv/imagenet100 \
  --output-dir /root/autodl-tmp/SEUF_MCU/CV/imagenet100/model \
  --image-size 224 \
  --patch-size 16 \
  --embed-dim 256 \
  --depth 12 \
  --num-heads 8 \
  --batch-size 32 \
  --epochs 200 \
  --warmup-epochs 10 \
  --lr 8e-4 \
  --mixup-alpha 0.2 \
  --num-workers 8
```

The best checkpoint is written to:

```text
CV/imagenet100/model/moevit_imagenet100_origin_best.pth
```

The checkpoint contains the model, optimizer and scheduler states, validation
accuracy, forget/retain train and test accuracies, model arguments, and the
deterministic class mapping.

Each epoch prints:

```text
forget train
retain train
forget test
retain test
```

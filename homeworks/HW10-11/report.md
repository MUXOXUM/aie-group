# HW10-11 – компьютерное зрение в PyTorch: CNN, transfer learning, detection/segmentation

## 1. Кратко: что сделано

- Часть A: выбран датасет CIFAR100 для классификации (100 классов, стандартный torchvision доступ).
- Часть B: выбран OxfordIIITPet, трек segmentation (понятный foreground: питомец vs фон).
- Часть A: сравнение C1–C4 (CNN без/с аугментациями, ResNet18 head-only, ResNet18 partial fine-tuning).
- Часть B: сравнение двух режимов постобработки (V1/V2) через различный порог foreground.

## 2. Среда и воспроизводимость

- Python: TBD
- torch / torchvision: TBD
- Устройство (CPU/GPU): TBD
- Seed: 42
- Как запустить: открыть `HW10-11.ipynb` и выполнить Run All.

## 3. Данные

### 3.1. Часть A: классификация

- Датасет: CIFAR100
- Разделение: train/val/test (val = 20% от train, фиксированный seed)
- Базовые transforms: `ToTensor + Normalize(CIFAR100 mean/std)`
- Augmentation transforms: `RandomCrop + RandomHorizontalFlip + ColorJitter`
- Комментарий: CIFAR100 — 100 классов, 32x32 изображения. Задача сложнее CIFAR10 из-за количества классов и ограниченного размера изображений.

### 3.2. Часть B: structured vision

- Датасет: OxfordIIITPet
- Трек: segmentation
- Что считается ground truth: бинарная маска питомца (trimap value == 1)
- Какие предсказания использовались: foreground маска как `1 - p(background)` от DeepLabV3
- Комментарий: постановка «питомец vs фон» соответствует имеющейся разметке, позволяет оценить базовую сегментацию без переобучения модели.

## 4. Часть A: модели и обучение (C1-C4)

- C1 (simple-cnn-base): SimpleCNN без аугментаций.
- C2 (simple-cnn-aug): SimpleCNN с аугментациями.
- C3 (resnet18-head-only): ResNet18 pretrained, обучается только классификационная голова.
- C4 (resnet18-finetune): ResNet18 pretrained, fine-tune `layer4 + fc`.

Дополнительно:

- Loss: CrossEntropyLoss
- Optimizer(ы): Adam (CNN), SGD (ResNet)
- Batch size: 128
- Epochs (макс): 5 (по умолчанию, можно увеличить)
- Критерий выбора лучшей модели: `best_val_accuracy`

## 5. Часть B: постановка задачи и режимы оценки (V1-V2)

### Если выбран segmentation track

- Модель: DeepLabV3_ResNet50 (pretrained)
- Что считается foreground: пиксели питомца (mask == 1)
- V1: базовая постобработка (threshold = 0.5 по `1 - p(background)`)
- V2: альтернативная постобработка (threshold = 0.7)
- Как считался mean IoU: по бинарным маскам, усреднение по изображениям
- Считались ли дополнительные pixel-level метрики: pixel precision/recall

## 6. Результаты

Ссылки на файлы в репозитории:

- Таблица результатов: `./artifacts/runs.csv`
- Лучшая модель части A: `./artifacts/best_classifier.pt`
- Конфиг лучшей модели части A: `./artifacts/best_classifier_config.json`
- Кривые лучшего прогона классификации: `./artifacts/figures/classification_curves_best.png`
- Сравнение C1-C4: `./artifacts/figures/classification_compare.png`
- Визуализация аугментаций: `./artifacts/figures/augmentations_preview.png`
- Визуализации второй части: `./artifacts/figures/segmentation_examples.png`, `./artifacts/figures/segmentation_metrics.png`

Короткая сводка (6-10 строк):

- Лучший эксперимент части A: TBD
- Лучшая `val_accuracy`: TBD
- Итоговая `test_accuracy` лучшего классификатора: TBD
- Что дали аугментации (C2 vs C1): TBD
- Что дал transfer learning (C3/C4 vs C1/C2): TBD
- Что оказалось лучше: head-only или partial fine-tuning: TBD
- Что показал режим V1 во второй части: TBD
- Что показал режим V2 во второй части: TBD
- Как интерпретируются метрики второй части: TBD

## 7. Анализ

TBD. Заполнить после запуска экспериментов и анализа результатов.

## 8. Итоговый вывод

TBD. Заполнить после запуска экспериментов.

## 9. Приложение (опционально)

Если будут дополнительные сравнения — добавить ссылки на соответствующие артефакты.

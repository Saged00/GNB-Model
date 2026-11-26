# Gaussian Naive Bayes — From Scratch

تطبيق يدوي لخوارزمية **Gaussian Naive Bayes** بدون استخدام sklearn للموديل نفسه، مع مقارنة أداء الموديل على بيانات خام ومعيارية.

---

## الفكرة

Naive Bayes هي خوارزمية تصنيف مبنية على **نظرية بايز**:

$$P(C \mid x) \propto P(C) \cdot \prod_{i} P(x_i \mid C)$$

النسخة **Gaussian** تفترض إن كل feature تتبع توزيعًا طبيعيًا (Gaussian) مستقلًا داخل كل class. بناءً على هذا الافتراض، يحسب الموديل:

- **المتوسط والتباين** لكل feature لكل class أثناء التدريب
- **الاحتمال اللوغاريتمي** لكل class لتجنب مشكلة underflow عند الضرب
- **Prior probability** لكل class من تكرارها في بيانات التدريب

---

## البيانات

**Breast Cancer Wisconsin Dataset** من sklearn:
- 569 عينة
- 30 feature (قياسات خلايا الورم)
- Binary classification: Malignant (0) / Benign (1)

---

## ما تم تنفيذه يدويًا

| المكوّن | التفاصيل |
|--------|----------|
| `train_test_split` | Stratified split — 80% train / 20% test مع الحفاظ على نسبة الـ classes |
| Feature Standardization | Z-score normalization محسوب من training set فقط |
| `GaussianNaiveBayes.fit` | حساب المتوسطات والتباينات والـ priors |
| `GaussianNaiveBayes.predict_proba` | Log-space Gaussian likelihood + softmax |
| `GaussianNaiveBayes.predict` | argmax على الاحتمالات |

---

## النتائج

| الإعداد | Accuracy |
|---------|----------|
| بدون Standardization | **93.81%** |
| مع Standardization | 92.92% |

> الموديل أدى أداءً أفضل على البيانات الخام في هذه الحالة — وهذا متوقع لأن GNB يحسب التباين لكل feature بشكل مستقل، مما يجعله أقل حساسية للـ scale مقارنةً بخوارزميات أخرى.

---

## الملفات

```
GNB-Model/
└── Model.ipynb    # الكود الكامل مع شرح كل خطوة
```

---

## التشغيل

```bash
jupyter notebook Model.ipynb
```

**المتطلبات:** `numpy`, `jupyter`

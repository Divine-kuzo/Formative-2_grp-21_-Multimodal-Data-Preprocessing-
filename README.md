# Group 21: Multimodal Data Preprocessing

A command-line system that only gives a product recommendation to a verified user. The user must pass two checks first:

1. **Face recognition**: the face image must belong to a known team member.
2. **Voiceprint verification**: the voice sample must belong to the **same** member.

If either check fails, access is denied and the product model is never called.

```
face image ──> Face model ──(known member?)──> Voice model ──(same member?)──> Product model ──> recommendation
                    │ no                             │ no
                    └──> ACCESS DENIED               └──> ACCESS DENIED
```

---

## 1. Team contributions

| Member | Name | Responsibility |
|---|---|---|
| member_01 | _(add name)_ | Customer data cleaning, merge and EDA (`notebooks/customer_data_preprocessing.ipynb`) |
| member_02 | _(add name)_ | Image collection, augmentation, features and face model (`notebooks/Image_processing.ipynb`, `src/image_utils.py`) |
| member_03 | Constantine | Audio collection, augmentation, features and voice model (`notebooks/Group_21_Audio_Preprocessing_Voice_Verification.ipynb`, `src/audio_utils.py`) |
| member_04 | _(add name)_ | Product recommendation model and CLI integration (`src/product_model.py`, `src/voice_fallback.py`, `app.py`) |

The full contribution sheet is linked in the report.

---

## 2. Quick start

```bash
pip install -r requirements.txt

python app.py --demo                                   # runs all 4 scenarios (1 approved, 3 denied)
python app.py                                          # interactive: asks for a face image and a .wav file
python app.py --face images/member_01/member_01_smile.jpg --voice audio/member_01/yes_approve.wav
```

Every attempt is appended to `outputs/simulation_log.csv`.

To retrain everything from the raw files:

```bash
python src/image_utils.py train          # data/image_features.csv + models/face_model.pkl
python -m src.build_audio_dataset        # data/audio_features.csv
python src/voice_fallback.py --from-audio  # models/voice_fallback.pkl (voice model used by app.py)
python src/product_model.py              # models/product_model.pkl + outputs/product_model/
```

---

## 3. Repository structure

```
app.py                      Command-line app (the full transaction flow)
src/
  image_utils.py            Face detection, augmentation, features, face model, predict_face()
  audio_utils.py            Audio feature extraction (13 MFCCs, roll-off, RMS energy)
  build_audio_dataset.py    Builds data/audio_features.csv (originals + 3 augmentations)
  train_voice_model.py      Member 3's SVM voice model (experiment)
  voice_fallback.py         Voice model used by app.py (Random Forest + stranger gate)
  product_model.py          Product recommendation model, evaluation and predict_product()
notebooks/                  One notebook per modality (data, image, audio)
data/
  merged_customer_data.csv  All 150 transactions after the merge
  product_model_data.csv    117 matched transactions + engineered features (model input)
  image_features.csv        91 rows: 13 images x (original + 6 augmentations), 382 features each
  audio_features.csv        32 rows: 8 recordings x (original + 3 augmentations), 15 features each
  voice_training_features.csv  48 rows used to train the app's voice model
images/member_XX/           3 photos per member: neutral, smile, surprised
images/unknown/             Stranger photo for the unauthorized attempt
audio/member_XX/            2 recordings per member: "Yes, approve" and "Confirm transaction"
samples/unknown_voice.wav   Stranger voice for the unauthorized attempt
models/                     Trained models (.pkl)
outputs/                    Figures, metrics and the simulation log
archive/                    Old debugging and patch scripts kept for reference (not used)
```

---

## 4. Tabular data: cleaning, merge and EDA

* **Inputs:** `customer_transactions` (150 rows) and `customer_social_profiles` (155 rows).
* **Cleaning:** 10 missing `customer_rating` values filled with the median (robust to extreme values); 5 exact duplicate social profiles removed.
* **ID fix:** transactions used `151`, social profiles used `A151`, so transaction IDs were given the `A` prefix.
* **Multiple profiles per customer:** grouped by customer (mean engagement and interest scores, combined platform list, platform count, most common sentiment), leaving 84 unique customers.
* **Merge:** a **left join** on `customer_id`, so no transaction is lost. Checked with `validate="many_to_one"`. Result: 150 rows (117 matched, 33 without a social profile). Row count is unchanged, so the join did not duplicate transactions.
* **EDA plots:** purchase amount histogram, box plot by category (outliers), correlation heatmap, category counts. Categories are fairly balanced (Sports 35, Electronics 34, Clothing 28, Groceries 28, Books 25).
* **Feature engineering:** `engagement_level` and `interest_level` (Low, Medium, High bins), `platform_count`, and one column per social platform. Purchase amount, date and rating are excluded from the inputs because they are only known after the purchase.

## 5. Images

* 4 members x 3 expressions (neutral, smile, surprised), named `member_XX_<expression>.jpg`, plus one stranger image.
* Faces detected and cropped with OpenCV Haar cascades, resized to 128x128.
* **6 augmentations per image:** rotate +15°, rotate −15°, horizontal flip, grayscale, brightness x1.3, Gaussian noise.
* **Features (382 per image):** 48 colour histogram values (16 bins each for hue, saturation and brightness), 324 HOG values (edge shape) and 10 LBP values (skin texture), saved to `data/image_features.csv`.

## 6. Audio

* Each member recorded "Yes, approve" and "Confirm transaction" (8 recordings).
* Waveforms and mel spectrograms are plotted and interpreted in the audio notebook.
* **3 augmentations per recording:** pitch shift (+2 semitones), time stretch (1.1x), background noise (20 dB SNR).
* **Features:** 13 MFCC means, spectral roll-off mean and RMS energy mean, saved to `data/audio_features.csv`.

## 7. Models and evaluation

All metrics are on data the model did not train on. Augmented copies of a photo or recording are always kept in the same fold as the original, so they cannot leak into the test set.

| Model | Algorithm | Accuracy | Weighted F1 | Log loss | Evaluation |
|---|---|---|---|---|---|
| Face recognition | Random Forest | 0.952 | 0.948 | 0.587 | Leave one expression out (test photos never seen, even augmented) |
| Voiceprint verification | Random Forest | 0.625 | 0.550 | 0.986 | Leave one recording (and its augmentations) out |
| Product recommendation | Logistic Regression | 0.316 | 0.254 | 1.720 | Held-out test set (baseline: 0.211 accuracy) |

**Rejecting strangers.** A classifier always picks one of the known classes, even for a stranger. So both gates also need a confidence of at least 60%, and the voice model additionally rejects any sample that is too far from the matched member's real recordings.

**Honest limitations.** The voice model is trained on only 2 recordings per person, so its held-out accuracy is modest; more recordings would help most. The product model beats the "always guess the most common product" baseline but is still weak, because social media engagement and sentiment carry little signal about which category someone buys.

## 8. System demonstration

`python app.py --demo` runs:

| # | Face | Voice | Result |
|---|---|---|---|
| 1 | member_01 | member_01 | **APPROVED**, product recommended |
| 2 | stranger | member_01 | DENIED at face check |
| 3 | member_02 | member_03 | DENIED, face and voice do not match |
| 4 | member_04 | stranger | DENIED at voice check |

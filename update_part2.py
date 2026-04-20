import json

answer_cell = r"""#### **Answer — Similarity Function**

The function `similarity_matrix(T, D)` computes the **Euclidean distance** between two flattened image vectors:

$$A_{i,j} = \|\vec{t}_i - \vec{d}_j\|_2 = \sqrt{\sum_{k=1}^{D}(t_{ik} - d_{jk})^2}$$

where $D = 56 \times 46 = 2576$ is the number of pixels. A **small value** means the two images are close in pixel space (potentially the same person); a **large value** means they differ greatly.

**Design choice — why store `sim` as shape `(280, 40, 3)` instead of `(280, 120)`?**

Our matrix has three indices:
- **Axis 0** — 280 test images (40 people × 7 test poses each)
- **Axis 1** — 40 training people
- **Axis 2** — 3 training poses per person

Keeping axis 2 separate allows us to directly compute `np.min(sim[i, p, :])` — the closest training pose for person $p$ to test image $i$ — without manually slicing a flat column range. It is equivalent to the $(280 \times 120)$ form via `.reshape(280, 120)`."""

shape_cell = r"""**What does `sim.shape = (280, 40, 3)` mean structurally?**

Each row $i$ of `sim` is one test image. Each entry `sim[i, p, q]` answers:

> *"How far is test image $i$ from the $q$-th training pose of person $p$?"*

For face **verification** of person $A$, we only need `sim[i, A, :]` — three distances to person A's training images. We accept if `min(sim[i, A, :]) < t`.

For full **recognition** across all people, we scan all 40 people and pick the person with the smallest minimum distance.

The reshaped form `(280, 120)` treats all 120 training images independently — useful for visualization as a 2D matrix, where columns are ordered: person 0 pose 0, person 0 pose 1, person 0 pose 2, person 1 pose 0, …"""

image_cell = r"""**Image of similarity matrix — how to read it**

The matrix is displayed as a grayscale image where:
- **Bright pixel (high value)** = large Euclidean distance = images are **dissimilar**
- **Dark pixel (low value)** = small Euclidean distance = images are **similar**

---

**What pattern should we expect to see, and why?**

The 280 test rows are ordered by person: rows 0–6 are person 0's 7 test poses, rows 7–13 are person 1's, and so on. The 120 training columns are also ordered: columns 0–2 are person 0's 3 poses, columns 3–5 are person 1's, etc.

Therefore, a **well-discriminating** system should produce a **block-diagonal** pattern:
- **40 dark blocks along the diagonal** (each block 7 rows × 3 columns) = genuine pairs → same person, small distance
- **Bright off-diagonal regions** = impostor pairs → different people, large distance

This is the ideal structure: rows and columns belonging to the same identity should cluster together with low values, while cross-identity entries should be high.

---

**What does our actual image show?**

The image appears mostly **uniform gray with no clear diagonal structure**. This visually confirms what Part 1 showed numerically: raw pixel Euclidean distances **do not discriminate identity**.

**How did this happen?** In the 2576-dimensional pixel space, the within-class distance (same person, different pose) is of comparable magnitude to the between-class distance (different people). As a result:
- The **genuine pair block** on the diagonal is barely darker than the background
- The **impostor region** is not significantly brighter

The distance matrix looks like unstructured noise — every test image is roughly equally far from every training image, regardless of identity match. There is no visible block pattern.

**Why does this matter?**

This failure of the raw-pixel similarity matrix is a direct consequence of the **Curse of Dimensionality** (slides p.3): in very high dimensions, distances between points concentrate around a narrow range, making it hard to distinguish near from far. PCA addresses this by projecting into a low-dimensional subspace where genuine pairs are pulled together and impostor pairs are pushed apart — producing the clear block-diagonal structure that raw pixels cannot achieve."""

with open(r'c:\Users\emper\Desktop\FIBO\2568_2\FRA501_pattern\hw3\merge_2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

updated = 0
for cell in nb['cells']:
    cid = cell.get('id', '')
    if cid == '371e8f63':
        cell['source'] = answer_cell
        updated += 1
    elif cid == '8152b283':
        cell['source'] = shape_cell
        updated += 1
    elif cid == '27c809ee':
        cell['source'] = image_cell
        updated += 1

with open(r'c:\Users\emper\Desktop\FIBO\2568_2\FRA501_pattern\hw3\merge_2.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'Updated {updated} cells')
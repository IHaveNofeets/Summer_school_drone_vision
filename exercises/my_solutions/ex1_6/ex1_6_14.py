import numpy as np
import cv2
import matplotlib.pyplot as plt

# Generalized Histogram Thresholding, from:
# "A Generalization of Otsu's Method and Minimum Error Thresholding" by Jonathan T. Barron
# https://arxiv.org/abs/2007.07350
csum = lambda z: np.cumsum(z)[:-1]
dsum = lambda z: np.cumsum(z[::-1])[-2::-1]
argmax = lambda x, f: np.mean(x[:-1][f == np.max(f)])  # use the mean for ties
clip = lambda z: np.maximum(1e-30, z)


def preliminaries(n, x):
    x = np.arange(len(n), dtype=n.dtype) if x is None else x
    w0 = clip(csum(n))
    w1 = clip(dsum(n))
    p0 = w0 / (w0 + w1)
    p1 = w1 / (w0 + w1)
    mu0 = csum(n * x) / w0
    mu1 = dsum(n * x) / w1
    d0 = csum(n * x**2) - w0 * mu0**2
    d1 = dsum(n * x**2) - w1 * mu1**2
    return x, w0, w1, p0, p1, mu0, mu1, d0, d1


def GHT(n, x=None, nu=0, tau=0, kappa=0, omega=0.5):
    x, w0, w1, p0, p1, _, _, d0, d1 = preliminaries(n, x)
    v0 = clip((p0 * nu * tau**2 + d0) / (p0 * nu + w0))
    v1 = clip((p1 * nu * tau**2 + d1) / (p1 * nu + w1))
    f0 = -d0 / v0 - w0 * np.log(v0) + 2 * (w0 + kappa * omega) * np.log(w0)
    f1 = -d1 / v1 - w1 * np.log(v1) + 2 * (w1 + kappa * (1 - omega)) * np.log(w1)
    return argmax(x, f0 + f1), f0 + f1


image = cv2.imread('exercises/my_solutions/ex1/1_6_11_distances.png', cv2.IMREAD_GRAYSCALE)

hist_n, hist_edges = np.histogram(image, bins=np.arange(-0.5, 256))
hist_x = (hist_edges[1:] + hist_edges[:-1]) / 2.0

ght_value, _ = GHT(hist_n, hist_x, nu=2**5, tau=2**10, kappa=0.1, omega=0.5)
print(f"GHT threshold: {ght_value}")

mask = ((image >= ght_value) * 255).astype(np.uint8)
cv2.imwrite('exercises/my_solutions/ex1/1_6_14_mask.png', mask)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Distance image')
axes[0].axis('off')

axes[1].imshow(mask, cmap='gray')
axes[1].set_title(f'GHT segmentation (t={ght_value:.1f})')
axes[1].axis('off')

plt.savefig('exercises/my_solutions/ex1/1_6_14_result.png')
plt.show()

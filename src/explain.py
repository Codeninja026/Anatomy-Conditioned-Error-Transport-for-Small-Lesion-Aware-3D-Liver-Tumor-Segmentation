import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def tumor_gradcam(model, x, spacing=None):
    """3D Grad-CAM for the final decoder feature map and tumor logit."""
    activation = {}
    def hook(_, __, output):
        output.retain_grad()
        activation["value"] = output
    handle = model.decoders[-1].register_forward_hook(hook)
    model.zero_grad(set_to_none=True)
    out = model(x, spacing)
    score = out["logits"][:, 2].mean()
    score.backward()
    feat = activation["value"]
    grad = feat.grad
    weights = grad.mean(dim=(2, 3, 4), keepdim=True)
    cam = F.relu((weights * feat).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[2:], mode="trilinear", align_corners=False)
    cam = cam / (cam.amax(dim=(2, 3, 4), keepdim=True) + 1e-6)
    handle.remove()
    return cam.detach(), out

def save_gradcam(image, mask, pred, cam, path, title):
    fig, ax = plt.subplots(1, 4, figsize=(15, 4))
    base = image[image.shape[0] // 2]
    ax[0].imshow(base, cmap="gray"); ax[0].set_title("CT")
    ax[1].imshow(base, cmap="gray"); ax[1].imshow(mask, alpha=.45, vmin=0, vmax=2); ax[1].set_title("Ground truth")
    ax[2].imshow(base, cmap="gray"); ax[2].imshow(pred, alpha=.45, vmin=0, vmax=2); ax[2].contour(pred == 2, colors="red"); ax[2].set_title("Tumor outlined")
    ax[3].imshow(base, cmap="gray"); ax[3].imshow(cam, cmap="jet", alpha=.55); ax[3].contour(mask == 2, colors="white", linewidths=.6); ax[3].set_title("Tumor Grad-CAM")
    for a in ax: a.axis("off")
    fig.suptitle(title); fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=150); plt.close(fig)

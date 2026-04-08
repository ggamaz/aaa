import cv2
import numpy as np
from PIL import Image, ImageFilter
import math, random, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import tqdm


# ─────────────────────────── 基础工具 ────────────────────────────

def screen(a, b, alpha=1.0):
    blended = 1.0 - (1.0 - a) * (1.0 - b)
    return a * (1 - alpha) + blended * alpha

def to_f(img):
    return img.astype(np.float32) / 255.0

def to_u8(img):
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def detect_lights(img_u8, thresh=210, max_n=3):
    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lights = []
    for c in cnts:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        x, y, w, h = cv2.boundingRect(c)
        brightness = float(np.percentile(gray[y:y + h, x:x + w], 95))
        lights.append(dict(pos=(cx, cy), size=max(w, h), brightness=brightness))
    lights.sort(key=lambda l: l["brightness"], reverse=True)
    return lights[:max_n]


def make_glow(H, W, cx, cy, base_r, intensity, scale=0.5):
    """辉光：降分辨率计算后上采样"""
    h, w = max(1, int(H * scale)), max(1, int(W * scale))
    scx, scy, sr = cx * scale, cy * scale, base_r * scale
    ys = np.arange(h, dtype=np.float32) - scy
    xs = np.arange(w, dtype=np.float32) - scx
    xx, yy = np.meshgrid(xs, ys)
    r2 = xx ** 2 + yy ** 2
    layers = [
        (sr * 0.4, intensity * 1.00),
        (sr * 1.2, intensity * 0.55),
        (sr * 3.0, intensity * 0.25),
        (sr * 7.0, intensity * 0.10),
    ]
    canvas = np.zeros((h, w), np.float32)
    for r, s in layers:
        canvas += np.exp(-0.5 * r2 / max(r ** 2, 1e-6)) * s
    out = np.empty((h, w, 3), np.float32)
    out[:, :, 0] = canvas * 1.00
    out[:, :, 1] = canvas * 0.96
    out[:, :, 2] = canvas * 0.85
    np.clip(out, 0, 1, out=out)
    return cv2.resize(out, (W, H), interpolation=cv2.INTER_LINEAR)


def make_bokeh(H, W, cx, cy, max_r, intensity, scale=0.5):
    """Bokeh 环：降分辨率计算后上采样"""
    h, w = max(1, int(H * scale)), max(1, int(W * scale))
    scx, scy, sr = cx * scale, cy * scale, max_r * scale
    ys = np.arange(h, dtype=np.float32) - scy
    xs = np.arange(w, dtype=np.float32) - scx
    xx, yy = np.meshgrid(xs, ys)
    dist = np.sqrt(xx ** 2 + yy ** 2)
    radii  = [sr * f for f in (0.30, 0.55, 0.80, 1.00)]
    colors = [(1.0, 0.92, 0.78), (0.82, 0.90, 1.0), (1.0, 0.88, 0.95), (0.90, 1.0, 0.88)]
    canvas = np.zeros((h, w, 3), np.float32)
    for i, (r, col) in enumerate(zip(radii, colors)):
        thickness = max(4 * scale, r * 0.10)
        ring = np.exp(-0.5 * ((dist - r) / thickness) ** 2) * intensity * (0.65 ** i)
        canvas[:, :, 0] += ring * col[0]
        canvas[:, :, 1] += ring * col[1]
        canvas[:, :, 2] += ring * col[2]
    np.clip(canvas, 0, 1, out=canvas)
    return cv2.resize(canvas, (W, H), interpolation=cv2.INTER_LINEAR)


def make_streaks(H, W, xx, yy, n, length, intensity):
    """
    衍射星芒：接受预计算的 xx/yy（以光源为原点）。
    主芒 + 次芒，1/r 衰减，末端色散。
    """
    angles, strengths = [], []
    for i in range(n):
        base = math.pi * i / n
        angles.append(base + random.uniform(-0.01, 0.01));    strengths.append(1.00)
        angles.append(base + math.pi / (2 * n) + random.uniform(-0.01, 0.01)); strengths.append(0.40)

    canvas_r = np.zeros((H, W), np.float32)
    canvas_g = np.zeros((H, W), np.float32)
    canvas_b = np.zeros((H, W), np.float32)
    width = max(0.8, length * 0.003)

    for angle, s_mult in zip(angles, strengths):
        dx, dy = math.cos(angle), math.sin(angle)
        proj  =  xx * dx + yy * dy          # 沿芒方向投影
        perp  = -xx * dy + yy * dx          # 垂直方向
        perp_m = np.exp(-0.5 * (perp / width) ** 4)
        proj_pos = np.maximum(proj, 0)
        along = (1.0 / (proj_pos / (length * 0.06) + 1.0) ** 1.8
                 * (proj > 0)
                 * np.clip(1.0 - proj / (length * 1.05), 0, 1))
        streak = along * perp_m * (intensity * s_mult)
        t = np.clip(proj_pos / length, 0, 1)
        canvas_r += streak * (1.00 + t * 0.06)
        canvas_g += streak * (1.00 - t * 0.03)
        canvas_b += streak * (1.00 - t * 0.10)

    out = np.stack([canvas_r, canvas_g, canvas_b], axis=2)
    np.clip(out, 0, 1, out=out)
    return out


def make_ghosts(H, W, cx, cy, img_cx, img_cy, intensity, scale=0.5):
    """镜头鬼影：降分辨率计算后上采样"""
    h, w = max(1, int(H * scale)), max(1, int(W * scale))
    canvas = np.zeros((h, w, 3), np.float32)
    diag = math.sqrt(img_cx ** 2 + img_cy ** 2)
    ghosts = [
        (0.5, 0.07, (1.0, 0.5, 0.2)),
        (0.9, 0.05, (0.3, 0.6, 1.0)),
        (1.3, 0.04, (0.8, 0.3, 1.0)),
    ]
    for t, sc, col in ghosts:
        gx = int((img_cx + t * (img_cx - cx)) * scale)
        gy = int((img_cy + t * (img_cy - cy)) * scale)
        r  = max(4 * scale, diag * sc * scale)
        ys2 = np.arange(h, dtype=np.float32) - gy
        xs2 = np.arange(w, dtype=np.float32) - gx
        xx2, yy2 = np.meshgrid(xs2, ys2)
        d2 = np.sqrt(xx2 ** 2 + yy2 ** 2)
        mask = (np.exp(-0.5 * ((d2 - r * 0.6) / (r * 0.12)) ** 2) * 0.6
                + np.exp(-0.5 * (d2 / (r * 0.20)) ** 2) * 0.4) * intensity
        for ch in range(3):
            canvas[:, :, ch] = np.maximum(canvas[:, :, ch], mask * col[ch])
    np.clip(canvas, 0, 1, out=canvas)
    return cv2.resize(canvas, (W, H), interpolation=cv2.INTER_LINEAR)


def make_flare_veil(H, W, cx, cy, intensity, scale=0.5):
    """炫光雾：降分辨率计算后上采样"""
    h, w = max(1, int(H * scale)), max(1, int(W * scale))
    scx, scy = cx * scale, cy * scale
    ys = np.arange(h, dtype=np.float32) - scy
    xs = np.arange(w, dtype=np.float32) - scx
    xx, yy = np.meshgrid(xs, ys)
    r = np.sqrt(xx ** 2 + yy ** 2)
    diag = math.sqrt((H / 2) ** 2 + (W / 2) ** 2) * scale
    veil = (np.exp(-r / (diag * 0.6)) * 0.6 + 0.4) * intensity
    out = np.empty((h, w, 3), np.float32)
    out[:, :, 0] = veil * 1.00
    out[:, :, 1] = veil * 0.97
    out[:, :, 2] = veil * 0.90
    np.clip(out, 0, 1, out=out)
    return cv2.resize(out, (W, H), interpolation=cv2.INTER_LINEAR)


def chromatic_aberration_radial(img_f, strength=0.004):
    """径向色差：红通道向外偏移，蓝通道向内收缩"""
    H, W = img_f.shape[:2]
    cx, cy = W / 2.0, H / 2.0
    xs = (np.arange(W, dtype=np.float32) - cx) / cx
    ys = (np.arange(H, dtype=np.float32) - cy) / cy
    xx, yy = np.meshgrid(xs, ys)
    result = img_f.copy()
    for ch, sc in ((0, 1.0 + strength), (2, 1.0 - strength)):
        map_x = (xx * sc * cx + cx).astype(np.float32)
        map_y = (yy * sc * cy + cy).astype(np.float32)
        result[:, :, ch] = cv2.remap(
            img_f[:, :, ch], map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
    return result


def local_contrast_loss(img_f, xx, yy, radius, strength=0.45):
    """
    光源附近局部对比度衰减。
    接受预计算的 xx/yy（以光源为原点），避免重复 meshgrid。
    """
    r = np.sqrt(xx ** 2 + yy ** 2)
    weight = np.exp(-0.5 * (r / (radius * 0.8)) ** 2) * strength
    weight = weight[:, :, np.newaxis]
    gray = img_f.mean(axis=2, keepdims=True)
    return img_f * (1 - weight) + gray * weight


def add_film_grain(img_f, sigma=0.018):
    noise = np.random.normal(0, sigma, img_f.shape).astype(np.float32)
    return np.clip(img_f + noise, 0, 1)


# ─────────────────────── 主流程 ──────────────────────────────────

def apply_lens_flare(input_path: Path, output_path: Path,
                     flare_strength=1.0,
                     overexpose=0.45,
                     manual_lights=None):
    img = np.array(Image.open(input_path).convert("RGB"))
    H, W = img.shape[:2]
    img_cx, img_cy = W // 2, H // 2
    base = to_f(img)

    if manual_lights:
        lights = [dict(pos=p, size=40, brightness=230) for p in manual_lights]
    else:
        lights = detect_lights(img)
        if not lights:
            lights = [dict(pos=(W // 2, H // 3), size=30, brightness=220)]

    composite = base.copy()
    short_side = min(H, W)

    # ── 关键优化：整张图的坐标网格只创建一次 ──────────────────
    # 各效果函数直接用 (xx_light, yy_light) = 全图网格 - 光源位置，
    # 这只是原地减法，不产生新的内存分配。
    _col = np.arange(W, dtype=np.float32)   # shape (W,)
    _row = np.arange(H, dtype=np.float32)   # shape (H,)
    XX, YY = np.meshgrid(_col, _row)        # shape (H, W)，仅此一次

    for L in lights:
        lx, ly = L["pos"]
        bf = min(1.0, L["brightness"] / 240.0)
        fi = flare_strength * bf

        # 以光源为原点的坐标（view，不复制内存）
        xx = XX - lx   # 实际上会新建数组，但比 meshgrid 快很多
        yy = YY - ly

        # 辉光（半分辨率）
        glow_r = int(short_side * 0.10 * fi)
        glow = make_glow(H, W, lx, ly, glow_r, intensity=0.55 * fi)
        composite = screen(composite, glow, alpha=0.9)

        # Bokeh（半分辨率）
        bokeh_r = int(short_side * 0.38 * fi)
        bokeh = make_bokeh(H, W, lx, ly, bokeh_r, intensity=0.15 * fi)
        composite = screen(composite, bokeh, alpha=0.6)

        # 星芒（全分辨率，复用 xx/yy）
        streak_len = int(short_side * 0.45 * fi)
        streaks = make_streaks(H, W, xx, yy, 6, streak_len, intensity=0.30 * fi)
        composite = screen(composite, streaks, alpha=0.85)

        # 鬼影（半分辨率）
        ghosts = make_ghosts(H, W, lx, ly, img_cx, img_cy, intensity=0.15 * fi)
        composite = screen(composite, ghosts, alpha=0.65)

        # 炫光雾（半分辨率）
        veil = make_flare_veil(H, W, lx, ly, intensity=0.12 * fi)
        composite = screen(composite, veil, alpha=0.75)

        # 局部对比度衰减（复用 xx/yy，无 meshgrid）
        loss_r = int(short_side * 0.22 * fi)
        composite = local_contrast_loss(composite, xx, yy, loss_r, strength=0.40 * fi)

    # 全局高亮辉光
    blur_k = max(3, (int(short_side * 0.06) | 1))
    blurred = cv2.GaussianBlur(to_u8(composite), (blur_k, blur_k), 0).astype(np.float32) / 255
    bright_w = np.power(np.clip(composite.mean(2, keepdims=True) - 0.5, 0, 1) / 0.5, 2)
    composite = screen(composite, blurred * bright_w, alpha=0.4)

    # 径向色差
    ca_strength = 0.002 + 0.003 * flare_strength
    composite = chromatic_aberration_radial(composite, strength=ca_strength)

    # 过曝
    gamma = 1.0 - overexpose * 0.35
    composite = np.power(np.clip(composite, 1e-6, 1), gamma)
    composite = composite + overexpose * 0.04
    np.clip(composite, 0, 1, out=composite)

    # 胶片颗粒
    composite = add_film_grain(composite, sigma=0.012)

    # 拼接：NIR | RGB原图 | 炫光图
    nir_img = np.array(Image.open(str(input_path).replace("rgb", "nir")).convert("L"))
    nir_img = np.stack([nir_img]*3, axis=2) / 255.0
    composite = np.concatenate([nir_img, base, composite], axis=1)

    out = Image.fromarray(to_u8(composite))
    out = out.filter(ImageFilter.GaussianBlur(0.6))
    out.save(output_path, quality=95)
    return out


# ─────────────────────── 多进程批处理 ────────────────────────────

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def _process_one(args):
    """多进程工作函数（顶层函数，可被 pickle）"""
    src, dst, flare_strength, overexpose = args
    try:
        apply_lens_flare(Path(src), Path(dst),
                         flare_strength=flare_strength,
                         overexpose=overexpose)
        return src, True, None
    except Exception as e:
        return src, False, str(e)


def process_directory(input_dir, output_dir,
                      flare_strength=1.0,
                      overexpose=0.45,
                      suffix="",
                      out_ext=".jpg",
                      num_workers=None):
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in input_dir.iterdir()
                    if p.suffix.lower() in SUPPORTED_EXTS)
    if not images:
        print(f"⚠️  目录 {input_dir} 下未找到支持的图像文件。")
        return

    if num_workers is None:
        num_workers = min(os.cpu_count() or 1, 8)

    print(f"📂 输入目录: {input_dir}  ({len(images)} 张图)")
    print(f"📂 输出目录: {output_dir}")
    print(f"   flare_strength={flare_strength}  overexpose={overexpose}"
          f"  workers={num_workers}\n")

    # 传字符串而非 Path，避免不同 Python 版本 pickle 问题
    tasks = [
        (str(src),
         str(output_dir / (src.stem + suffix + out_ext)),
         flare_strength,
         overexpose)
        for src in images
    ]

    ok = fail = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futs = {executor.submit(_process_one, t): t[0] for t in tasks}
        for fut in tqdm.tqdm(as_completed(futs), total=len(tasks)):
            _, success, err = fut.result()
            if success:
                ok += 1
            else:
                print(f"  ❌ {Path(futs[fut]).name}  错误: {err}")
                fail += 1

    print(f"\n完成：成功 {ok} 张，失败 {fail} 张。")


if __name__ == "__main__":
    inp_dir = "/home/visdrone/Music/data_process/COCO/images/val2017"
    out_dir = inp_dir + "_flare"
    process_directory(inp_dir, out_dir, flare_strength=1.0, overexpose=0.6)
    
    
    

"""
Samakan ukuran sepasang gambar pre/post-disaster sebelum diupload ke dashboard.

Cara pakai:
    pip install pillow
    python resize_pair.py pre.jpg post.jpg

Hasil: pre_resized.png dan post_resized.png (ukuran sama, siap diupload).

Strategi: ambil ukuran TERKECIL di antara kedua gambar (width & height masing-masing),
lalu resize keduanya ke ukuran itu. Ini menghindari perlu cropping tebakan dan tidak
membuat gambar jadi lebih besar dari aslinya (menghindari upscaling yang blur).
"""
import sys
from PIL import Image

def main():
    if len(sys.argv) != 3:
        print("Cara pakai: python resize_pair.py <pre_image> <post_image>")
        sys.exit(1)

    pre_path, post_path = sys.argv[1], sys.argv[2]
    pre_img = Image.open(pre_path).convert("RGB")
    post_img = Image.open(post_path).convert("RGB")

    target_w = min(pre_img.width, post_img.width)
    target_h = min(pre_img.height, post_img.height)
    print(f"Ukuran asli — pre: {pre_img.size}, post: {post_img.size}")
    print(f"Menyamakan ke: {target_w}x{target_h}")

    pre_resized = pre_img.resize((target_w, target_h), Image.LANCZOS)
    post_resized = post_img.resize((target_w, target_h), Image.LANCZOS)

    pre_resized.save("pre_resized.png")
    post_resized.save("post_resized.png")
    print("Selesai! Upload pre_resized.png dan post_resized.png ke dashboard.")


if __name__ == "__main__":
    main()

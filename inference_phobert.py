import os
import sys
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_tokenizer(model_dir: str):
    try:
        return AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    except Exception:
        return AutoTokenizer.from_pretrained("vinai/phobert-large", use_fast=False)


def load_model(model_dir: str, device: torch.device):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return model


def format_dialog(text: str, add_markers: bool) -> str:
    if not add_markers:
        return text
    if "[USER]" in text or "[AGENT]" in text:
        return text
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return text
    marked = []
    for idx, ln in enumerate(lines):
        prefix = "[USER]" if idx % 2 == 0 else "[AGENT]"
        marked.append(f"{prefix} {ln}")
    return "\n".join(marked)


def predict(
    text: str,
    model_dir: str = "model",
    max_length: int = 256,
    add_markers: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(model_dir)
    model = load_model(model_dir, device)

    prepared = format_dialog(text, add_markers)
    enc = tokenizer(
        prepared,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        outputs = model(**enc)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).tolist()

    pred_id = int(torch.argmax(logits, dim=-1).item())
    label_names = {0: "Không lừa đảo", 1: "Lừa đảo"}
    label = label_names.get(pred_id, str(pred_id))
    score = probs[pred_id]

    return {
        "label_id": pred_id,
        "label": label,
        "score": float(score),
        "probs": {"not_scam": float(probs[0]), "scam": float(probs[1]) if len(probs) > 1 else None},
        "text": prepared,
        "device": str(device),
    }


def main():
    parser = argparse.ArgumentParser(description="PhoBERT scam classifier inference")
    parser.add_argument("--model_dir", type=str, default="model", help="Path to saved model directory")
    parser.add_argument("--text", type=str, default=None, help="Input text or conversation snippet")
    parser.add_argument("--file", type=str, default=None, help="Path to a text file to read input from")
    parser.add_argument("--max_length", type=int, default=256, help="Max sequence length for tokenization")
    parser.add_argument("--add_markers", action="store_true", help="Auto add [USER]/[AGENT] markers alternately per line if absent")
    args = parser.parse_args()

    if args.text is None and args.file is None:
        print("Nhập đoạn hội thoại (kết thúc bằng Ctrl+D/Ctrl+Z):")
        try:
            raw = sys.stdin.read()
        except KeyboardInterrupt:
            raw = ""
    elif args.file is not None:
        if not os.path.exists(args.file):
            print(f"Không tìm thấy file: {args.file}")
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as f:
            raw = f.read()
    else:
        raw = args.text

    if not raw or not raw.strip():
        print("Không có nội dung để dự đoán.")
        sys.exit(1)

    result = predict(
        text=raw.strip(),
        model_dir=args.model_dir,
        max_length=args.max_length,
        add_markers=args.add_markers,
    )

    print(f"Nhãn: {result['label']}  (id={result['label_id']}, score={result['score']:.4f})")
    if result["probs"]["scam"] is not None:
        print(f"Xác suất - Không lừa đảo: {result['probs']['not_scam']:.4f} | Lừa đảo: {result['probs']['scam']:.4f}")


if __name__ == "__main__":
    main()



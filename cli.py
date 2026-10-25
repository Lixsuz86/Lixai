"""
LixAI Komut Satırı Arayüzü (CLI)
Kullanım:
  python cli.py train --steps 50
  python cli.py generate --prompt "Once upon a time" --tokens 30 --load-best
  python cli.py benchmark
"""
import argparse
import sys
from pathlib import Path

# Windows Türkçe konsolunda UTF-8 desteği
try:
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.stderr.encoding != "utf-8":
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.engine.interface import LixAI


def main():
    parser = argparse.ArgumentParser(description="LixAI: Intel CPU Optimize Dil Modeli CLI")
    subparsers = parser.add_subparsers(dest="command", help="Komutlar")

    # Train
    train_parser = subparsers.add_parser("train", help="Modeli eğit")
    train_parser.add_argument("--steps", type=int, default=50, help="Eğitim adım sayısı")
    train_parser.add_argument("--dataset", type=str, default=None, help="Hugging Face dataset adı (opsiyonel)")
    train_parser.add_argument("--batch-size", type=int, default=None, help="Batch boyutu")

    # Train SFT (Instruction Tuning)
    sft_parser = subparsers.add_parser("train-sft", help="Diyalog veri setiyle SFT (Instruction Tuning) eğitimi yap")
    sft_parser.add_argument("--steps", type=int, default=30, help="SFT eğitim adım sayısı")
    sft_parser.add_argument("--dataset", type=str, default="data/dialogue_sft.jsonl", help="SFT JSONL veri yolu")
    sft_parser.add_argument("--batch-size", type=int, default=4, help="Batch boyutu")
    sft_parser.add_argument("--lr", type=float, default=2e-4, help="Öğrenme oranı")
    sft_parser.add_argument("--checkpoint", type=str, default="lixai_chat_sft.pt", help="Kaydedilecek model adı")

    # Generate
    gen_parser = subparsers.add_parser("generate", help="Metin üret")
    gen_parser.add_argument("--prompt", type=str, required=True, help="Başlangıç metni")
    gen_parser.add_argument("--tokens", type=int, default=30, help="Üretilecek token sayısı")
    gen_parser.add_argument("--temperature", type=float, default=0.8, help="Yaratıcılık katsayısı")
    gen_parser.add_argument("--top-k", type=int, default=40, help="Top-K filtreleme sınırı")
    gen_parser.add_argument("--top-p", type=float, default=0.9, help="Top-P nucleus örnekleme eşiği (default: 0.9)")
    gen_parser.add_argument("--min-p", type=float, default=0.05, help="Min-P taban olasılık oranı (default: 0.05)")
    gen_parser.add_argument("--repetition-penalty", "--rep-penalty", type=float, default=1.15, help="Tekrarlama cezası katsayısı (default: 1.15)")
    gen_parser.add_argument("--quantize", action="store_true", help="INT8 hızlandırmalı çıkarım kullan")
    gen_parser.add_argument("--load-best", action="store_true", help="Kaydedilmiş en iyi ağırlıkları yükle (best_model.pt)")
    gen_parser.add_argument("--checkpoint", type=str, default=None, help="Yüklenecek özel checkpoint dosya adı")
    gen_parser.add_argument("--no-checkpoint", action="store_true", help="Kayıtlı ağırlıkları yükleme, sıfırdan rastgele başlat")

    # Chat (İnteraktif CLI Sohbet Ekranı)
    chat_parser = subparsers.add_parser("chat", help="Canlı interaktif terminal sohbet ekranını başlat")
    chat_parser.add_argument("--temperature", type=float, default=0.8, help="Yaratıcılık katsayısı")
    chat_parser.add_argument("--top-k", type=int, default=40, help="Top-K filtreleme sınırı")
    chat_parser.add_argument("--top-p", type=float, default=0.9, help="Top-P nucleus örnekleme eşiği (default: 0.9)")
    chat_parser.add_argument("--min-p", type=float, default=0.05, help="Min-P taban olasılık oranı (default: 0.05)")
    chat_parser.add_argument("--repetition-penalty", "--rep-penalty", type=float, default=1.15, help="Tekrarlama cezası katsayısı (default: 1.15)")
    chat_parser.add_argument("--quantize", action="store_true", help="INT8 hızlandırma modunda başlat")
    chat_parser.add_argument("--load-best", action="store_true", help="Kaydedilmiş en iyi ağırlıkları yükle (best_model.pt)")
    chat_parser.add_argument("--checkpoint", type=str, default=None, help="Yüklenecek özel checkpoint dosya adı")
    chat_parser.add_argument("--no-checkpoint", action="store_true", help="Kayıtlı ağırlıkları yükleme, sıfırdan rastgele başlat")

    # Benchmark
    subparsers.add_parser("benchmark", help="CPU FP32 vs INT8 hız testi yap")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    load_best = None
    if getattr(args, "no_checkpoint", False):
        load_best = False
    elif getattr(args, "load_best", False):
        load_best = True

    ai = LixAI(load_best=load_best, checkpoint_name=getattr(args, "checkpoint", None))

    if args.command == "chat":
        from src.engine.chat import InteractiveChatSession
        session = InteractiveChatSession(
            model=ai.model,
            tokenizer=ai.tokenizer,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            min_p=args.min_p,
            repetition_penalty=args.repetition_penalty,
            use_quantized=args.quantize,
            checkpoint_name=ai.loaded_checkpoint_name,
        )
        session.run_interactive_loop()

    if args.command == "train":
        print(f"\n🚀 LixAI Eğitimi Başlatılıyor ({args.steps} adım)...")
        results = ai.train(
            max_steps=args.steps,
            dataset_name=args.dataset,
            batch_size=args.batch_size,
        )
        print("\n✅ Eğitim Tamamlandı!")
        print(
            f"Toplam Süre: {results['total_time']:.2f} sn | "
            f"Tamamlanan Adım: {results['completed_steps']} | "
            f"En İyi Kayıp (Loss): {results.get('best_loss', 0.0):.4f}"
        )

    elif args.command == "train-sft":
        print(f"\n🚀 LixAI SFT Asistan Eğitimi Başlatılıyor ({args.steps} adım)...")
        results = ai.train_sft(
            dataset_path=args.dataset,
            max_steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            checkpoint_name=args.checkpoint,
        )
        print("\n✅ SFT Eğitimi Başarıyla Tamamlandı!")
        print(
            f"Toplam Süre: {results['total_time']:.2f} sn | "
            f"Tamamlanan Adım: {results['completed_steps']} | "
            f"Son Kayıp (Loss): {results.get('final_loss', 0.0):.4f} | "
            f"En İyi Kayıp (Loss): {results.get('best_loss', 0.0):.4f} | "
            f"Ortalama Token Doğruluğu: %{results.get('avg_accuracy', 0.0):.1f} | "
            f"Kaydedilen Model: {results.get('saved_checkpoint')}"
        )

    elif args.command == "generate":
        print(f"\n🧠 Metin Üretiliyor (Kuantize: {args.quantize})...")
        output = ai.generate(
            prompt=args.prompt,
            max_new_tokens=args.tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            min_p=args.min_p,
            repetition_penalty=args.repetition_penalty,
            use_quantized=args.quantize,
        )
        print("\n--- Üretilen Metin ---")
        print(output)
        print("----------------------\n")

    elif args.command == "benchmark":
        print("\n⚡ CPU Benchmark Testi Başlatılıyor...")
        results = ai.benchmark()
        print("\n=== Benchmark Sonuçları ===")
        print(f"FP32 Hız: {results['fp32']['tokens_per_sec']} token/sn ({results['fp32']['ms_per_token']} ms/token)")
        print(f"INT8 Hız: {results['int8']['tokens_per_sec']} token/sn ({results['int8']['ms_per_token']} ms/token)")
        print(f"Model Boyutu: {results['fp32']['model_size_mb']} MB")
        print(f"Hızlanma Oranı: {results['speedup_ratio']}x\n")


if __name__ == "__main__":
    main()

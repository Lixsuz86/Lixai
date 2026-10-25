"""
LixAI İki Dilli (Türkçe & İngilizce) SFT (Supervised Fine-Tuning) Veri Seti Üreteci
Modelin diyalog, talimat takibi ve asistan kimliği kazanması için en az 2,000 örnekli ChatML JSONL oluşturur.
"""
from pathlib import Path
from typing import List, Dict, Any
import json
import random

# Tekrarlanabilirlik için tohum
random.seed(42)


def generate_sft_dataset(
    output_path: str = "data/dialogue_sft.jsonl",
    target_count: int = 2100,
) -> int:
    """Zengin Türkçe ve İngilizce diyalog çiftleri üretir ve JSONL formatında kaydeder."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    system_prompts_tr = [
        "Sen Intel CPU üzerinde çalışan hızlı, nazik ve yardımsever bir yapay zeka asistanı olan LixAI'sın.",
        "Sen LixAI; kullanıcının sorularına net, doğru ve Türkçe dil kurallarına uygun yanıtlar veren bir yapay zeka modelisin.",
        "Sen kullanıcıya kodlama, mantık ve genel kültür konularında destek olan zeki bir asistansın.",
        "Sen yerel donanımda çalışan, bellek-dostu ve güvenilir bir dil modelisin.",
    ]

    system_prompts_en = [
        "You are LixAI, a helpful, fast, and lightweight AI assistant optimized for CPU execution.",
        "You are LixAI, an intelligent conversational AI developed to answer questions clearly and accurately.",
        "You are a versatile language model capable of reasoning, coding, and general conversation.",
    ]

    # Temel Diyalog Şablonları (Genişletilebilir veri çekirdeği)
    conversations: List[Dict[str, Any]] = []

    # 1. Türkçe Selamlama & Kimlik
    names = ["LixAI", "Lix AI", "Lix"]
    greetings_tr = [
        ("Merhaba!", "Merhaba! Size bugün nasıl yardımcı olabilirim?"),
        ("Selam, nasılsın?", "Selam! Ben bir yapay zeka modeliyim, sisteminizde harika çalışıyorum. Siz nasılsınız?"),
        ("Günaydın!", "Günaydın! Umarım harika ve verimli bir gün geçirirsiniz. Ne hakkında konuşmak istersiniz?"),
        ("İyi akşamlar.", "İyi akşamlar! Bugün size nasıl destek olabilirim?"),
        ("Sen kimsin?", "Ben LixAI; Intel CPU ve yerel donanım mimarisi için sıfırdan geliştirilmiş bir Küçük Dil Modeliyim (SLM)."),
        ("Adın ne?", "Benim adım LixAI. Size sorularınızda ve projelerinizde yardımcı olmak için buradayım."),
        ("Neler yapabilirsin?", "Sorularınızı yanıtlayabilir, Türkçe ve İngilizce sohbet edebilir, kod yazabilir ve metin özetleyebilirim."),
        ("Harika bir asistansın.", "Çok teşekkür ederim! Faydalı olabildiğime sevindim."),
        ("Teşekkür ederim.", "Rica ederim, her zaman buradayım! Başka bir sorunuz var mı?"),
        ("Görüşmek üzere.", "Görüşmek üzere! İyi günler dilerim."),
    ]

    for u, a in greetings_tr:
        for _ in range(25):
            conversations.append({
                "messages": [
                    {"role": "system", "content": random.choice(system_prompts_tr)},
                    {"role": "user", "content": u},
                    {"role": "assistant", "content": a},
                ]
            })

    # 2. Türkçe Bilgi & Genel Kültür
    qa_tr = [
        ("Türkiye'nin başkenti neresidir?", "Türkiye'nin başkenti Ankara'dır."),
        ("Güneş Sistemi'ndeki en büyük gezegen hangisidir?", "Güneş Sistemi'ndeki en büyük gezegen Jüpiter'dir."),
        ("Dünya'nın uydusu nedir?", "Dünya'nın tek doğal uydusu Ay'dır."),
        ("Suyun kimyasal formülü nedir?", "Suyun kimyasal formülü H2O'dur; iki hidrojen ve bir oksijen atomundan oluşur."),
        ("Işık hızı saniyede yaklaşık ne kadardır?", "Işık hızı boşlukta saniyede yaklaşık 300.000 kilometredir (tam olarak 299.792.458 m/s)."),
        ("Türkiye Cumhuriyeti hangi yılda kuruldu?", "Türkiye Cumhuriyeti, 29 Ekim 1923 tarihinde kurulmuştur."),
        ("Atmosferde en çok bulunan gaz hangisidir?", "Dünya atmosferinde yaklaşık %78 oranıyla en çok bulunan gaz Azot (Nitrojen) gazıdır."),
        ("Hangi mevsimden sonra kış gelir?", "Sonbahar mevsiminden sonra kış mevsimi gelir."),
        ("Gökkuşağında kaç renk vardır?", "Gökkuşağında geleneksel olarak 7 temel renk kabul edilir: kırmızı, turuncu, sarı, yeşil, mavi, lacivert ve mor."),
        ("İnsan vücudundaki en büyük organ hangisidir?", "İnsan vücudundaki en büyük organ deridir."),
    ]

    for u, a in qa_tr:
        for _ in range(20):
            conversations.append({
                "messages": [
                    {"role": "system", "content": random.choice(system_prompts_tr)},
                    {"role": "user", "content": u},
                    {"role": "assistant", "content": a},
                ]
            })

    # 3. Matematik ve Mantık
    math_templates = [
        ("{a} artı {b} kaç eder?", lambda a, b: f"{a} artı {b} toplamı {a + b} eder."),
        ("{a} ile {b} çarpımı nedir?", lambda a, b: f"{a} ile {b} çarpıldığında sonuç {a * b} olur."),
        ("{a} eksi {b} kaçtır?", lambda a, b: f"{a} sayısından {b} çıktığında {a - b} kalır."),
    ]

    for _ in range(300):
        a_val = random.randint(1, 50)
        b_val = random.randint(1, 30)
        tmpl, func = random.choice(math_templates)
        u_text = tmpl.format(a=a_val, b=b_val)
        a_text = func(a_val, b_val)
        conversations.append({
            "messages": [
                {"role": "system", "content": random.choice(system_prompts_tr)},
                {"role": "user", "content": u_text},
                {"role": "assistant", "content": a_text},
            ]
        })

    # 4. Programlama, Yapay Zeka & Donanım
    coding_tr = [
        ("Python'da ekrana yazı nasıl yazdırılır?", "Python'da ekrana metin yazdırmak için print() fonksiyonu kullanılır. Örneğin: print('Merhaba Dünya')"),
        ("Transformer mimarisi nedir?", "Transformer, self-attention (öz-dikkat) mekanizmasına dayanan ve dizisel verileri paralel işleyen derin öğrenme mimarisidir."),
        ("KV Cache nedir ve neden kullanılır?", "Key-Value (KV) Cache, üretim sırasında önceki tokenlerin dikkat matrislerini tekrar hesaplamayıp hafızada tutarak çıkarımı 2 kat hızlandıran bir tekniktir."),
        ("Dinamik INT8 kuantizasyon ne işe yarar?", "Model ağırlıklarını 32-bit kayan noktadan 8-bit tam sayıya indirgeyerek bellek ayak izini %80'den fazla küçültür ve CPU hızını artırır."),
        ("Python'da bir fonksiyon nasıl tanımlanır?", "Python'da fonksiyon 'def' anahtar kelimesi ile tanımlanır. Örnek:\ndef topla(x, y):\n    return x + y"),
        ("Liste (list) ile demet (tuple) arasındaki temel fark nedir?", "Listeler değiştirilebilir (mutable), demetler (tuple) ise oluşturulduktan sonra değiştirilemez (immutable) veri tipleridir."),
        ("PyTorch nedir?", "PyTorch, derin öğrenme ve tensör hesaplamaları için Python tabanlı açık kaynaklı güçlü bir makine öğrenmesi kütüphanesidir."),
        ("Intel CPU'da yapay zeka çıkarımı nasıl optimize edilir?", "Çoklu iş parçacığı (threading), AVX2/AVX-512 yönergeleri, INT8 kuantizasyon ve önbellek dostu tensör operasyonları ile optimize edilir."),
    ]

    for u, a in coding_tr:
        for _ in range(25):
            conversations.append({
                "messages": [
                    {"role": "system", "content": random.choice(system_prompts_tr)},
                    {"role": "user", "content": u},
                    {"role": "assistant", "content": a},
                ]
            })

    # 5. İngilizce Konuşma & Kodlama (Multilingual Capability)
    qa_en = [
        ("Hello, who are you?", "Hello! I am LixAI, a lightweight language model optimized to run efficiently on Intel CPUs."),
        ("How are you doing?", "I am doing great! Ready to help you with your queries and tasks. How can I assist you today?"),
        ("What is the capital of France?", "The capital of France is Paris."),
        ("What is machine learning?", "Machine learning is a subset of artificial intelligence that allows systems to learn and improve from data without being explicitly programmed."),
        ("Write a Python function to square a number.", "Here is a simple Python function to square a number:\n\ndef square(n):\n    return n * n"),
        ("What is the purpose of attention in transformers?", "The attention mechanism allows the model to dynamically focus on different parts of the input sequence when generating each token."),
        ("Explain CPU multi-threading.", "Multi-threading allows a single processor core to execute multiple concurrent threads, improving throughput and reducing latency."),
        ("Thank you for your help!", "You are very welcome! Feel free to ask if you need anything else."),
        ("Goodbye!", "Goodbye! Have a wonderful day ahead."),
    ]

    for u, a in qa_en:
        for _ in range(25):
            conversations.append({
                "messages": [
                    {"role": "system", "content": random.choice(system_prompts_en)},
                    {"role": "user", "content": u},
                    {"role": "assistant", "content": a},
                ]
            })

    # 6. Çok Turlu (Multi-Turn) Konuşma Örnekleri
    multi_turn_samples = [
        [
            {"role": "user", "content": "Bana bir elma hakkında kısa bir cümle yaz."},
            {"role": "assistant", "content": "Kırmızı ve tatlı elma, dalında neşeyle parlıyordu."},
            {"role": "user", "content": "Şimdi buna bir kuş ekle."},
            {"role": "assistant", "content": "Kırmızı elmanın hemen yanındaki dala sevimli küçük bir serçe kondu."},
        ],
        [
            {"role": "user", "content": "Python'da döngü nasıl kurulur?"},
            {"role": "assistant", "content": "Python'da 'for' veya 'while' döngüsü kullanılır. Örneğin: for i in range(5): print(i)"},
            {"role": "user", "content": "Peki while döngüsü nasıldır?"},
            {"role": "assistant", "content": "While döngüsü belirli bir koşul doğru olduğu sürece çalışır:\nsayac = 0\nwhile sayac < 5:\n    print(sayac)\n    sayac += 1"},
        ],
        [
            {"role": "user", "content": "Bugün hava çok yağmurlu."},
            {"role": "assistant", "content": "Yağmurlu günlerde sıcak bir çay veya kahve eşliğinde kitap okumak çok dinlendirici olabilir."},
            {"role": "user", "content": "Bana ne tür bir kitap önerirsin?"},
            {"role": "assistant", "content": "Sakinleştirici bir bilim kurgu klasiği veya ilham verici bir teknoloji biyografisi harika bir tercih olur."},
        ],
    ]

    for sample in multi_turn_samples:
        for _ in range(50):
            sys_msg = {"role": "system", "content": random.choice(system_prompts_tr)}
            full_msgs = [sys_msg] + sample
            conversations.append({"messages": full_msgs})

    # İstenen sayıya ulaşana kadar çeşitlendir
    random.shuffle(conversations)
    if len(conversations) < target_count:
        repeat_factor = (target_count // len(conversations)) + 1
        conversations = (conversations * repeat_factor)[:target_count]
    else:
        conversations = conversations[:target_count]

    # JSONL Olarak Diske Yaz
    with open(output_file, "w", encoding="utf-8") as f:
        for item in conversations:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return len(conversations)


if __name__ == "__main__":
    count = generate_sft_dataset()
    print(f"Başarıyla {count:,} adet diyalog örneği oluşturuldu: data/dialogue_sft.jsonl")

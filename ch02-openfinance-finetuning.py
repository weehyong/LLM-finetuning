#!/usr/bin/env python3
"""
================================================================================

    BUILDING A FINANCIAL LLM: A COMPLETE TUTORIAL

    How Dragon LLM Turned Qwen 3 into an Open Finance Model
    — and how you can replicate it in under an hour

================================================================================

ABOUT THIS TUTORIAL
───────────────────
In November 2025, a French AI company called Dragon LLM released two open-source
8B-parameter language models fine-tuned for finance. They published their full
methodology in "The LLM Pro Finance Suite" (arXiv:2511.08621), describing how
they curated training data and adapted general-purpose models for financial tasks
like sentiment analysis, regulatory compliance, financial translation, and
retrieval-augmented generation (RAG).

This tutorial walks through their approach end-to-end. You will:

  Step 1 — Learn why Dragon LLM started from instruction-tuned models
  Step 2 — Load a base model with Unsloth for efficient training
  Step 3 — Build a training dataset mirroring Dragon LLM's 5-category mix
  Step 4 — Format the data for supervised fine-tuning
  Step 5 — Train the model
  Step 6 — Test the fine-tuned model across financial tasks
  Step 7 — Save and export the model

PREREQUISITES
─────────────
  Hardware:  NVIDIA GPU with CUDA support (tested on DGX Spark, also works
             on Colab T4/L4/A100, RTX 3090/4090, etc.)
  Software:  Python 3.10+, pip

  Install dependencies before running:

    pip install --upgrade unsloth unsloth_zoo
    pip install datasets trl transformers

ESTIMATED TIME
──────────────
  Qwen3-0.6B on DGX Spark:  ~15-25 minutes
  Qwen3-0.6B on Colab T4:   ~20-35 minutes
  Qwen3-8B on A100:          ~45-60 minutes

================================================================================
"""


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                                                                          ║
# ║   STEP 1: WHY THIS APPROACH WORKS                                     ║
# ║                                                                          ║
# ╚════════════════════════════════════════════════════════════════════════════╝
#
# Before we write any training code, it is worth understanding the two key
# decisions that define Dragon LLM's methodology. Getting these right matters
# more than any hyperparameter.
#
#
# DECISION 1: START FROM AN INSTRUCTION-TUNED MODEL
# ──────────────────────────────────────────────────
# Most financial LLM projects start from a "base" model — one that has only
# been pre-trained on raw text. It can predict the next token, but it cannot
# follow instructions, hold a conversation, or reason step by step.
#
# Dragon LLM made a different choice. They started from instruction-tuned
# models (Qwen 3 8B, Llama 3.1 8B Instruct). These models have already
# learned, through extensive post-training:
#
#   - How to follow complex instructions
#   - How to reason step by step
#   - Safety and toxicity guardrails
#   - Multi-turn conversation skills
#   - Tool use and structured output
#
# By starting here, they did not need to teach the model HOW to be helpful.
# They only needed to teach it WHAT finance is. This is far more efficient
# and avoids the catastrophic forgetting problems that plague base-model
# fine-tuning.
#
# In their own words (from the paper): they focused on "enhancing generalist
# instruction-tuned models, leveraging their existing strengths in instruction
# following, reasoning, and toxicity control."
#
#
# DECISION 2: BALANCE THE TRAINING DATA
# ──────────────────────────────────────
# The second insight is about data composition. Naively, you might think:
# "I want a financial model, so I should train only on financial data."
#
# This is a mistake. When you fine-tune a model exclusively on domain data,
# it "forgets" its general capabilities — a phenomenon called catastrophic
# forgetting. The model gets better at finance but worse at everything else,
# including basic reasoning and instruction following.
#
# Dragon LLM's solution was a carefully balanced dataset:
#
#   ┌──────────────┬────────┬─────────────────────────────────────────────┐
#   │ Category     │ Share  │ Purpose                                     │
#   ├──────────────┼────────┼─────────────────────────────────────────────┤
#   │ Financial    │  54%   │ Core domain knowledge — sentiment analysis, │
#   │              │        │ regulatory Q&A, financial terminology        │
#   ├──────────────┼────────┼─────────────────────────────────────────────┤
#   │ Translation  │  20%   │ EN/FR/DE financial document translation —   │
#   │              │        │ critical for European finance workflows      │
#   ├──────────────┼────────┼─────────────────────────────────────────────┤
#   │ General      │  16%   │ General instruction data — prevents the     │
#   │              │        │ model from forgetting how to be helpful      │
#   ├──────────────┼────────┼─────────────────────────────────────────────┤
#   │ RAG          │   8%   │ Document-grounded Q&A — teaches the model   │
#   │              │        │ to answer from provided context (retrieval)  │
#   ├──────────────┼────────┼─────────────────────────────────────────────┤
#   │ Math & Code  │   2%   │ Financial math + Python — preserves the     │
#   │              │        │ base model's quantitative reasoning          │
#   └──────────────┴────────┴─────────────────────────────────────────────┘
#
# Notice that only 54% is financial data. Nearly half the dataset exists
# purely to PRESERVE capabilities the base model already has. This is the
# price of avoiding catastrophic forgetting, and it is worth paying.
#
# Let's begin building.


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                                                                          ║
# ║   STEP 2: LOADING THE MODEL WITH UNSLOTH                              ║
# ║                                                                          ║
# ╚════════════════════════════════════════════════════════════════════════════╝
#
# We need three things to start training:
#   1. A pre-trained model (our starting point)
#   2. A tokenizer (converts text to numbers and back)
#   3. LoRA adapters (small trainable layers we attach to the frozen model)
#
# WHAT IS UNSLOTH?
# ────────────────
# Unsloth is a library that makes fine-tuning 2x faster and uses 70% less
# GPU memory compared to standard HuggingFace training. It achieves this
# through custom CUDA kernels, optimized memory management, and intelligent
# gradient checkpointing — all without sacrificing accuracy.
#
# For us, this means we can train on smaller (cheaper) GPUs and finish
# faster. On a DGX Spark, training that would take an hour with vanilla
# HuggingFace finishes in 15–25 minutes with Unsloth.
#
# WHAT IS LoRA?
# ─────────────
# Low-Rank Adaptation (LoRA) is a technique where, instead of updating all
# the model's billions of parameters, we freeze them and attach small
# trainable "adapter" matrices to specific layers. During training, only
# these adapters are updated.
#
# The benefits are significant:
#   - Memory: We only store gradients for ~1-2% of parameters
#   - Speed: Far fewer computations per training step
#   - Storage: The saved adapter is typically 50–200 MB, not 16 GB
#   - Composability: You can swap adapters for different tasks
#
# The key hyperparameter is `r` (rank), which controls the adapter size.
# Higher rank = more capacity = more memory. For our 0.6B model, rank 16
# is sufficient. Dragon LLM likely used rank 32–64 for their 8B models.

import os
import json
import random
import warnings
from dataclasses import dataclass

# bitsandbytes uses _check_is_size which is deprecated in torch nightly.
# It's harmless and will be fixed in a future bitsandbytes release.
warnings.filterwarnings("ignore", message="_check_is_size", category=FutureWarning)

# ---------------------------------------------------------------------------
# 2.1 Configuration
# ---------------------------------------------------------------------------
# We centralize every tunable parameter in a single Config class. This makes
# it easy to experiment: change one value here and the entire pipeline adapts.

@dataclass
class Config:
    # -- Model Selection --
    # We use Qwen3-0.6B, the smallest dense model in the Qwen 3 family.
    # Dragon LLM used Qwen3-8B. We use a smaller model for two reasons:
    #   1. Faster training (15 min vs. 60 min)
    #   2. Lower memory bandwidth demand on DGX Spark
    # Once your data pipeline is working, scale up by changing this one line.
    model_name: str = "unsloth/Qwen3-0.6B"

    # Maximum sequence length for training. Qwen 3 supports up to 40,960
    # tokens natively (128K with YaRN), but 2048 is enough for our examples
    # and keeps memory usage low.
    max_seq_length: int = 2048

    # 4-bit quantization reduces the model's memory footprint by ~4x.
    # The model weights are stored in NF4 (NormalFloat 4-bit) format, but
    # computations still happen in bf16 for accuracy. This is sometimes
    # called "QLoRA" — quantized LoRA.
    load_in_4bit: bool = True

    # -- LoRA Hyperparameters --
    # r (rank): Size of the adapter matrices. Higher = more capacity.
    #   Rule of thumb: r=16 for models ≤2B, r=32 for 4B–8B, r=64 for 70B+
    lora_r: int = 16
    # alpha: Scaling factor. Setting alpha = r is a common default that
    # means the adapter's contribution is neither amplified nor dampened.
    lora_alpha: int = 16
    # dropout: Regularization. 0.0 is standard for LoRA fine-tuning.
    lora_dropout: float = 0.0
    # target_modules: Which layers get adapters. We target all linear
    # layers in the transformer — this is what Unsloth recommends and what
    # gives the best results.
    target_modules: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj",   # attention layers
        "gate_proj", "up_proj", "down_proj",        # MLP layers
    )

    # -- Training Hyperparameters --
    num_train_epochs: int = 1           # 1 epoch for quick demo; use 2-3 for quality
    per_device_train_batch_size: int = 4  # Samples processed per step (per GPU)
    gradient_accumulation_steps: int = 4  # Accumulate 4 steps before updating
    # ^ Effective batch size = 4 × 4 = 16 samples per weight update

    learning_rate: float = 2e-4         # Standard for LoRA fine-tuning
    warmup_ratio: float = 0.05          # Warm up for first 5% of steps
    weight_decay: float = 0.01          # Light L2 regularization
    lr_scheduler_type: str = "cosine"   # Cosine annealing — smooth decay
    max_grad_norm: float = 1.0          # Gradient clipping for stability
    seed: int = 42
    bf16: bool = True                   # bfloat16 — supported on Blackwell/Ampere+
    fp16: bool = False

    # -- Dataset Configuration --
    # Dragon LLM used tens of thousands of samples. We use 2000 for speed.
    # The ratios below match their published data mix exactly.
    total_samples: int = 2000
    finance_ratio: float = 0.54         # 54% financial domain
    translation_ratio: float = 0.20     # 20% translation pairs
    general_ratio: float = 0.16         # 16% general instruction
    rag_ratio: float = 0.08             #  8% document-grounded Q&A
    math_code_ratio: float = 0.02       #  2% math and code

    # -- Output --
    output_dir: str = "./qwen-open-finance-output"

    @property
    def sample_counts(self):
        """Convert ratios to concrete sample counts."""
        return {
            "finance":     int(self.total_samples * self.finance_ratio),
            "translation": int(self.total_samples * self.translation_ratio),
            "general":     int(self.total_samples * self.general_ratio),
            "rag":         int(self.total_samples * self.rag_ratio),
            "math_code":   int(self.total_samples * self.math_code_ratio),
        }


config = Config()

print("Tutorial: Fine-Tuning Qwen 3 into a Financial LLM")
print("=" * 55)
print(f"\nModel:          {config.model_name}")
print(f"LoRA rank:      {config.lora_r}")
print(f"Sequence length: {config.max_seq_length}")
print(f"Total samples:  {config.total_samples}")
print(f"Sample counts:  {config.sample_counts}")


# ---------------------------------------------------------------------------
# 2.2 Load the pre-trained model
# ---------------------------------------------------------------------------
# FastModel.from_pretrained() downloads the model from HuggingFace (first
# run only) and loads it onto the GPU with 4-bit quantization. On the DGX
# Spark, this takes about 30 seconds for 0.6B. The model weights are frozen
# — we will not modify them directly.

print("\n\n>>> Step 2: Loading the model...")

from unsloth import FastModel
import torch

model, tokenizer = FastModel.from_pretrained(
    model_name=config.model_name,
    max_seq_length=config.max_seq_length,
    load_in_4bit=config.load_in_4bit,
    load_in_8bit=False,
    full_finetuning=False,           # We want LoRA, not full fine-tuning
)


# ---------------------------------------------------------------------------
# 2.3 Attach LoRA adapters
# ---------------------------------------------------------------------------
# This is where we make the model trainable. get_peft_model() inserts small
# adapter matrices into every layer listed in target_modules. After this
# call, only these adapters will receive gradient updates during training.
#
# use_gradient_checkpointing="unsloth" enables Unsloth's memory-optimized
# version of gradient checkpointing, which trades a small amount of compute
# time for a large reduction in peak memory usage.

model = FastModel.get_peft_model(
    model,
    r=config.lora_r,
    lora_alpha=config.lora_alpha,
    lora_dropout=config.lora_dropout,
    target_modules=list(config.target_modules),
    use_gradient_checkpointing="unsloth",
    random_state=config.seed,
)

# Let's see how many parameters are actually trainable:
print(f"\nModel loaded: {config.model_name}")
model.print_trainable_parameters()
# You should see something like: "trainable params: 5.5M || all params: 630M || 0.87%"
# This means we are only training ~1% of the model — the rest stays frozen.


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                                                                          ║
# ║   STEP 3: BUILDING THE TRAINING DATASET                               ║
# ║                                                                          ║
# ╚════════════════════════════════════════════════════════════════════════════╝
#
# This is the most important chapter. The quality of your fine-tuned model
# is determined almost entirely by the quality and composition of your data.
#
# Dragon LLM built their dataset through a sophisticated pipeline:
#
#   1. They trained a small classifier called "ClassiFin" (based on
#      DeBERTa) to identify financial content in web crawls.
#   2. They ran ClassiFin over the massive FineWeb and FineWeb-2 corpora
#      to extract millions of financial documents.
#   3. They supplemented this with Wikipedia articles from financial
#      categories (navigating the category graph from seeds like
#      "Corporate finance", "Investment", "Banking").
#   4. They filtered everything for quality using Qwen3-235B as a judge.
#   5. They added translation pairs from OPUS parallel corpora.
#   6. They mixed in general, RAG, and math/code data for balance.
#
# We cannot replicate their full pipeline here (it required a 235B model
# as a quality judge and access to massive web corpora). Instead, we use
# publicly available HuggingFace datasets that cover the same categories
# and task types. The proportions match their published ratios exactly.

print("\n\n>>> Step 3: Building the training dataset...")

from datasets import load_dataset, Dataset

counts = config.sample_counts


# ---------------------------------------------------------------------------
# Helper function: format data as chat conversations
# ---------------------------------------------------------------------------
# Every training example must be formatted as a multi-turn conversation in
# the model's expected chat template. Qwen 3 uses a system/user/assistant
# format. This function creates one such conversation.

def make_chat_message(system: str, user: str, assistant: str) -> dict:
    """
    Create a single training example in chat format.

    Args:
        system:    The system prompt (sets the model's persona/behavior)
        user:      The user's question or instruction
        assistant: The ideal response we want the model to learn

    Returns:
        dict with a "messages" key containing the conversation
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    messages.append({"role": "assistant", "content": assistant})
    return {"messages": messages}


# ═══════════════════════════════════════════════════════════════════════════
# 3.1  FINANCIAL DATA  (54% of the dataset)
# ═══════════════════════════════════════════════════════════════════════════
# This is the core of the dataset. Dragon LLM used financial content
# filtered from web crawls, Wikipedia, and curated Q&A pairs.
#
# We draw from three public datasets that cover the same task types:
#
#   • Financial PhraseBank — Sentiment classification of financial news
#     sentences. Dragon LLM evaluated their models on this exact benchmark.
#
#   • Sujet Finance Instruct — 177K high-quality financial instruction/
#     response pairs covering topics from investment analysis to
#     regulatory compliance.
#
#   • FiQA — Financial question answering from community forums.
#     Dragon LLM also benchmarked on FiQA in their paper.

print(f"\n  [1/5] Financial data ({counts['finance']} samples)...")

finance_samples = []

# --- Source A: Financial PhraseBank (sentiment analysis) ---
# This dataset contains ~4,800 English sentences from financial news,
# each labeled as positive, negative, or neutral by multiple annotators.
# We convert these into instruction-tuning format by asking the model
# to analyze sentiment and explain its reasoning.

try:
    # financial_phrasebank uses a loading script unsupported in datasets>=3.0.
    # Download the source zip from the Hub and parse the text file directly.
    import zipfile
    from huggingface_hub import hf_hub_download

    zip_path = hf_hub_download(
        repo_id="takala/financial_phrasebank",
        filename="data/FinancialPhraseBank-v1.0.zip",
        repo_type="dataset",
    )
    fpb = []
    with zipfile.ZipFile(zip_path) as z:
        with z.open("FinancialPhraseBank-v1.0/Sentences_AllAgree.txt") as f:
            for line in f.read().decode("latin-1").splitlines():
                line = line.strip()
                if "@" in line:
                    sentence, label = line.rsplit("@", 1)
                    fpb.append({"sentence": sentence.strip(), "label": label.strip()})

    for row in fpb[:min(len(fpb), counts["finance"] // 3)]:
        label = row["label"]
        finance_samples.append(make_chat_message(
            system=(
                "You are a financial analyst specializing in sentiment analysis "
                "of financial texts."
            ),
            user=(
                f"Analyze the sentiment of this financial statement and classify "
                f"it as positive, negative, or neutral. Explain your reasoning.\n\n"
                f"Statement: \"{row['sentence']}\""
            ),
            assistant=(
                f"Sentiment: **{label}**\n\n"
                f"This statement expresses a {label} sentiment. "
                + {
                    "positive": "The language suggests favorable financial performance or outlook.",
                    "negative": "The language suggests unfavorable financial performance or declining metrics.",
                    "neutral": "The language is factual and descriptive without expressing a clearly positive or negative outlook.",
                }[label]
            ),
        ))
    print(f"         Financial PhraseBank: {len(finance_samples)} samples loaded")
except Exception as e:
    print(f"         Financial PhraseBank: failed ({e})")


# --- Source B: Sujet Finance Instruct (financial Q&A) ---
# This is a large instruction-tuning dataset covering a wide range of
# financial topics: investment strategies, risk management, derivatives,
# corporate finance, accounting standards, and more. It provides the
# kind of broad financial knowledge Dragon LLM extracted from curated
# web content.

try:
    sujet = load_dataset("sujet-ai/Sujet-Finance-Instruct-177k", split="train")
    n_sujet = min(len(sujet), counts["finance"] // 3)
    sujet_subset = sujet.shuffle(seed=config.seed).select(range(n_sujet))

    for row in sujet_subset:
        instruction = row.get("instruction", row.get("input", ""))
        output = row.get("output", row.get("response", ""))
        if instruction and output:
            finance_samples.append(make_chat_message(
                system=(
                    "You are a knowledgeable financial advisor and analyst. "
                    "Provide accurate, detailed answers to financial questions."
                ),
                user=instruction,
                assistant=output,
            ))
    print(f"         Sujet Finance: {n_sujet} samples loaded")
except Exception as e:
    print(f"         Sujet Finance: failed ({e})")


# --- Source C: FiQA (financial opinion Q&A) ---
# FiQA-2018 contains questions and answers from financial forums and
# communities. These tend to be more conversational and opinion-oriented
# than the structured Sujet data, adding diversity to the training mix.

try:
    fiqa = load_dataset("pauri32/fiqa-2018", split="train")
    n_fiqa = min(len(fiqa), counts["finance"] - len(finance_samples))
    fiqa_subset = fiqa.shuffle(seed=config.seed).select(range(max(1, n_fiqa)))

    for row in fiqa_subset:
        question = row.get("question", row.get("query", ""))
        answer = row.get("answer", row.get("response", ""))
        if question and answer:
            finance_samples.append(make_chat_message(
                system="You are a financial expert. Answer financial questions accurately and concisely.",
                user=question,
                assistant=str(answer),
            ))
    print(f"         FiQA: {n_fiqa} samples loaded")
except Exception as e:
    print(f"         FiQA: failed ({e})")

finance_samples = finance_samples[:counts["finance"]]
print(f"         → Total financial: {len(finance_samples)} samples")


# ═══════════════════════════════════════════════════════════════════════════
# 3.2  TRANSLATION DATA  (20% of the dataset)
# ═══════════════════════════════════════════════════════════════════════════
# Dragon LLM placed unusual emphasis on translation. Their models support
# English, French, and German — the three most important languages for
# European financial markets. They used OPUS parallel corpora (large
# collections of professionally translated texts) to build translation
# pairs specifically from financial documents.
#
# Why does translation matter for a financial model? Because in European
# finance, professionals routinely work across languages:
#   - A KIID (Key Investor Information Document) must be translated into
#     every EU language where a fund is marketed
#   - Regulatory filings from the ECB are published in multiple languages
#   - Cross-border M&A deals involve documents in multiple languages
#
# We create representative financial translation pairs below. In a
# production system, you would source these from OPUS or professional
# translation memories.

print(f"\n  [2/5] Translation data ({counts['translation']} samples)...")

translation_samples = []

# Parallel financial sentences (English ↔ French)
# These represent the kind of content found in annual reports, regulatory
# decisions, fund documentation, and market analysis.

financial_sentences = [
    (
        "The company reported a 15% increase in quarterly revenue driven by strong demand in emerging markets.",
        "La société a enregistré une augmentation de 15 % de son chiffre d'affaires trimestriel grâce à une forte demande sur les marchés émergents.",
    ),
    (
        "Risk-adjusted returns on the fixed income portfolio exceeded the benchmark by 200 basis points.",
        "Les rendements ajustés au risque du portefeuille obligataire ont dépassé l'indice de référence de 200 points de base.",
    ),
    (
        "The central bank's monetary policy committee voted unanimously to maintain interest rates at current levels.",
        "Le comité de politique monétaire de la banque centrale a voté à l'unanimité le maintien des taux d'intérêt à leur niveau actuel.",
    ),
    (
        "Shareholders approved the proposed merger with a 78% majority vote at the annual general meeting.",
        "Les actionnaires ont approuvé la fusion proposée avec une majorité de 78 % lors de l'assemblée générale annuelle.",
    ),
    (
        "The fund's net asset value declined by 3.2% due to mark-to-market losses on sovereign bond holdings.",
        "La valeur liquidative du fonds a diminué de 3,2 % en raison de pertes de valorisation sur les obligations souveraines.",
    ),
    (
        "Operating expenses were reduced by 12% through operational efficiency improvements and headcount optimization.",
        "Les charges d'exploitation ont été réduites de 12 % grâce à l'amélioration de l'efficacité opérationnelle et à l'optimisation des effectifs.",
    ),
    (
        "The issuer's credit rating was downgraded from A+ to A- following the leveraged buyout announcement.",
        "La notation de crédit de l'émetteur a été abaissée de A+ à A- suite à l'annonce du rachat par emprunt.",
    ),
    (
        "Liquidity coverage ratio remained above regulatory minimums at 135% as of the reporting date.",
        "Le ratio de couverture des liquidités est resté au-dessus des minimums réglementaires à 135 % à la date de publication.",
    ),
    (
        "The PRIIP regulation requires key information documents to be provided to retail investors before purchase.",
        "Le règlement PRIIP exige que des documents d'informations clés soient fournis aux investisseurs particuliers avant l'achat.",
    ),
    (
        "Environmental, social, and governance factors are increasingly integrated into institutional investment decisions.",
        "Les facteurs environnementaux, sociaux et de gouvernance sont de plus en plus intégrés dans les décisions d'investissement institutionnel.",
    ),
    (
        "The swap spread widened significantly amid concerns over counterparty risk in the interbank market.",
        "Le spread de swap s'est considérablement élargi en raison des préoccupations liées au risque de contrepartie sur le marché interbancaire.",
    ),
    (
        "Year-over-year growth in assets under management reached EUR 2.3 billion across all fund strategies.",
        "La croissance annuelle des actifs sous gestion a atteint 2,3 milliards d'euros pour l'ensemble des stratégies de fonds.",
    ),
    (
        "The securitization vehicle issued three tranches of asset-backed securities rated AAA, AA, and BBB.",
        "Le véhicule de titrisation a émis trois tranches de titres adossés à des actifs notés AAA, AA et BBB.",
    ),
    (
        "Regulatory stress tests indicated that Tier 1 capital ratios would remain above 10% under adverse scenarios.",
        "Les tests de résistance réglementaires ont indiqué que les ratios de fonds propres de catégorie 1 resteraient supérieurs à 10 % dans les scénarios défavorables.",
    ),
    (
        "The company's free cash flow conversion rate improved to 92% from 85% in the prior fiscal year.",
        "Le taux de conversion du flux de trésorerie disponible de la société s'est amélioré à 92 % contre 85 % l'exercice précédent.",
    ),
]

# We create bidirectional translation pairs (EN→FR and FR→EN), because
# Dragon LLM evaluated on both directions.

for en, fr in financial_sentences:
    # English to French
    translation_samples.append(make_chat_message(
        system=(
            "You are a professional financial translator specializing in "
            "English to French translation of financial documents."
        ),
        user=f"Translate the following financial text from English to French:\n\n\"{en}\"",
        assistant=fr,
    ))
    # French to English
    translation_samples.append(make_chat_message(
        system=(
            "You are a professional financial translator specializing in "
            "French to English translation of financial documents."
        ),
        user=f"Traduisez le texte financier suivant du français vers l'anglais:\n\n\"{fr}\"",
        assistant=en,
    ))

# Financial terminology (EN/FR/DE trilingual)
# Dragon LLM also focused on domain-specific terminology accuracy.
financial_terms = [
    ("balance sheet", "bilan", "Bilanz"),
    ("income statement", "compte de résultat", "Gewinn- und Verlustrechnung"),
    ("cash flow", "flux de trésorerie", "Cashflow"),
    ("equity", "capitaux propres", "Eigenkapital"),
    ("dividend yield", "rendement du dividende", "Dividendenrendite"),
    ("risk assessment", "évaluation des risques", "Risikobewertung"),
    ("regulatory compliance", "conformité réglementaire", "Einhaltung gesetzlicher Vorschriften"),
    ("market capitalization", "capitalisation boursière", "Marktkapitalisierung"),
    ("asset management", "gestion d'actifs", "Vermögensverwaltung"),
    ("due diligence", "diligence raisonnable", "Sorgfaltspflicht"),
    ("hedge fund", "fonds spéculatif", "Hedgefonds"),
    ("initial public offering", "introduction en bourse", "Börsengang"),
    ("credit rating", "notation de crédit", "Kreditbewertung"),
    ("derivatives trading", "négoce de produits dérivés", "Derivatehandel"),
    ("portfolio diversification", "diversification du portefeuille", "Portfoliodiversifikation"),
]

for en, fr, de in financial_terms:
    translation_samples.append(make_chat_message(
        system="You are a financial terminology expert fluent in English, French, and German.",
        user=f"What is the French and German translation of the financial term '{en}'?",
        assistant=(
            f"The financial term '{en}' translates to:\n"
            f"- **French**: {fr}\n"
            f"- **German**: {de}\n\n"
            f"These terms are commonly used in financial reporting and regulatory "
            f"documents across European markets."
        ),
    ))

# Pad with resampled examples if we need more to hit the target count
random.seed(config.seed)
base_count = len(translation_samples)
while len(translation_samples) < counts["translation"]:
    translation_samples.append(random.choice(translation_samples[:base_count]))

translation_samples = translation_samples[:counts["translation"]]
print(f"         → Total translation: {len(translation_samples)} samples")


# ═══════════════════════════════════════════════════════════════════════════
# 3.3  GENERAL DOMAIN DATA  (16% of the dataset)
# ═══════════════════════════════════════════════════════════════════════════
# This is the "forgetting prevention" slice. Dragon LLM used FineWeb-Edu
# (educational web content) and general instruction data to ensure their
# models stayed good at everyday tasks: summarization, explanation,
# creative writing, general Q&A.
#
# We use the Alpaca dataset — a widely-used collection of ~52K general
# instruction/response pairs. It covers topics from science to history
# to cooking, providing broad diversity.

print(f"\n  [3/5] General domain data ({counts['general']} samples)...")

general_samples = []

try:
    alpaca = load_dataset("yahma/alpaca-cleaned", split="train")
    n_general = min(len(alpaca), counts["general"])
    alpaca_subset = alpaca.shuffle(seed=config.seed).select(range(n_general))

    for row in alpaca_subset:
        instruction = row["instruction"]
        inp = row.get("input", "")
        output = row["output"]
        user_msg = f"{instruction}\n\n{inp}".strip() if inp else instruction

        general_samples.append(make_chat_message(
            system="You are a helpful, accurate, and concise assistant.",
            user=user_msg,
            assistant=output,
        ))
    print(f"         Alpaca-cleaned: {len(general_samples)} samples loaded")
except Exception as e:
    print(f"         Alpaca-cleaned: failed ({e})")

general_samples = general_samples[:counts["general"]]
print(f"         → Total general: {len(general_samples)} samples")


# ═══════════════════════════════════════════════════════════════════════════
# 3.4  RAG DATA  (8% of the dataset)
# ═══════════════════════════════════════════════════════════════════════════
# RAG (Retrieval-Augmented Generation) is a pattern where you provide a
# document to the model and ask it to answer questions grounded in that
# document — rather than relying on its memorized knowledge.
#
# Dragon LLM explicitly stated their models are "RAG-ready", meaning they
# included substantial RAG training data. This teaches the model to:
#   1. Read a provided context carefully
#   2. Answer only from the given information
#   3. Cite or reference the document
#   4. Say "I don't know" when the answer isn't in the context
#
# In financial applications, RAG is critical. Analysts need models that
# can process a 50-page annual report and answer specific questions about
# it — without hallucinating numbers.
#
# We create representative examples using synthetic financial documents
# (annual reports, ECB decisions, KIIDs, risk assessments).

print(f"\n  [4/5] RAG data ({counts['rag']} samples)...")

rag_samples = []

rag_documents = [
    {
        "doc": (
            "Annual Report 2024 — TechCorp Inc.\n\n"
            "Revenue for fiscal year 2024 reached $12.4 billion, representing a "
            "23% year-over-year increase. Operating income was $3.1 billion with "
            "an operating margin of 25%. The company repurchased $2.0 billion in "
            "common stock during the year. Research and development expenses "
            "increased to $1.8 billion, reflecting continued investment in AI and "
            "cloud technologies. Total headcount grew to 45,000 employees across "
            "30 countries."
        ),
        "pairs": [
            (
                "What was TechCorp's revenue in fiscal year 2024?",
                "According to the Annual Report 2024, TechCorp Inc. reported "
                "revenue of $12.4 billion for fiscal year 2024, which represents "
                "a 23% year-over-year increase."
            ),
            (
                "What was the operating margin?",
                "Based on the document, TechCorp's operating margin was 25%, "
                "with operating income of $3.1 billion on revenue of $12.4 billion."
            ),
            (
                "How much did TechCorp spend on R&D?",
                "The Annual Report states that research and development expenses "
                "increased to $1.8 billion, driven by continued investment in AI "
                "and cloud technologies."
            ),
        ],
    },
    {
        "doc": (
            "ECB Monetary Policy Decision — March 2024\n\n"
            "The Governing Council of the European Central Bank decided to keep "
            "the three key ECB interest rates unchanged. The main refinancing rate "
            "stands at 4.50%, the marginal lending facility rate at 4.75%, and the "
            "deposit facility rate at 4.00%. Inflation has continued to decline but "
            "remains above the 2% target. The Governing Council emphasized that "
            "future policy decisions will be data-dependent and made on a "
            "meeting-by-meeting basis. Economic growth in the euro area remained "
            "subdued in Q4 2023."
        ),
        "pairs": [
            (
                "What are the current ECB interest rates?",
                "According to the ECB Monetary Policy Decision, the main "
                "refinancing rate is 4.50%, the marginal lending facility rate "
                "is 4.75%, and the deposit facility rate is 4.00%."
            ),
            (
                "What is the ECB's inflation target?",
                "The document indicates that while inflation has continued to "
                "decline, it remains above the ECB's 2% target."
            ),
        ],
    },
    {
        "doc": (
            "KIID — European Growth Equity Fund\n\n"
            "Objective: The fund aims to achieve long-term capital growth by "
            "investing primarily in European equities. Risk Profile: 5 out of 7 "
            "(medium-high risk). The fund invests at least 80% of its assets in "
            "shares of companies listed on European stock exchanges. The fund may "
            "use financial derivative instruments for hedging purposes. Ongoing "
            "charges: 1.45% per annum. Entry charge: up to 5%. Exit charge: none. "
            "Performance fee: 15% of outperformance above the STOXX Europe 600 index."
        ),
        "pairs": [
            (
                "What is the risk profile of this fund?",
                "According to the KIID document, the European Growth Equity Fund "
                "has a risk profile of 5 out of 7, classified as medium-high risk."
            ),
            (
                "What are the fund's charges?",
                "The KIID states the following charges: ongoing charges of 1.45% "
                "per annum, an entry charge of up to 5%, no exit charge, and a "
                "performance fee of 15% on outperformance above the STOXX Europe "
                "600 index."
            ),
            (
                "What does the fund invest in?",
                "Based on the KIID, the fund invests at least 80% of its assets "
                "in shares of companies listed on European stock exchanges, with "
                "the objective of achieving long-term capital growth. It may also "
                "use financial derivative instruments for hedging purposes."
            ),
        ],
    },
    {
        "doc": (
            "Risk Assessment Report — Global Credit Portfolio Q3 2024\n\n"
            "The portfolio's Value-at-Risk (VaR) at the 99% confidence level "
            "stands at EUR 45 million, a decrease of 8% from Q2. Expected "
            "Shortfall (CVaR) is EUR 62 million. Credit exposure to "
            "investment-grade issuers represents 72% of the portfolio, while "
            "high-yield exposure accounts for 28%. Geographic concentration: "
            "45% North America, 35% Europe, 15% Asia-Pacific, 5% Emerging "
            "Markets. The portfolio's weighted average credit rating is A-. "
            "Duration is 4.2 years. Stress test results show maximum portfolio "
            "loss of EUR 180 million under a severe recession scenario."
        ),
        "pairs": [
            (
                "What is the portfolio's VaR?",
                "According to the Risk Assessment Report, the portfolio's "
                "Value-at-Risk (VaR) at the 99% confidence level is EUR 45 million, "
                "representing an 8% decrease from Q2."
            ),
            (
                "What is the credit quality breakdown?",
                "The report indicates that investment-grade issuers represent 72% "
                "of the portfolio, while high-yield exposure accounts for 28%. "
                "The weighted average credit rating is A-."
            ),
        ],
    },
]

# Convert each document + Q&A pair into a RAG training example
for doc_group in rag_documents:
    for question, answer in doc_group["pairs"]:
        rag_samples.append(make_chat_message(
            system=(
                "You are a financial analyst assistant. Answer questions based "
                "solely on the provided document. If the information is not in "
                "the document, say so."
            ),
            user=(
                f"Based on the following document, answer the question.\n\n"
                f"**Document:**\n{doc_group['doc']}\n\n"
                f"**Question:** {question}"
            ),
            assistant=answer,
        ))

# Pad to target count
base_rag = len(rag_samples)
while len(rag_samples) < counts["rag"]:
    rag_samples.append(random.choice(rag_samples[:base_rag]))

rag_samples = rag_samples[:counts["rag"]]
print(f"         → Total RAG: {len(rag_samples)} samples")


# ═══════════════════════════════════════════════════════════════════════════
# 3.5  MATH & CODE DATA  (2% of the dataset)
# ═══════════════════════════════════════════════════════════════════════════
# The smallest slice, but important. Dragon LLM included math, reasoning,
# and coding data specifically to prevent the base model from losing its
# quantitative skills during fine-tuning.
#
# We focus on financial math (compound interest, bond pricing, portfolio
# returns) and Python code for quantitative finance — tasks that a
# financial LLM should excel at.

print(f"\n  [5/5] Math & code data ({counts['math_code']} samples)...")

math_code_samples = []

# --- Financial math problems with step-by-step solutions ---
math_problems = [
    (
        "Calculate the compound interest on a $10,000 investment at 5% annual rate "
        "compounded quarterly for 3 years.",
        "Using the compound interest formula: A = P(1 + r/n)^(nt)\n\n"
        "Where:\n"
        "- P = $10,000 (principal)\n"
        "- r = 0.05 (annual rate)\n"
        "- n = 4 (quarterly compounding)\n"
        "- t = 3 years\n\n"
        "A = 10000 × (1 + 0.05/4)^(4×3)\n"
        "A = 10000 × (1.0125)^12\n"
        "A = 10000 × 1.16075\n"
        "A = $11,607.55\n\n"
        "Compound interest earned = $11,607.55 - $10,000 = **$1,607.55**"
    ),
    (
        "A bond has a face value of $1,000, a coupon rate of 6%, and matures in "
        "5 years. If the yield to maturity is 4%, what is the bond's price?",
        "The bond price is the present value of all future cash flows:\n\n"
        "Annual coupon = $1,000 × 6% = $60\n\n"
        "Price = Σ [60/(1.04)^t] for t=1 to 5 + 1000/(1.04)^5\n\n"
        "= 57.69 + 55.47 + 53.34 + 51.29 + 49.31 + 821.93\n"
        "= **$1,089.04**\n\n"
        "The bond trades at a premium because the coupon rate (6%) exceeds the "
        "yield to maturity (4%)."
    ),
    (
        "A portfolio has the following weights and returns: Stock A (40%, 12%), "
        "Stock B (35%, 8%), Stock C (25%, -3%). Calculate the portfolio return.",
        "Portfolio return = Σ (weight × return)\n\n"
        "= (0.40 × 12%) + (0.35 × 8%) + (0.25 × -3%)\n"
        "= 4.80% + 2.80% + (-0.75%)\n"
        "= **6.85%**\n\n"
        "The portfolio achieved a positive return of 6.85%, driven primarily by "
        "Stock A's strong performance."
    ),
    (
        "If inflation is 3.2% and the nominal interest rate is 5.5%, what is the "
        "real interest rate using the Fisher equation?",
        "Using the Fisher equation:\n\n"
        "(1 + nominal) = (1 + real) × (1 + inflation)\n\n"
        "1.055 = (1 + real) × 1.032\n"
        "(1 + real) = 1.055 / 1.032 = 1.02229\n\n"
        "Real interest rate = **2.23%**\n\n"
        "Note: The simplified approximation (5.5% - 3.2% = 2.3%) gives a close "
        "but slightly less accurate result."
    ),
]

# --- Python code for quantitative finance ---
code_problems = [
    (
        "Write a Python function to calculate the Sharpe ratio of a portfolio.",
        '```python\n'
        'def sharpe_ratio(portfolio_returns: list, risk_free_rate: float) -> float:\n'
        '    """\n'
        '    Calculate the Sharpe Ratio of a portfolio.\n'
        '    \n'
        '    Args:\n'
        '        portfolio_returns: List of periodic returns\n'
        '        risk_free_rate: Risk-free rate for the same period\n'
        '    \n'
        '    Returns:\n'
        '        Sharpe ratio as a float\n'
        '    """\n'
        '    import numpy as np\n'
        '    \n'
        '    returns = np.array(portfolio_returns)\n'
        '    excess_returns = returns - risk_free_rate\n'
        '    \n'
        '    mean_excess = np.mean(excess_returns)\n'
        '    std_excess = np.std(excess_returns, ddof=1)\n'
        '    \n'
        '    if std_excess == 0:\n'
        '        return 0.0\n'
        '    \n'
        '    return mean_excess / std_excess\n'
        '\n'
        '# Example:\n'
        'returns = [0.05, 0.02, -0.01, 0.08, 0.03]\n'
        'print(f"Sharpe Ratio: {sharpe_ratio(returns, 0.01):.4f}")\n'
        '```'
    ),
    (
        "Write a Python function to calculate the moving average of a stock price series.",
        '```python\n'
        'def moving_average(prices: list, window: int) -> list:\n'
        '    """\n'
        '    Calculate the simple moving average of a price series.\n'
        '    \n'
        '    Args:\n'
        '        prices: List of stock prices\n'
        '        window: Number of periods for the moving average\n'
        '    \n'
        '    Returns:\n'
        '        List of moving average values\n'
        '    """\n'
        '    if len(prices) < window:\n'
        '        return []\n'
        '    \n'
        '    ma = []\n'
        '    for i in range(len(prices) - window + 1):\n'
        '        avg = sum(prices[i:i + window]) / window\n'
        '        ma.append(round(avg, 2))\n'
        '    \n'
        '    return ma\n'
        '\n'
        '# Example:\n'
        'prices = [100, 102, 101, 105, 107, 110, 108]\n'
        'print(f"3-day MA: {moving_average(prices, 3)}")\n'
        '```'
    ),
]

for q, a in math_problems:
    math_code_samples.append(make_chat_message(
        system="You are a financial mathematics expert. Show your work step by step.",
        user=q,
        assistant=a,
    ))

for q, a in code_problems:
    math_code_samples.append(make_chat_message(
        system="You are a Python developer specializing in quantitative finance. Write clean, well-documented code.",
        user=q,
        assistant=a,
    ))

# Pad to target count
base_mc = len(math_code_samples)
while len(math_code_samples) < counts["math_code"]:
    math_code_samples.append(random.choice(math_code_samples[:base_mc]))

math_code_samples = math_code_samples[:counts["math_code"]]
print(f"         → Total math/code: {len(math_code_samples)} samples")


# ═══════════════════════════════════════════════════════════════════════════
# 3.6  COMBINE AND SHUFFLE
# ═══════════════════════════════════════════════════════════════════════════
# We concatenate all five categories and shuffle. The shuffle is important:
# if the model sees all financial data first, then all translation data,
# it may forget earlier categories by the end of training. Shuffling
# ensures a balanced exposure throughout.

print(f"\n  Combining all data categories...")

all_samples = (
    finance_samples +
    translation_samples +
    general_samples +
    rag_samples +
    math_code_samples
)

random.seed(config.seed)
random.shuffle(all_samples)

dataset = Dataset.from_list(all_samples)

total = len(all_samples)
print(f"\n  Final dataset composition:")
print(f"  {'─' * 50}")
print(f"  {'Category':<15} {'Count':>6} {'Actual %':>9}  {'Target %':>9}")
print(f"  {'─' * 50}")
print(f"  {'Financial':<15} {len(finance_samples):>6} {len(finance_samples)/total*100:>8.1f}%  {config.finance_ratio*100:>8.1f}%")
print(f"  {'Translation':<15} {len(translation_samples):>6} {len(translation_samples)/total*100:>8.1f}%  {config.translation_ratio*100:>8.1f}%")
print(f"  {'General':<15} {len(general_samples):>6} {len(general_samples)/total*100:>8.1f}%  {config.general_ratio*100:>8.1f}%")
print(f"  {'RAG':<15} {len(rag_samples):>6} {len(rag_samples)/total*100:>8.1f}%  {config.rag_ratio*100:>8.1f}%")
print(f"  {'Math/Code':<15} {len(math_code_samples):>6} {len(math_code_samples)/total*100:>8.1f}%  {config.math_code_ratio*100:>8.1f}%")
print(f"  {'─' * 50}")
print(f"  {'TOTAL':<15} {total:>6}")


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                                                                          ║
# ║   STEP 4: FORMATTING DATA FOR TRAINING                                ║
# ║                                                                          ║
# ╚════════════════════════════════════════════════════════════════════════════╝
#
# Our dataset currently contains raw conversation dicts. Before training,
# we need to convert these into the exact text format that Qwen 3 expects.
#
# Each model family has its own chat template — special tokens that mark
# where system prompts, user messages, and assistant responses begin and
# end. Qwen 3's template looks roughly like:
#
#   <|im_start|>system
#   You are a financial analyst...<|im_end|>
#   <|im_start|>user
#   What is the P/E ratio?<|im_end|>
#   <|im_start|>assistant
#   The P/E ratio is...<|im_end|>
#
# The tokenizer's apply_chat_template() method handles this formatting
# automatically. We also set enable_thinking=False because Dragon LLM
# trained without the /think reasoning blocks — though the capability
# is preserved in the final model.

print("\n\n>>> Step 4: Formatting data for training...")

def apply_chat_template(example):
    """Convert a conversation dict into the model's formatted text."""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    return {"text": text}

dataset = dataset.map(apply_chat_template, num_proc=1)

# Let's inspect one formatted example to verify it looks right:
print(f"\n  Sample formatted example (first 400 chars):")
print(f"  {'─' * 60}")
print(f"  {dataset[0]['text'][:400]}")
print(f"  {'─' * 60}")


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                                                                          ║
# ║   STEP 5: TRAINING                                                    ║
# ║                                                                          ║
# ╚════════════════════════════════════════════════════════════════════════════╝
#
# We use HuggingFace's SFTTrainer (Supervised Fine-Tuning Trainer) from the
# TRL library, running on top of Unsloth's optimized backend.
#
# KEY TRAINING CONCEPTS
# ─────────────────────
# Effective batch size:
#   per_device_batch_size (4) × gradient_accumulation_steps (4) = 16
#   This means the model sees 16 examples before each weight update.
#   Larger effective batch sizes give more stable gradients but slower
#   iteration.
#
# Learning rate schedule:
#   We use cosine annealing with a brief warmup. The learning rate starts
#   at 0, rises to 2e-4 during the first 5% of steps, then smoothly
#   decays back toward 0 following a cosine curve. This is the most
#   common schedule for LoRA fine-tuning.
#
# Packing:
#   When packing=True, Unsloth concatenates multiple short examples into
#   a single sequence (up to max_seq_length). This eliminates padding
#   waste and can speed up training by 30-50% for datasets with many
#   short examples (like ours).
#
# bf16 vs fp16:
#   bfloat16 has the same range as float32 but lower precision. It is
#   more numerically stable than fp16 for training. The DGX Spark's
#   Blackwell GPU (and all Ampere+ GPUs) supports bf16 natively.

print("\n\n>>> Step 5: Training the model...")

from trl import SFTTrainer, SFTConfig

# --- Detect GPU capabilities ---
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    use_bf16 = torch.cuda.is_bf16_supported()
    use_fp16 = not use_bf16
    print(f"  GPU detected: {gpu_name}")
    print(f"  Memory: {gpu_mem_gb:.1f} GB")
    print(f"  Precision: {'bf16' if use_bf16 else 'fp16'}")
else:
    use_bf16 = False
    use_fp16 = False
    print("  WARNING: No GPU detected. Training will be extremely slow.")

# --- Configure the trainer ---
training_args = SFTConfig(
    output_dir=config.output_dir,

    # How long to train
    num_train_epochs=config.num_train_epochs,

    # Batch size and accumulation
    per_device_train_batch_size=config.per_device_train_batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,

    # Learning rate and schedule
    learning_rate=config.learning_rate,
    warmup_ratio=config.warmup_ratio,
    weight_decay=config.weight_decay,
    lr_scheduler_type=config.lr_scheduler_type,
    max_grad_norm=config.max_grad_norm,

    # Precision
    bf16=use_bf16,
    fp16=use_fp16,

    # Logging and checkpoints
    logging_steps=10,                # Print loss every 10 steps
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,              # Keep only last 2 checkpoints

    # Data handling
    max_seq_length=config.max_seq_length,
    dataset_text_field="text",
    packing=True,                    # Concatenate short examples for efficiency

    # Reproducibility
    seed=config.seed,
    report_to="none",                # Change to "wandb" for Weights & Biases logging
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
)

# --- Print a summary before we start ---
eff_batch = config.per_device_train_batch_size * config.gradient_accumulation_steps
est_steps = len(dataset) // eff_batch * config.num_train_epochs

print(f"\n  Training plan:")
print(f"  {'─' * 45}")
print(f"  {'Epochs:':<30} {config.num_train_epochs}")
print(f"  {'Batch size (per device):':<30} {config.per_device_train_batch_size}")
print(f"  {'Gradient accumulation:':<30} {config.gradient_accumulation_steps}")
print(f"  {'Effective batch size:':<30} {eff_batch}")
print(f"  {'Estimated steps:':<30} ~{est_steps}")
print(f"  {'Learning rate:':<30} {config.learning_rate}")
print(f"  {'LoRA rank:':<30} {config.lora_r}")
print(f"  {'Packing:':<30} enabled")
print(f"  {'─' * 45}")
print(f"\n  Starting training... (this will take 15-60 minutes)\n")

# --- Train! ---
train_result = trainer.train()

# --- Report results ---
print(f"\n  Training complete!")
print(f"  {'─' * 45}")
print(f"  {'Total steps:':<30} {train_result.global_step}")
print(f"  {'Final training loss:':<30} {train_result.training_loss:.4f}")
print(f"  {'Runtime:':<30} {train_result.metrics['train_runtime']:.0f} seconds")
print(f"  {'Samples/second:':<30} {train_result.metrics['train_samples_per_second']:.2f}")
print(f"  {'─' * 45}")

# A training loss around 1.0-2.0 is typical for a first fine-tuning run.
# Lower is generally better, but very low loss (<0.5) on limited data
# could indicate overfitting. With only 2000 samples, expect ~1.0-1.5.


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                                                                          ║
# ║   STEP 6: TESTING THE FINE-TUNED MODEL                                ║
# ║                                                                          ║
# ╚════════════════════════════════════════════════════════════════════════════╝
#
# Now let's see if the training actually worked. We test one prompt from
# each category in our training mix:
#   1. Financial sentiment analysis
#   2. Financial translation
#   3. RAG (document-grounded Q&A)
#   4. Financial math
#   5. General financial knowledge
#
# Note: with only 2000 training samples and a 0.6B model, do not expect
# GPT-4-level answers. The goal here is to verify that the model has
# learned the FORMAT and STYLE of financial responses. Quality improves
# dramatically when you scale up the data and model size.

print("\n\n>>> Step 6: Testing the fine-tuned model...")

def generate_response(prompt, system="You are a financial expert.", max_tokens=256):
    """
    Run inference on the fine-tuned model.

    This function:
      1. Formats the prompt using Qwen 3's chat template
      2. Tokenizes and sends it to the GPU
      3. Generates a response using sampling (temperature=0.7)
      4. Decodes and returns the text
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,   # Add the assistant turn start token
        return_tensors="pt",
        enable_thinking=False,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    # Decode only the NEW tokens (skip the input prompt)
    response = tokenizer.decode(
        outputs[0][inputs.shape[-1]:],
        skip_special_tokens=True,
    )
    return response


# --- Test 1: Financial sentiment analysis ---
print(f"\n  Test 1: Financial Sentiment Analysis")
print(f"  {'─' * 55}")
prompt = (
    "Analyze the sentiment: 'The company's quarterly earnings "
    "exceeded analyst expectations, driven by strong growth in "
    "cloud services.'"
)
print(f"  Q: {prompt}")
print(f"  A: {generate_response(prompt, 'You are a financial sentiment analyst.')}")

# --- Test 2: Financial translation ---
print(f"\n  Test 2: Financial Translation (EN → FR)")
print(f"  {'─' * 55}")
prompt = (
    "Translate to French: 'The portfolio's risk-adjusted return "
    "outperformed the benchmark index by 150 basis points.'"
)
print(f"  Q: {prompt}")
print(f"  A: {generate_response(prompt, 'You are a professional financial translator.')}")

# --- Test 3: RAG / Document-grounded Q&A ---
print(f"\n  Test 3: Document-Grounded Q&A (RAG)")
print(f"  {'─' * 55}")
prompt = (
    "Based on the following: 'Q3 revenue was €5.2B, up 18% YoY. "
    "EBITDA margin improved to 32%.' What was the EBITDA margin?"
)
print(f"  Q: {prompt}")
print(f"  A: {generate_response(prompt, 'You are a financial analyst. Answer based on the provided information only.')}")

# --- Test 4: Financial math ---
print(f"\n  Test 4: Financial Mathematics")
print(f"  {'─' * 55}")
prompt = "If a stock is trading at $50 with a P/E ratio of 25, what are the earnings per share?"
print(f"  Q: {prompt}")
print(f"  A: {generate_response(prompt, 'You are a financial mathematics expert.')}")

# --- Test 5: General financial knowledge ---
print(f"\n  Test 5: Financial Domain Knowledge")
print(f"  {'─' * 55}")
prompt = "What is the difference between a UCITS fund and an AIF under European regulations?"
print(f"  Q: {prompt}")
print(f"  A: {generate_response(prompt, 'You are a European financial regulatory expert.')}")


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                                                                          ║
# ║   STEP 7: SAVING AND EXPORTING THE MODEL                              ║
# ║                                                                          ║
# ╚════════════════════════════════════════════════════════════════════════════╝
#
# After training, we have a LoRA adapter — a small file (~50-200 MB) that
# modifies the behavior of the base model. We need to save it.
#
# There are several export options:
#
#   1. Save the LoRA adapter only (smallest, requires base model at load)
#   2. Merge adapter into base model and save as 16-bit (standard HF format)
#   3. Export as GGUF for use with llama.cpp / Ollama (local inference)
#   4. Push to HuggingFace Hub (share with the world)
#
# We demonstrate option 1 below. The others are shown as commented code
# you can uncomment when ready.

print("\n\n>>> Step 7: Saving the model...")

# --- Option 1: Save LoRA adapter (recommended for development) ---
# This saves only the trained adapter weights, not the full model.
# To use it later, you load the base model + adapter together.
model.save_pretrained(config.output_dir)
tokenizer.save_pretrained(config.output_dir)
print(f"  ✓ LoRA adapter saved to: {config.output_dir}/")
print(f"    To reload later:")
print(f"    >>> model, tokenizer = FastModel.from_pretrained(\"{config.output_dir}\")")

# --- Option 2: Save merged 16-bit model (full standalone model) ---
# Uncomment to merge the adapter into the base weights and save.
# The result is a standard HuggingFace model that doesn't need Unsloth.
#
# model.save_pretrained_merged(
#     f"{config.output_dir}-merged",
#     tokenizer,
#     save_method="merged_16bit",
# )
# print(f"  ✓ Merged 16-bit model saved to: {config.output_dir}-merged/")

# --- Option 3: Export as GGUF for llama.cpp / Ollama ---
# GGUF is the format used by llama.cpp and Ollama for local inference.
# q4_k_m is a good balance of quality vs. size. On the DGX Spark,
# the model will run at high speed using the FP4 tensor cores.
#
# model.save_pretrained_gguf(
#     f"{config.output_dir}-gguf",
#     tokenizer,
#     quantization_method="q4_k_m",
# )
# print(f"  ✓ GGUF model saved to: {config.output_dir}-gguf/")
# print(f"    Run with: ollama create my-finance-model -f Modelfile")

# --- Option 4: Push to HuggingFace Hub ---
# Share your model with the community. You'll need a HuggingFace token.
#
# model.push_to_hub("your-username/qwen-open-finance", tokenizer, token="hf_...")
# print(f"  ✓ Model pushed to HuggingFace Hub")


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                                                                          ║
# ║   SUMMARY AND NEXT STEPS                                                 ║
# ║                                                                          ║
# ╚════════════════════════════════════════════════════════════════════════════╝

print(f"\n\n{'═' * 60}")
print(f"  TUTORIAL COMPLETE")
print(f"{'═' * 60}")
print(f"""
  What you built:
  ───────────────
  A financial domain LLM fine-tuned from Qwen 3, following the
  same methodology Dragon LLM used to create their Open Finance
  models. Your model has been trained on a balanced mix of:

    • Financial sentiment, Q&A, and domain knowledge   (54%)
    • Financial document translation (EN/FR/DE)         (20%)
    • General instruction data (catastrophe prevention) (16%)
    • Document-grounded RAG examples                    ( 8%)
    • Financial math and Python code                    ( 2%)

  What to do next:
  ────────────────
  1. SCALE THE DATA
     Increase total_samples from 2,000 to 50,000+. This single
     change will have the biggest impact on quality.

  2. SCALE THE MODEL
     Change model_name to "unsloth/Qwen3-8B" (what Dragon LLM
     used). This fits comfortably on the DGX Spark.

  3. ADD REAL TRANSLATION DATA
     Replace our synthetic pairs with OPUS parallel corpora for
     authentic financial document translations.

  4. USE CLASSIFIN FOR DATA FILTERING
     Dragon LLM released their ClassiFin classifier on HuggingFace.
     Use it to extract high-quality financial content from FineWeb.

  5. EVALUATE ON BENCHMARKS
     Run the model on FPB, FiQA, FinQA, and ConvFinQA to measure
     improvement quantitatively. Dragon LLM's evaluation cookbook
     is available at: github.com/Dragon-LLM/llm-open-finance-cookbook

  6. EXPORT FOR PRODUCTION
     Uncomment the GGUF export above and serve with Ollama or
     llama.cpp for fast local inference on the DGX Spark.

  References:
  ───────────
  Paper:       arxiv.org/abs/2511.08621
  Models:      huggingface.co/collections/DragonLLM/llm-open-finance
  ClassiFin:   huggingface.co/DragonLLM/ClassiFin
  Cookbook:     github.com/Dragon-LLM/llm-open-finance-cookbook
  Unsloth:     unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune
""")

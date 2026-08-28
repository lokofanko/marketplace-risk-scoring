"""Generate synthetic listing creation events for risk scoring."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42
DEFAULT_ROWS = 5_000
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "listing_events.csv"

CATEGORY_MEDIANS = {
    "electronics": 900.0,
    "vehicles": 8500.0,
    "real_estate": 75000.0,
    "furniture": 450.0,
    "fashion": 120.0,
    "home_appliances": 650.0,
}

LOCATIONS = ["Moscow", "Saint Petersburg", "Kazan", "Novosibirsk", "Sochi"]

LEGIT_TITLES = {
    "electronics": ["Used iPhone 13 in good condition", "Gaming laptop with warranty"],
    "vehicles": ["Toyota Corolla 2016 clean documents", "Used city bike"],
    "real_estate": ["One bedroom apartment near metro", "Studio for long term rent"],
    "furniture": ["Wooden desk in good condition", "Dining table with chairs"],
    "fashion": ["Winter jacket lightly used", "Leather shoes size 42"],
    "home_appliances": ["Washing machine works well", "Microwave oven"],
}

RISKY_TITLES = {
    "electronics": [
        "iPhone 15 Pro very cheap urgent",
        "MacBook sealed prepayment only",
    ],
    "vehicles": ["Car urgent sale no questions", "Scooter cheap today only"],
    "real_estate": [
        "Apartment below market prepayment",
        "Rent urgent contact Telegram",
    ],
    "furniture": ["Designer sofa very cheap urgent", "New table prepay reserve"],
    "fashion": ["Luxury jacket original cheap", "Sneakers limited urgent"],
    "home_appliances": ["New fridge half price urgent", "Washer cheap Telegram"],
}

LEGIT_DESCRIPTIONS = [
    "Personal item, can meet in person and answer questions.",
    "Used carefully, normal sale, local pickup is possible.",
    "Available for inspection before purchase.",
]

RISKY_DESCRIPTIONS = [
    "Prepayment only, contact me in Telegram.",
    "Urgent sale today, write outside the platform.",
    "Delivery after deposit, no calls, messenger only.",
]

URGENCY_WORDS = ("urgent", "today", "срочно", "only")
EXTERNAL_CONTACT_WORDS = ("telegram", "messenger", "outside the platform", "no calls")
SUSPICIOUS_DESCRIPTION_WORDS = ("prepayment", "deposit", "messenger only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def contains_any(text: str, words: tuple[str, ...]) -> bool:
    text_lower = text.lower()
    return any(word in text_lower for word in words)


def build_listing(index: int, rng: np.random.Generator) -> dict[str, object]:
    category = rng.choice(list(CATEGORY_MEDIANS))
    location = rng.choice(LOCATIONS)
    is_scam_like_profile = rng.random() < 0.25

    if is_scam_like_profile:
        account_age_days = int(rng.gamma(2.0, 20.0))
        num_ads_last_24h = int(rng.poisson(3.5))
        num_ads_last_7d = int(num_ads_last_24h + rng.poisson(8))
        is_verified_user = bool(rng.random() < 0.30)
        previous_rejected_ads_count = int(rng.poisson(0.9))
        num_images = int(rng.integers(0, 5))
        price_ratio = float(np.clip(rng.normal(0.68, 0.25), 0.12, 1.5))
    else:
        account_age_days = int(np.clip(rng.gamma(2.5, 180.0), 1, 1800))
        num_ads_last_24h = int(rng.poisson(1.2))
        num_ads_last_7d = int(num_ads_last_24h + rng.poisson(4))
        is_verified_user = bool(rng.random() < 0.70)
        previous_rejected_ads_count = int(rng.poisson(0.15))
        num_images = int(rng.integers(1, 7))
        price_ratio = float(np.clip(rng.normal(1.0, 0.22), 0.25, 1.7))

    # Text signals overlap on purpose: legitimate sellers can also write "urgent".
    uses_risky_title = rng.random() < (0.62 if is_scam_like_profile else 0.14)
    uses_risky_description = rng.random() < (0.58 if is_scam_like_profile else 0.10)
    title_pool = RISKY_TITLES if uses_risky_title else LEGIT_TITLES
    description_pool = (
        RISKY_DESCRIPTIONS if uses_risky_description else LEGIT_DESCRIPTIONS
    )
    title = str(rng.choice(title_pool[category]))
    description = str(rng.choice(description_pool))

    median_price = CATEGORY_MEDIANS[category]
    price = round(median_price * price_ratio * float(rng.uniform(0.92, 1.08)), 2)

    has_telegram = contains_any(f"{title} {description}", ("telegram",))
    has_urgency_word = contains_any(f"{title} {description}", URGENCY_WORDS)
    has_external_contact = contains_any(
        f"{title} {description}", EXTERNAL_CONTACT_WORDS
    )
    has_suspicious_description = contains_any(description, SUSPICIOUS_DESCRIPTION_WORDS)

    # A noisy probability is closer to real moderation labels than a hard rule.
    risk_logit = -3.2
    risk_logit += 0.9 if account_age_days < 14 else 0.0
    risk_logit += 0.65 if num_ads_last_24h >= 4 else 0.0
    risk_logit += 0.45 if num_ads_last_7d >= 12 else 0.0
    risk_logit += 0.35 if not is_verified_user else 0.0
    risk_logit += min(previous_rejected_ads_count, 3) * 0.55
    risk_logit += 0.9 if price_ratio < 0.55 else 0.0
    risk_logit += 0.65 if has_telegram else 0.0
    risk_logit += 0.35 if has_urgency_word else 0.0
    risk_logit += 0.55 if has_external_contact else 0.0
    risk_logit += 0.35 if has_suspicious_description else 0.0
    risk_logit += 0.4 if num_images <= 1 else 0.0

    # Interactions and unobserved noise make a nonlinear model worth comparing.
    risk_logit += 0.8 if account_age_days < 14 and price_ratio < 0.55 else 0.0
    risk_logit += 0.7 if num_ads_last_24h >= 4 and has_external_contact else 0.0
    risk_logit += float(rng.normal(0.0, 0.75))
    risk_probability = 1.0 / (1.0 + np.exp(-risk_logit))

    return {
        "listing_id": f"listing_{index:06d}",
        "user_id": f"user_{int(rng.integers(1, 1800)):05d}",
        "title": title,
        "description": description,
        "price": price,
        "category": category,
        "location": location,
        "account_age_days": account_age_days,
        "num_ads_last_24h": num_ads_last_24h,
        "num_ads_last_7d": num_ads_last_7d,
        "is_verified_user": is_verified_user,
        "previous_rejected_ads_count": previous_rejected_ads_count,
        "num_images": num_images,
        "has_telegram": has_telegram,
        "has_urgency_word": has_urgency_word,
        "has_external_contact": has_external_contact,
        "price_to_category_median_ratio": round(price / median_price, 4),
        "label": int(rng.random() < risk_probability),
    }


def generate_dataset(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = [build_listing(index, rng) for index in range(1, rows + 1)]
    return pd.DataFrame.from_records(records)


def main() -> None:
    args = parse_args()
    df = generate_dataset(rows=args.rows, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"Dataset saved to: {args.output}")
    print(f"Dataset shape: {df.shape}")
    print(f"Label rate: {df['label'].mean():.4f}")


if __name__ == "__main__":
    main()

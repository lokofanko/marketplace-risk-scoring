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
    "electronics": ["iPhone 15 Pro very cheap urgent", "MacBook sealed prepayment only"],
    "vehicles": ["Car urgent sale no questions", "Scooter cheap today only"],
    "real_estate": ["Apartment below market prepayment", "Rent urgent contact Telegram"],
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
    is_scam_like_profile = rng.random() < 0.28

    if is_scam_like_profile:
        account_age_days = int(rng.integers(0, 21))
        num_ads_last_24h = int(rng.poisson(5))
        num_ads_last_7d = int(num_ads_last_24h + rng.poisson(12))
        is_verified_user = bool(rng.random() < 0.18)
        previous_rejected_ads_count = int(rng.poisson(1.4))
        num_images = int(rng.choice([0, 1, 1, 2]))
        price_ratio = float(rng.uniform(0.18, 0.75))
        title = str(rng.choice(RISKY_TITLES[category]))
        description = str(rng.choice(RISKY_DESCRIPTIONS))
    else:
        account_age_days = int(rng.integers(30, 1500))
        num_ads_last_24h = int(rng.poisson(1))
        num_ads_last_7d = int(num_ads_last_24h + rng.poisson(3))
        is_verified_user = bool(rng.random() < 0.78)
        previous_rejected_ads_count = int(rng.choice([0, 0, 0, 1]))
        num_images = int(rng.choice([2, 3, 4, 5, 6]))
        price_ratio = float(rng.uniform(0.75, 1.35))
        title = str(rng.choice(LEGIT_TITLES[category]))
        description = str(rng.choice(LEGIT_DESCRIPTIONS))

    # Add some overlap so the baseline is not perfectly trivial.
    if rng.random() < 0.08:
        description = str(rng.choice(RISKY_DESCRIPTIONS))
    if rng.random() < 0.06:
        title = str(rng.choice(RISKY_TITLES[category]))
    if rng.random() < 0.05:
        price_ratio = float(rng.uniform(0.35, 0.7))

    median_price = CATEGORY_MEDIANS[category]
    price = round(median_price * price_ratio * float(rng.uniform(0.92, 1.08)), 2)

    has_telegram = contains_any(f"{title} {description}", ("telegram",))
    has_urgency_word = contains_any(f"{title} {description}", URGENCY_WORDS)
    has_external_contact = contains_any(f"{title} {description}", EXTERNAL_CONTACT_WORDS)
    has_suspicious_description = contains_any(description, SUSPICIOUS_DESCRIPTION_WORDS)

    risk_points = 0.0
    risk_points += 0.18 if account_age_days < 7 else 0.0
    risk_points += 0.12 if num_ads_last_24h >= 5 else 0.0
    risk_points += 0.10 if num_ads_last_7d >= 15 else 0.0
    risk_points += 0.10 if not is_verified_user else 0.0
    risk_points += min(previous_rejected_ads_count, 3) * 0.07
    risk_points += 0.18 if price_ratio < 0.55 else 0.0
    risk_points += 0.08 if price_ratio < 0.35 else 0.0
    risk_points += 0.14 if has_telegram else 0.0
    risk_points += 0.08 if has_urgency_word else 0.0
    risk_points += 0.12 if has_external_contact else 0.0
    risk_points += 0.10 if has_suspicious_description else 0.0
    risk_points += 0.08 if num_images <= 1 else 0.0

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
        "label": int(risk_points >= 0.55),
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

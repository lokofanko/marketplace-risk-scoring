# Backend

This package owns the mini classified marketplace flow.

## Imported Scaffold

The initial Django marketplace scaffold was imported from:

https://github.com/mahmud-sajib/Classified-Ads-Marketplace

Imported upstream commit:

```text
bcae64b4a995d9e4c623e1f58014155bd81e8343
```

The imported scaffold includes Django apps, templates, static assets, and `manage.py`.
The upstream `requirements.txt` is kept as `requirements.upstream.txt` for reference.

## Future Stack

- Django
- Django REST Framework
- PostgreSQL
- uv-managed Python environment

## Planned Responsibilities

- User/account workflows
- Listing creation and lifecycle
- Calling the ML risk scoring service before publishing listings
- Storing risk predictions and decisions
- Moderator review queue
- Moderator feedback as labels for future retraining

## Current Notes

The upstream project is an older Django scaffold. Dependencies have not been migrated into
`pyproject.toml` yet because they need a compatibility pass before becoming part of the
main backend environment.

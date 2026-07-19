# gobekli-tepe-data-platform

> A modular **ELT data platform on Google Cloud** built around **dbt** and a
> **medallion (bronze → silver → gold)** architecture.

## Overview

Göbekli Tepe — named after the oldest known megalithic sanctuary, the metaphorical
"foundation" — is a data platform that turns raw source data into analytics-ready models on
GCP. Raw data is ingested and landed, loaded into a **bronze** layer, then progressively
refined into **silver** and **gold** layers with **dbt** running on **BigQuery**. All cloud
infrastructure is provisioned by a reusable **Terraform module**, and two lightweight
services move data into the platform: an **ingestion service** and a **data load service**.

## Architecture

```
external sources
   │  (ingestion service: extract)
   ▼
Landing zone (GCS)
   │  (data load service: load)
   ▼
Bronze  ──dbt──▶  Silver  ──dbt──▶  Gold  ──▶  BI / analytics (BigQuery)
```

- **Bronze** – raw, append-only mirror of source data.
- **Silver** – cleaned, conformed, deduplicated entities.
- **Gold** – business-level marts and aggregates for reporting.

## Components

- **`terraform/` module** – provisions BigQuery datasets (one per medallion layer), GCS
  landing buckets, service accounts with least-privilege IAM, Cloud Run services, and Cloud
  Scheduler triggers. Parameterized through variables so it is reusable across environments
  (dev/prod).
- **Ingestion service** – pulls data from external APIs and files and writes raw extracts to
  the GCS landing zone, with retries and incremental extraction.
- **Data load service** – loads landed files into the bronze BigQuery datasets (load jobs /
  external tables) while tracking load state.
- **dbt project** – staging models (bronze → silver) and mart models (silver → gold), with
  schema/data tests and generated documentation.

## Technology stack

| Layer | Technology |
|------|-----------|
| Transformations | dbt (dbt-bigquery) |
| Warehouse | BigQuery |
| Landing storage | Google Cloud Storage |
| Services | Python (ingestion + data load) on Cloud Run |
| Infrastructure | Terraform |
| Scheduling | Cloud Scheduler |
| CI/CD | GitHub Actions |

## Key features

- Clean medallion separation with dbt-enforced transformations and tests.
- Reusable, parameterized Terraform module for repeatable environments.
- Decoupled ingestion vs. load services that scale independently.
- Incremental models to control BigQuery cost.

## Snapshot (Q1 2025)

- Primary languages: Python, SQL (dbt), HCL (Terraform)
- Status: active
- Popularity at snapshot: ~18 stars, 3 forks
- Last significant update at snapshot: February 2025

## Notes

Part of a portfolio whose repositories are named after ancient archaeological sites (see also
**machu-pichu** and **carthage-architecture-center**). This naming convention and the design
rationale live only in this knowledge base, not in the repository's GitHub metadata.

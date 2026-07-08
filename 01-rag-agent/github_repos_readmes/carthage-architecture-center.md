# carthage-architecture-center

> A hub of **Terraform** modules and reference architectures for learning and standardizing
> **infrastructure on Google Cloud**.

## Overview

Carthage — the great architectural capital of the ancient Mediterranean — is an
"architecture center": a personal collection of reusable **Terraform** modules and reference
designs for building infrastructure on GCP. It exists primarily as a learning ground for
infrastructure-as-code patterns and as a library of building blocks reused across other
projects.

## Contents

- **Networking** – VPCs, subnets, firewall rules, Cloud NAT, private service connectivity.
- **Compute** – GKE clusters, Cloud Run services, instance templates.
- **Data & storage** – GCS buckets, BigQuery datasets, Cloud SQL.
- **Security & identity** – IAM roles/bindings, service accounts, Workload Identity
  Federation.
- **Reference architectures** – landing-zone / project-factory style examples that compose
  the modules into complete environments.

## Approach

Each module is written to be composable and environment-agnostic (configured through
variables), with an emphasis on least-privilege IAM and reproducible, version-controlled
infrastructure. The repository doubles as documentation of GCP architecture best practices.
While the reference architectures themselves are expressed in Terraform, the repository's
module generators, validation scripts, and helper tooling are written primarily in
**Python**, which makes up the majority of the codebase.

## Technology stack

| Concern | Technology |
|--------|-----------|
| Infrastructure as Code | Terraform (HCL) |
| Cloud | Google Cloud Platform |
| Auth | Workload Identity Federation |

## Snapshot (Q1 2025)

- Primary language: Python
- Status: actively curated
- Popularity at snapshot: ~11 stars, 2 forks
- Last significant update at snapshot: January 2025

## Notes

Shares the ancient-sites naming theme with **gobekli-tepe** and **machu-pichu**. Serves as
the infrastructure "reference library" that the other GCP projects borrow patterns from — a
relationship documented here in the knowledge base rather than in GitHub metadata.

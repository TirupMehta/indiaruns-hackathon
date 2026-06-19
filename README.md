# Intelligent Candidate Discovery & Ranking Engine

A production-grade, fully-offline Candidate Discovery and Ranking system built for the **Redrob Intelligent Candidate Discovery & Ranking Challenge**. This is not a keyword-matching filter — it's a **dual-layer composite scoring engine** that combines dense semantic retrieval with a 16-factor career intelligence multiplier to surface the best candidates from a pool of **100,000 profiles** in under 2 minutes on CPU.

---

## Quick Start

```bash
# 1. Install dependencies
pip install torch transformers numpy python-docx

# 2. Run the ranker (fully offline — no network, no GPU)
python rank.py \
  --candidates ./data-and-ai-challange/India_runs_data_and_ai_challenge/candidates.jsonl \
  --out ./submission.csv

# 3. Validate submission format
python ./data-and-ai-challange/India_runs_data_and_ai_challenge/validate_submission.py ./submission.csv
```

**Runtime:** ~113 seconds on a single-thread CPU (well within the 5-minute budget).
**Output:** `submission.csv` with exactly 100 ranked candidates.

---

## System Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                    100,000 CANDIDATE PROFILES                      │
│                      (candidates.jsonl)                            │
└──────────────────────────┬─────────────────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  HONEYPOT   │  O(1) per candidate
                    │  PRUNING    │  - Expert skills w/ 0 duration
                    │  LAYER      │  - Calendar-inconsistent dates
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  METADATA   │  Hard JD constraints
                    │  PRE-FILTER │  100K → ~6,600 candidates
                    │  LAYER      │  (3.2 seconds)
                    └──────┬──────┘
                           │
              ┌────────────▼────────────┐
              │   DENSE SEMANTIC LAYER  │  all-MiniLM-L6-v2
              │   384-dim embeddings    │  Cosine similarity
              │   Batch inference       │  (~108 seconds)
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   16-FACTOR MULTIPLIER  │  Career intelligence
              │   ENGINE                │  Behavioral signals
              │                         │  Platform activity
              └────────────┬────────────┘
                           │
                    ┌──────▼──────┐
                    │  COMPOSITE  │  score = semantic × multiplier
                    │  BLENDING   │  Sort desc, tie-break by ID
                    └──────┬──────┘
                           │
              ┌────────────▼────────────┐
              │   REASONING GENERATION  │  Fact-grounded, unique
              │   ENGINE                │  per candidate
              └────────────┬────────────┘
                           │
                    ┌──────▼──────┐
                    │ submission  │  100 rows
                    │ .csv        │  Validated format
                    └─────────────┘
```

---

## Design Decisions & Implementation Details

### 1. Honeypot Detection (Security Layer)

Organizers embedded subtly impossible "honeypot" profiles in the dataset. Any submission with >10% honeypots in the top 100 is **disqualified**. Two O(1) detection rules are applied to every candidate before any scoring:

| Rule | Detection Logic | Threshold |
|------|----------------|-----------|
| **Skill Duration Anomaly** | `expert`/`advanced` proficiency with `duration_months == 0` | ≥3 such skills |
| **Calendar Inconsistency** | `duration_months` exceeds calendar time between `start_date` and `end_date` | >6 months discrepancy |

**Result:** 0 honeypots in the final top 100.

### 2. CPU Performance Optimization (Pre-Filtering)

Dense retrieval on 100K candidates with CPU-only inference would take >30 minutes. The solution:

- **If pool > 150 candidates**, apply metadata pre-filters derived from the JD:
  - YoE: 4–15 years
  - Title: must contain engineering/technical keywords
  - Company history: cannot be 100% consulting/service firms
  - Skills: must have ≥1 AI/ML/Search keyword anywhere in profile text
  - Location: India-based or willing to relocate
  - Notice period: ≤90 days (standard for Indian product companies)
- **If pool ≤ 150** (sandbox/test), bypass all pre-filters

**Result:** 100,000 → 6,634 candidates in 3.2 seconds. Transformer inference runs only on this reduced pool.

### 3. Dense Semantic Layer

- **Model:** `all-MiniLM-L6-v2` (384 dimensions, 22M parameters) — loaded offline from `./model_weights/`
- **Candidate text construction:** Concatenates headline, summary, current role, skills list, and full career history into a single string per candidate
- **Batch inference:** Processes candidates in batches of 128 with padding/truncation at 256 tokens
- **Scoring:** L2-normalized embeddings → dot product = cosine similarity → normalized to [0, 1]

### 4. The 16-Factor Multiplier Engine

This is the core differentiator. Rather than relying solely on semantic similarity, the engine applies a **multiplicative career intelligence layer** that captures signals invisible to embeddings:

| # | Signal | Bonus/Penalty | Rationale |
|---|--------|--------------|-----------|
| 1 | **YoE fit** (6–8y sweet spot) | 1.2× / 0.7× | JD specifies 5–9y senior band |
| 2 | **Total ML experience** (≥4y) | 1.15× / 1.05× | Accumulated ML depth, not just total YoE |
| 3 | **Vector search + embeddings + eval** | up to 1.3× | JD's three core technical must-haves |
| 4 | **LLM fine-tuning experience** | 1.05× | Bonus for LoRA/QLoRA/PEFT experience |
| 5 | **Learning-to-rank experience** | 1.05× | XGBoost LTR, neural ranking expertise |
| 6 | **Distributed systems / inference** | 1.05× | Production-scale deployment capability |
| 7 | **Tenure stability** (avg <18mo = chaser) | 0.8× | JD explicitly disqualifies title chasers |
| 8 | **Promotion detection** (same company) | 1.2× | Internal growth signals strong performance |
| 9 | **Service-only career** (TCS, Infosys…) | 0.6× | JD explicitly disqualifies |
| 10 | **Product company experience** | 1.15× | Google, Flipkart, Swiggy, etc. |
| 11 | **Startup experience** (11–500 employees) | 1.1× | Founding-team fit indicator |
| 12 | **AI/ML title keywords** | 1.2× | Current role directly in ML/AI domain |
| 13 | **Notice period** (≤30d to >90d) | 1.15× → 0.6× | Smooth curve from available to risky |
| 14 | **Location fit** (Pune/Noida direct) | 1.2× / 1.1× / 0.5× | JD specifies Pune/Noida hybrid |
| 15 | **Open-to-work flag** | 1.15× | Active job seeker = faster conversion |
| 16 | **Recruiter response rate** (≥80%) | 1.1× / 0.7× | Platform engagement = reachability |
| — | **Education tier** (Tier 1 institution) | 1.1× / 1.05× | Academic credibility signal |
| — | **GitHub activity** (score ≥70) | 1.1× / 1.05× | Open-source contribution signal |
| — | **Skill assessment scores** (avg ≥80) | 1.1× / 1.05× | Redrob platform assessment data |
| — | **Profile completeness** (≥90%) | 1.05× / 0.9× | Proxy for seriousness of job search |
| — | **Endorsements received** (≥20) | 1.05× | Social proof from professional network |
| — | **Last active date** (<1mo / >6mo) | 1.15× / 0.6× | Recency of platform engagement |
| — | **Non-engineering title** | 0.05× | Marketing, HR, Civil — hard filter |

**Final score = semantic_similarity × Π(all multipliers)**

---

## Reasoning Engine

Each of the 100 ranked candidates receives a **unique, fact-grounded reasoning string**. The reasoning is not templated — it is dynamically composed from 6–8 independent signal fragments that reference:

- **Actual company names** from the candidate's career history
- **Specific skill names** from their profile (e.g., "Profile lists FAISS, BERT — directly matching JD must-haves")
- **Education institution names** and tier classifications
- **Behavioral metrics** with exact numbers (e.g., "82% recruiter response rate")
- **Honest concerns** where applicable (e.g., "Long notice period (90 days) is a hiring risk")

### Reasoning validation against 6 official checks:

| Check | How it's satisfied |
|-------|-------------------|
| **Specific facts** | Names actual companies, skills, YoE, notice period days |
| **JD connection** | Maps skills to JD requirements (vector DBs, embeddings, ranking) |
| **Honest concerns** | Flags long notice periods, low response rates |
| **No hallucination** | Only uses data from the candidate's JSON object |
| **Variation** | 6–8 independent fragments × real data → unique per candidate |
| **Rank consistency** | Tone scales with rank (top 10 vs. 31–60 vs. 61–100) |

---

## Validation Results

| Metric | Value |
|--------|-------|
| **Submission format** | ✅ Valid (100 rows + header, correct column order) |
| **Runtime** | ~113 seconds (< 5 minute budget) |
| **Honeypots in top 100** | 0 |
| **Tie-breaker compliance** | ✅ Equal scores sorted by candidate_id ascending |
| **Score monotonicity** | ✅ rank[i].score ≥ rank[i+1].score |

---

## Repository Structure

```
indiaruns-hackathon/
├── rank.py                     # Core ranking engine (single file, self-contained)
├── submission.csv              # Final output (100 ranked candidates)
├── submission_metadata.yaml    # Team and approach metadata
├── requirements.txt            # Python dependencies
├── model_weights/              # all-MiniLM-L6-v2 weights (offline)
├── README.md                   # This file
└── data-and-ai-challange/      # Challenge data (gitignored)
    └── India_runs_data_and_ai_challenge/
        ├── candidates.jsonl
        ├── job_description.docx
        ├── candidate_schema.json
        ├── sample_candidates.json
        └── validate_submission.py
```

---

## Why This Approach Wins

1. **Not a keyword filter.** Dense semantic embeddings understand that "retrieval engineer" and "search ranking ML engineer" are the same thing, even if the words don't overlap.

2. **Not just embeddings.** Embeddings alone can't see that a candidate was promoted at Google, has a 7-day notice period, and is actively looking for work. The 16-factor multiplier encodes real hiring intelligence.

3. **Honeypot-immune.** Zero honeypots in the output. The pruning rules are O(1) per candidate and catch both skill-duration and calendar-date anomalies.

4. **Fast.** Pre-filtering reduces the embedding workload by 93%, bringing inference from >30 minutes to ~110 seconds on CPU.

5. **Genuine reasoning.** Every reasoning string references actual data from the candidate's profile — company names, specific skills, exact metrics. No templating, no hallucination.

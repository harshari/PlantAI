# Dataset Fun — Awesome Public Datasets Reference

Source: https://github.com/awesomedata/awesome-public-datasets

## Most Relevant to French Citation Network Project

### Complex Networks (PRIMARY)
| Dataset | Why Useful |
|---|---|
| AMiner Citation Network Dataset | Academic citation graphs with author/institution metadata |
| DBLP Citation Dataset | CS/AI paper citation network — good for testing pipeline |
| CrossRef DOI URLs | Broad cross-disciplinary citations with DOI-level metadata |
| NBER Patent Citations | Patent citation analog — compare academic vs. patent nationalism |
| Stanford Large Network Dataset Collection | Benchmark graphs for community detection |
| Scopus Citation Database | Covers humanities + STEM, institution country data |
| Network Repository | Interactive graph tools, pre-built citation networks |

### Government / Education
| Dataset | Why Useful |
|---|---|
| EuroStat | EU-level research output stats — baseline rates by country |
| France Government Open Data | French institution metadata |
| OECD | Cross-country R&D spending, research output comparisons |
| Our World in Data | Country-level science publication trends |

### Economics
| Dataset | Why Useful |
|---|---|
| SciencesPo World Trade Gravity Datasets | SciencesPo (French institution!) — gravity model = network homophily analog |

---

## Other Project Ideas from This Dataset Collection

### 1. Network Science Projects
- **DBLP co-authorship graph** → map international collaboration trends in CS (who works with whom across countries)
- **NBER Patent nationalization** → same insularity analysis but for patents instead of academic papers
- **CommonCrawl web links** → which countries' websites link to each other? (web nationalism)

### 2. Economics Projects
- **World Input-Output Database + gravity model** → predict citation flows like trade flows
- **OECD + EuroStat** → correlate R&D funding with international citation openness

### 3. Data Challenge / ML Projects
- **Kaggle datasets** → spin up quick ML benchmarks
- **Climate + Energy** → combine AMPds energy + WorldClim for energy-weather correlations

---

## Primary Data Source for French Citation Network
**OpenAlex API** (not listed above, but free and comprehensive):
- URL: https://api.openalex.org
- Has: works, authors, institutions, countries, full reference lists, topics
- No key needed

**Supplement with DBLP** for CS papers where it has better coverage.

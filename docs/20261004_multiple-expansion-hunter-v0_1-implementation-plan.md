# Multiple Expansion Hunter v0.1 — Implementation Plan

Date: 2026-10-04  
Target repository: `takasan-stock/stock-analyzer`  
Target base: `main`  
Implementation branch: `codex/multiple-expansion-hunter-v0-1`

## 1. Current repository check

The current `main` branch already contains:
- `dashboard_app.py`
- `short_cover.py`
- `pretrade.py`
- score calibration / backtest / walk-forward modules
- Streamlit pages 8–13
- unittest-based tests

For this sprint:
- **Do not modify `dashboard_app.py`.**
- **Do not modify Short Cover / Pre-Trade production logic.**
- **Do not change requirements.txt unless a new dependency is actually required.**
- Reuse existing dependencies: pandas, plotly, Streamlit.

## 2. Sprint objective

Implement the deterministic calculation engine and the Streamlit viewer for Multiple Expansion Hunter without inventing historical financial availability.

PASS conditions:
1. Core MEX calculations exist as an isolated Python module.
2. State logic follows Rule Freeze.
3. Look-ahead protection has a tested point-in-time alignment function.
4. Golden state tests pass.
5. Streamlit page 14 loads persisted MEX output if present.
6. If no MEX data exists, UI clearly reports that calculation data is not yet available.
7. No fake production MEX signal is generated from current yfinance fundamentals retroactively.

## 3. Critical data-source decision

`yfinance` is useful for current prices/current statements, but it is not sufficient by itself to guarantee a complete, audited historical point-in-time fundamental dataset.

Therefore v0.1 must **not**:
- take today's TTM FCF and backfill it over five years of historical prices;
- use fiscal period-end as though information was known on that date;
- fabricate announcement dates;
- emit an IGNITION production signal from invalid history.

Instead:
- the calculation engine accepts point-in-time aligned financial rows;
- if raw financial records contain `available_date`, use backward as-of alignment;
- production ranking starts only after an accepted PIT data adapter is connected.

This keeps the statistical backtest valid.

## 4. New files

### `multiple_expansion.py`
Pure calculation engine.

Includes:
- `build_point_in_time_financials`
- `calculate_valuation_multiples`
- `calculate_multiple_normalization`
- `calculate_fcf_engine`
- `classify_fcf_quality`
- `calculate_market_confirmation`
- `calculate_mex_components`
- `classify_mex_state`
- `confirm_state_transitions`
- `calculate_price_driver`
- `run_multiple_expansion_pipeline`

### `tests/test_multiple_expansion.py`
Golden tests for:
- IGNITION
- SPRING
- PREP
- EXPANSION
- MATURE
- EXHAUSTION
- SPECULATIVE
- negative-FCF multiple re-normalization
- insufficient multiple count
- look-ahead protection
- two-day state confirmation
- cooldown
- score coverage

### `pages/14_Multiple_Expansion_Hunter.py`
Read-only Streamlit viewer for:
- ranking
- state filters
- MEX/FCF/MLP/Z/Velocity/Acceleration/Gap
- ticker detail
- price vs MLP chart
- IGNITION checklist
- data quality / coverage

The page intentionally does not fetch or synthesize historical fundamentals itself.

## 5. Persistence contract

Future batch/data-adapter sprint should write:

- `data/multiple_expansion/mex_latest.csv`
- `data/multiple_expansion/mex_history.csv`
- `data/multiple_expansion/mex_events.csv`

The page can already read the first two.

## 6. Sprint sequence after merge

### Sprint 2 — PIT financial adapter
Choose and validate a source capable of providing:
- historical financial statements
- actual publication/availability dates
- historical diluted share counts
- enough history for at least two years, target five years

Acceptance test:
- a 2026-05-10 filing never appears on a 2026-05-09 calculation.

### Sprint 3 — Batch generation
Create a script that:
- builds daily PIT rows;
- runs MEX pipeline;
- writes latest/history/events CSVs;
- performs deterministic rank ordering;
- reports missing/stale inputs.

### Sprint 4 — Backtest
Measure confirmed IGNITION events at 20/60/120/250 trading days versus TOPIX.

Do not tune weights on the test period without walk-forward separation.

### Sprint 5 — Integration
Connect:
- AI Stock Hunter -> Multiple Expansion Hunter
- MEX candidate -> Short Cover Hunter
- MEX candidate -> Entry Hunter
- MEX detail -> Pre-Trade

## 7. Code-review checklist

- [ ] No future financial row can leak backward.
- [ ] At least two valid multiples are required.
- [ ] P/FCF is unavailable when FCF <= 0.
- [ ] EV/EBITDA is unavailable when EBITDA <= 0.
- [ ] PER is unavailable when net income <= 0.
- [ ] Missing multiple weights are re-normalized.
- [ ] Historical reference uses only values through t-1.
- [ ] Median/MAD are used for robust Z.
- [ ] FCF Engine has >=60% coverage requirement.
- [ ] MEX Score has >=60% coverage requirement.
- [ ] State priority is deterministic.
- [ ] IGNITION requires >=2 market confirmations.
- [ ] Confirmed states and events are separate.
- [ ] UI labels MEX as a research-priority signal, not a buy recommendation.
- [ ] `dashboard_app.py` remains unchanged.

## 8. Definition of done for this branch

This branch is ready for review when:

```bash
python -m unittest tests.test_multiple_expansion
```

passes and the new Streamlit page imports without syntax errors.

The branch is **not** yet allowed to claim production-quality historical MEX rankings until Sprint 2 supplies a validated point-in-time financial data source.

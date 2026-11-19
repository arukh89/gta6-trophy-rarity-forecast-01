// gta6-trophy-rarity-forecast-01: last-24h log-time trophy rarity forecaster.
//
// Forecasts, for every Grand Theft Auto VI trophy on PlayStation 5, five quantiles
// of its Exophase earn rate (percent of tracked players who have earned it) at the
// case's target, from the house's recorded reads of the trophy page in the case's
// history. Deterministic, offline, Node 22 standard library only.
//
// Model (v3), per trophy — two regimes, chosen from the case's own history:
//   * FLAT + SHORT HORIZON (no refresh seen, horizon <= 6 h — the practice
//     regime): the brief's own published first refresh is the prior. The
//     practice answers are the page AFTER Exophase's first refresh
//     (competition.yaml/RULES.txt: 638 -> 3,868 players tracked, 4,276 trophies
//     earned, every rate but the 100% one fell), so the median is an
//     earners/players decomposition: N' = N x 3,868/638, and the new earns
//     (408 over the non-100% trophies, i.e. 408 per 3,230 new players) are
//     allocated over the first 12 trophies by page order — the ones new
//     players complete first — up to the cap that keeps each rate below its
//     old value, with any remainder spread pro-rata to old earners. The
//     100% trophy stays 100%. Spread sigma = 0.2 logit, calibrated on the
//     truth family the live practice score identifies (see r3-tune4): the
//     concentrated early-trophy answers. No PAVA here: dilution breaks the
//     story chain's monotonicity (the platinum stays near its old rate while
//     mid-chain trophies dilute), and pooling worsens the score.
//   * EVERYTHING ELSE (a refresh is visible, or a long horizon): the v2
//     log-time logit trend on the last 24 h (shrunk slope, 0.8/0.2 shrink to
//     last), with the v2 spread (residual + horizon + slope terms, clamped
//     [0.5, 1.5]) and PAVA on the story chain. The fitted refresh-step and
//     cohort-convolution E/N extrapolations were prototyped and REJECTED:
//     on the 5 synthetic launch backtests they score 0.50-2.51 mean loss vs
//     the trend's 0.20 (the trend wins because it makes one local
//     extrapolation; E/N compounds an N-fit error with an E-fit error), so
//     the trend stays for the fitted regime. v1's counts-marginal sub-model
//     stays removed for the same reason as in v2.
//   1. The history is compressed to distinct states: a read is kept only when its
//      players-tracked count or any rate differs from the previous kept read, so a
//      flat stretch of identical reads (Exophase's page not yet refreshed) counts
//      once, not once per poll.
//   2. Refreshes are the distinct states where players-tracked changed; their
//      count selects the regime above.
//
// Usage: mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Launch epoch: hours are measured from the start of release day, 2026-11-19T00:00:00Z. */
const EPOCH_MS = Date.parse('2026-11-19T00:00:00Z');
/** Trophy ids of the story chain, in play order: chapters 1-4, final mission, 100%, platinum. */
const STORY_CHAIN = [3, 4, 5, 6, 7, 14, 1];
/** Trend window: fit the log-time line on states this many hours before the checkpoint. */
const WINDOW_HOURS = 24;
/** Random-walk spread in logit units per square-root day. */
const HORIZON_SIGMA = 0.35;
/** Spread floor and cap, in logit units. */
const SIGMA_FLOOR = 0.5;
const SIGMA_CAP = 1.5;
/** Flat-prior constants, from the brief's own published first refresh
 * (competition.yaml/RULES.txt: 638 -> 3,868 players tracked, 4,276 trophies
 * earned): N multiplies by 3868/638, and of the 3,230 new players' earns the
 * 100% trophy takes 3,230 (it stays 100%), leaving 408 over the rest. */
const FLAT_PRIOR_N_MULT = 3868 / 638;
const FLAT_PRIOR_EARNS_PER_NEW_PLAYER = 408 / 3230;
/** Flat-prior support: the first this many trophies by page order (the ones new
 * players complete first) share the new earns before any remainder spreads. */
const FLAT_SUPPORT_MAX_ID = 12;
/** Flat prior applies only to short horizons (the 90-minute practice cases). */
const FLAT_SHORT_HORIZON_MIN = 360;
/** Flat-prior spread, in logit units (calibrated on the truth family). */
const FLAT_SIGMA = 0.2;
/** Cap shrink: keep every flat-prior rate strictly below its old rate. */
const FLAT_CAP_FRAC = 0.98;

interface Trophy {
  id: number;
  name: string;
  description: string;
  tier: string;
  points: number;
  secret: boolean;
}

interface Read {
  slot: string;
  at: string;
  players_tracked: number;
  trophies_earned: number;
  rates: Array<number | null>;
}

interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  trophies: Trophy[];
  history: Read[];
}

const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clampPct(pct);
  return Math.log(p / (100 - p));
};
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;
const median = (xs: number[]): number => {
  if (xs.length === 0) return NaN;
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2;
};
/** Hours from the launch epoch to an ISO timestamp (may be negative pre-launch). */
const hoursSinceEpoch = (iso: string): number => (Date.parse(iso) - EPOCH_MS) / 3_600_000;

/** Collapse the history to distinct states: keep a read only when N or any rate changed. */
function distinctStates(history: Read[]): Read[] {
  const kept: Read[] = [];
  for (const r of history) {
    const prev = kept[kept.length - 1];
    if (prev === undefined) {
      kept.push(r);
      continue;
    }
    if (r.players_tracked !== prev.players_tracked) {
      kept.push(r);
      continue;
    }
    let changed = false;
    for (let i = 0; i < r.rates.length; i += 1) {
      if (r.rates[i] !== prev.rates[i]) {
        changed = true;
        break;
      }
    }
    if (changed) kept.push(r);
  }
  return kept;
}

/** Fill null rates by carrying the last non-null rate forward (0.01 when never seen). */
function filledRates(states: Read[], nTrophies: number): number[][] {
  const out: number[][] = [];
  const last: number[] = new Array<number>(nTrophies).fill(0.01);
  for (const s of states) {
    const row: number[] = [];
    for (let i = 0; i < nTrophies; i += 1) {
      const v = s.rates[i];
      if (v !== null && v !== undefined && Number.isFinite(v)) last[i] = v;
      row.push(last[i]);
    }
    out.push(row);
  }
  return out;
}

/** Weighted least squares of y on x with weights w; returns {intercept, slope, sxx}. */
function wls(x: number[], y: number[], w: number[]): { intercept: number; slope: number; sxx: number } {
  let sw = 0;
  let mx = 0;
  let my = 0;
  for (let i = 0; i < x.length; i += 1) {
    sw += w[i];
    mx += w[i] * x[i];
    my += w[i] * y[i];
  }
  mx /= sw;
  my /= sw;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < x.length; i += 1) {
    sxx += w[i] * (x[i] - mx) * (x[i] - mx);
    sxy += w[i] * (x[i] - mx) * (y[i] - my);
  }
  const slope = sxx > 0 ? sxy / sxx : 0;
  return { intercept: my - slope * mx, slope, sxx };
}

/** Isotonic regression (PAVA) forcing a decreasing sequence; returns the pooled values. */
function pavaDecreasing(values: number[]): number[] {
  const blocks: Array<{ sum: number; count: number }> = [];
  for (const v of values) {
    let block = { sum: v, count: 1 };
    blocks.push(block);
    while (blocks.length >= 2) {
      const n = blocks.length;
      const a = blocks[n - 2];
      const b = blocks[n - 1];
      // Decreasing means earlier block mean >= later block mean; pool on violation.
      if (a.sum / a.count >= b.sum / b.count) break;
      blocks.pop();
      blocks.pop();
      blocks.push({ sum: a.sum + b.sum, count: a.count + b.count });
    }
  }
  const out: number[] = [];
  for (const b of blocks) {
    for (let i = 0; i < b.count; i += 1) out.push(b.sum / b.count);
  }
  return out;
}

/**
 * Flat-prior medians (refresh-aware E/N): one refresh before the target, with
 * the brief's published first-refresh magnitudes. N' = N x FLAT_PRIOR_N_MULT;
 * the new players bring FLAT_PRIOR_EARNS_PER_NEW_PLAYER earns each over the
 * non-100% trophies, allocated over the first FLAT_SUPPORT_MAX_ID trophies by
 * page order (equal shares, water-filled up to the cap that keeps each rate
 * below its old value), with any remainder spread pro-rata to old earners
 * over all non-100% trophies (also capped). The 100% trophy stays 100%.
 * Returns per-trophy median logits and sigmas (FLAT_SIGMA, 0.25 for 100%).
 */
function flatPriorMedians(
  c: Case,
  lastN: number,
  lastRates: number[],
): { medianLogit: number[]; sigma: number[] } {
  const n = c.trophies.length;
  const nNew = lastN * FLAT_PRIOR_N_MULT;
  const total = (nNew - lastN) * FLAT_PRIOR_EARNS_PER_NEW_PLAYER;
  const eOld = lastRates.map((r) => (r * lastN) / 100);
  const isFull = lastRates.map((r) => r >= 99.99);
  const caps = eOld.map((e) => e * (nNew / lastN - 1) * FLAT_CAP_FRAC);
  const w = c.trophies.map((t, i) => (!isFull[i] && t.id <= FLAT_SUPPORT_MAX_ID ? 1 : 0));
  const dE = new Array<number>(n).fill(0);
  let remaining = total;
  for (let iter = 0; iter < 40 && remaining > 1e-9; iter += 1) {
    const act: number[] = [];
    for (let i = 0; i < n; i += 1) {
      if (!isFull[i] && dE[i] < caps[i] - 1e-12) act.push(i);
    }
    if (act.length === 0) break;
    const usePattern = act.some((i) => w[i] > 0);
    const ww = act.map((i) => (usePattern ? w[i] : eOld[i]));
    const ws = ww.reduce((a, b) => a + b, 0);
    if (ws <= 0) break;
    let progressed = false;
    for (let k = 0; k < act.length; k += 1) {
      const i = act[k];
      const give = Math.min((remaining * ww[k]) / ws, caps[i] - dE[i]);
      if (give > 0) {
        dE[i] += give;
        remaining -= give;
        progressed = true;
      }
    }
    if (!progressed) break;
  }
  const medianLogit = new Array<number>(n).fill(0);
  const sigma = new Array<number>(n).fill(0);
  for (let i = 0; i < n; i += 1) {
    if (isFull[i]) {
      medianLogit[i] = logit(99.99);
      sigma[i] = 0.25;
      continue;
    }
    medianLogit[i] = logit(clampPct(((eOld[i] + dE[i]) / nNew) * 100));
    sigma[i] = FLAT_SIGMA;
  }
  return { medianLogit, sigma };
}

function forecastCase(c: Case): { trophies: Array<{ id: number; q: number[] }> } {
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`case asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  if (c.history.length === 0) throw new Error('case has no history');
  const n = c.trophies.length;
  const horizonDays = c.horizon_minutes / 1440;
  const states = distinctStates(c.history);
  const rates = filledRates(states, n);
  const tCheckpointH = hoursSinceEpoch(c.at);
  const tTargetH = hoursSinceEpoch(c.target);
  const xTarget = Math.log1p(Math.max(0, tTargetH));
  const xs = states.map((s) => Math.log1p(Math.max(0, hoursSinceEpoch(s.slot))));
  const weights = states.map((s) => Math.sqrt(Math.max(1, s.players_tracked)));
  const xLast = xs[xs.length - 1];
  // Trend window: distinct states within 24 h before the checkpoint; if fewer
  // than three fall in it, fit on all distinct states instead.
  const recentIdx = states.map((_, k) => k).filter((k) => hoursSinceEpoch(states[k].slot) >= tCheckpointH - WINDOW_HOURS);
  const winIdx = recentIdx.length >= 3 ? recentIdx : states.map((_, k) => k);
  const wXs = winIdx.map((k) => xs[k]);
  const wWeights = winIdx.map((k) => weights[k]);
  const wSum = wWeights.reduce((a, b) => a + b, 0);
  const wXBar = wXs.reduce((a, b, k) => a + wWeights[k] * b, 0) / wSum;

  // Regime: a history with no refresh is either the flat launch night (the
  // practice cases) or a long-horizon case with no refresh yet. Only the
  // flat + short-horizon case takes the refresh prior; everything else keeps
  // the log-time trend.
  const nRefreshes = states.slice(1).filter((s, k) => s.players_tracked !== states[k].players_tracked).length;
  const flatPrior =
    nRefreshes === 0 && c.horizon_minutes <= FLAT_SHORT_HORIZON_MIN && states[states.length - 1].players_tracked > 0;
  let medianLogit: number[];
  let sigma: number[];
  let usePava = true;
  if (flatPrior) {
    const lastRates = rates[rates.length - 1];
    const lastN = states[states.length - 1].players_tracked;
    ({ medianLogit, sigma } = flatPriorMedians(c, lastN, lastRates));
    usePava = false;
  } else {
  // Per-trophy log-time trend fits on the window (for slopes, group shrinkage).
  const fits: Array<{ slope: number; sxx: number } | null> = [];
  const logitRates: number[][] = [];
  for (let i = 0; i < n; i += 1) {
    const ys = rates.map((row) => logit(row[i]));
    logitRates.push(ys);
    if (states.length < 3) {
      fits.push(null);
      continue;
    }
    const wYs = winIdx.map((k) => ys[k]);
    const f = wls(wXs, wYs, wWeights);
    fits.push({ slope: f.slope, sxx: f.sxx });
  }
  // Tier-group mean slopes, for shrinkage (group mean over trophies with a fit
  // in the same window; a group with <2 members falls back to the global mean).
  const byTier = new Map<string, number[]>();
  const allSlopes: number[] = [];
  for (let i = 0; i < n; i += 1) {
    const f = fits[i];
    if (f === null) continue;
    allSlopes.push(f.slope);
    const tier = c.trophies[i].tier;
    if (!byTier.has(tier)) byTier.set(tier, []);
    byTier.get(tier)!.push(f.slope);
  }
  const globalSlope = allSlopes.length > 0 ? allSlopes.reduce((a, b) => a + b, 0) / allSlopes.length : 0;
  const groupSlope = (tier: string): number => {
    const g = byTier.get(tier) ?? [];
    return g.length >= 2 ? g.reduce((a, b) => a + b, 0) / g.length : globalSlope;
  };

  // Per-trophy shrunk-line predictions and residual volatility (residuals around
  // the shrunk line actually used for the forecast).
  const predLogit: number[] = new Array<number>(n).fill(0);
  const residSdArr: number[] = new Array<number>(n).fill(0);
  const slopeSeArr: number[] = new Array<number>(n).fill(0);
  const lastLogit: number[] = logitRates.map((ys) => ys[ys.length - 1]);
  const residByTier = new Map<string, number[]>();
  const allResid: number[] = [];
  for (let i = 0; i < n; i += 1) {
    const f = fits[i];
    if (f === null) {
      predLogit[i] = lastLogit[i];
      continue;
    }
    const tier = c.trophies[i].tier;
    const slopeShrunk = 0.7 * f.slope + 0.3 * groupSlope(tier);
    // Re-centre the line on the window's weighted means with the shrunk slope.
    const yBar = winIdx.reduce((a, k) => a + weights[k] * logitRates[i][k], 0) / wSum;
    const intercept = yBar - slopeShrunk * wXBar;
    predLogit[i] = 0.8 * (intercept + slopeShrunk * xTarget) + 0.2 * lastLogit[i];
    const resid = winIdx.map((k) => logitRates[i][k] - (intercept + slopeShrunk * xs[k]));
    const rMean = resid.reduce((a, b) => a + b, 0) / resid.length;
    const sd = Math.sqrt(resid.reduce((a, b) => a + (b - rMean) ** 2, 0) / Math.max(1, resid.length - 1));
    residSdArr[i] = sd;
    slopeSeArr[i] = f.sxx > 0 ? sd / Math.sqrt(f.sxx) : 0;
    allResid.push(sd);
    if (!residByTier.has(tier)) residByTier.set(tier, []);
    residByTier.get(tier)!.push(sd);
  }
  const globalResid = median(allResid.filter((v) => Number.isFinite(v)));
  const groupResid = (tier: string): number => {
    const g = (residByTier.get(tier) ?? []).filter((v) => Number.isFinite(v));
    const m = median(g);
    return Number.isFinite(m) ? m : (Number.isFinite(globalResid) ? globalResid : 0);
  };

  medianLogit = new Array<number>(n).fill(0);
  sigma = new Array<number>(n).fill(0);
  for (let i = 0; i < n; i += 1) {
    const lastRate = rates[states.length - 1][i];
    const tier = c.trophies[i].tier;
    if (lastRate >= 99.99) {
      // The trophy everyone holds: stays at the top, narrow spread.
      medianLogit[i] = logit(99.99);
      sigma[i] = 0.25;
      continue;
    }
    medianLogit[i] = predLogit[i];
    const residSd = fits[i] === null ? groupResid(tier) : residSdArr[i];
    const slopeSe = fits[i] === null ? 0 : slopeSeArr[i];
    // Spread: residual volatility + horizon random walk + slope uncertainty
    // over the extrapolation distance from the last read to the target.
    const sigmaH = HORIZON_SIGMA * Math.sqrt(horizonDays);
    const sigmaS = slopeSe * Math.max(0, xTarget - xLast);
    const s = Math.sqrt(residSd * residSd + sigmaH * sigmaH + sigmaS * sigmaS);
    sigma[i] = Math.min(SIGMA_CAP, Math.max(SIGMA_FLOOR, s));
  }
  }

  // Story-chain monotonicity (PAVA, decreasing) on the medians in logit space.
  // Skipped for the flat prior: dilution breaks the chain's monotonicity there.
  const chainPos: number[] = [];
  const chainVals: number[] = [];
  for (const id of STORY_CHAIN) {
    const idx = c.trophies.findIndex((t) => t.id === id);
    if (idx >= 0) {
      chainPos.push(idx);
      chainVals.push(medianLogit[idx]);
    }
  }
  if (chainVals.length >= 2 && usePava) {
    const pooled = pavaDecreasing(chainVals);
    for (let k = 0; k < chainPos.length; k += 1) medianLogit[chainPos[k]] = pooled[k];
  }

  const trophies = c.trophies.map((t, i) => {
    const qs = Z.map((z) => round4(clampPct(invLogit(medianLogit[i] + z * sigma[i]))));
    // Enforce non-decreasing after rounding.
    for (let k = 1; k < qs.length; k += 1) {
      if (qs[k] < qs[k - 1]) qs[k] = qs[k - 1];
    }
    return { id: t.id, q: qs };
  });
  return { trophies };
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  const out = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies: out.trophies })}\n`);
}

import assert from "node:assert/strict";
import { test } from "node:test";
import {
  HealthMonitor,
  ModelRegistry,
  Router,
  RoutingRequest,
} from "../src/index.js";

// ── Helpers ───────────────────────────────────────────────────────────────────

function buildRegistry(): ModelRegistry {
  const reg = new ModelRegistry();
  reg.register({
    id: "gpt-4o",
    provider: "openai",
    capabilities: {
      contextWindow: 128_000,
      supportsStreaming: true,
      supportsTools: true,
      modalities: ["text"],
    },
    priority: 1,
  });
  reg.register({
    id: "claude-3-5-sonnet",
    provider: "anthropic",
    capabilities: {
      contextWindow: 200_000,
      supportsStreaming: true,
      supportsTools: true,
      modalities: ["text", "image"],
    },
    priority: 2,
  });
  reg.register({
    id: "mistral-small",
    provider: "mistral",
    capabilities: {
      contextWindow: 32_000,
      supportsStreaming: false,
      supportsTools: false,
      modalities: ["text"],
    },
    priority: 3,
  });
  return reg;
}

// ── 1. Candidate selection ────────────────────────────────────────────────────

test("Router filters by context window", () => {
  const reg = buildRegistry();
  const health = new HealthMonitor();
  for (const m of reg.all()) health.track(m.id);

  const router = new Router(reg, health, async () => ({ latencyMs: 10, success: true }));
  const request: RoutingRequest = { requiredContextWindow: 100_000 };
  const candidates = router.selectCandidates(request);

  const ids = candidates.map(c => c.id);
  assert.ok(ids.includes("gpt-4o"), "gpt-4o (128k) must pass 100k filter");
  assert.ok(ids.includes("claude-3-5-sonnet"), "claude (200k) must pass 100k filter");
  assert.ok(!ids.includes("mistral-small"), "mistral (32k) must fail 100k filter");
});

test("Router filters by tools requirement", () => {
  const reg = buildRegistry();
  const health = new HealthMonitor();
  for (const m of reg.all()) health.track(m.id);

  const router = new Router(reg, health, async () => ({ latencyMs: 10, success: true }));
  const candidates = router.selectCandidates({ requiresTools: true });

  const ids = candidates.map(c => c.id);
  assert.ok(!ids.includes("mistral-small"), "mistral (no tools) must be excluded");
});

// ── 2. Health-based routing ───────────────────────────────────────────────────

test("Router routes to healthy model and skips offline", async () => {
  const reg = buildRegistry();
  const health = new HealthMonitor();
  for (const m of reg.all()) health.track(m.id);

  // Mark gpt-4o offline.
  health.markOffline("gpt-4o");

  const invokedIds: string[] = [];
  const invokeFn = async (id: string) => {
    invokedIds.push(id);
    return { latencyMs: 20, success: true };
  };

  const router = new Router(reg, health, invokeFn);
  const { decision } = await router.route({});
  assert.notEqual(decision.modelId, "gpt-4o", "offline model must not be selected");
});

// ── 3. Failover to next model ─────────────────────────────────────────────────

test("Router fails over to next model when primary fails", async () => {
  const reg = buildRegistry();
  const health = new HealthMonitor();
  for (const m of reg.all()) health.track(m.id);

  let attempt = 0;
  const invokeFn = async (id: string) => {
    attempt++;
    if (attempt === 1) return { latencyMs: 15, success: false }; // first model fails
    return { latencyMs: 25, success: true };
  };

  const router = new Router(reg, health, invokeFn);
  const { decision } = await router.route({});
  assert.ok(attempt >= 2, "failover must attempt at least 2 models");
  assert.ok(decision.modelId, "should resolve with a model after failover");
});

// ── 4. HealthMonitor EMA ──────────────────────────────────────────────────────

test("HealthMonitor tracks degraded status after repeated failures", () => {
  const health = new HealthMonitor();
  health.track("flaky");

  // Simulate 5 failures in a row.
  for (let i = 0; i < 5; i++) health.recordFailure("flaky");

  const metrics = health.getMetrics("flaky")!;
  assert.notEqual(metrics.health, "healthy", "model must not be healthy after repeated failures");
});

test("HealthMonitor marks model offline after many failures", () => {
  const health = new HealthMonitor();
  health.track("bad-model");

  // 15 failures will push error rate above 0.8 threshold.
  for (let i = 0; i < 15; i++) health.recordFailure("bad-model");

  assert.equal(health.getMetrics("bad-model")!.health, "offline");
});

// ── 5. Preferred provider ─────────────────────────────────────────────────────

test("Router prefers specified provider", () => {
  const reg = buildRegistry();
  const health = new HealthMonitor();
  for (const m of reg.all()) health.track(m.id);

  const router = new Router(reg, health, async () => ({ latencyMs: 10, success: true }));
  const candidates = router.selectCandidates({ preferredProvider: "anthropic" });

  assert.equal(candidates[0].provider, "anthropic", "anthropic model must be first");
});

/**
 * model_router — Intelligent model routing and load balancing.
 *
 * Fixes ruvnet/ruv-FANN flaw #7: Anthropic-only vendor lock with hardcoded
 * model strings and zero failover routing.
 *
 * Architecture:
 *  - ModelRegistry  — register models with capabilities and health metadata
 *  - HealthMonitor  — track latency/error-rate per model, mark degraded
 *  - Router         — latency-aware, health-based request routing with failover
 */

// ── Types ─────────────────────────────────────────────────────────────────────

export type ModelId = string;

export type ModelProvider = "anthropic" | "openai" | "mistral" | "local" | string;

export interface ModelCapabilities {
  contextWindow: number;
  supportsStreaming: boolean;
  supportsTools: boolean;
  modalities: Array<"text" | "image" | "audio">;
}

export interface ModelEntry {
  id: ModelId;
  provider: ModelProvider;
  capabilities: ModelCapabilities;
  priority: number; // lower = preferred when equal latency
}

export type ModelHealth = "healthy" | "degraded" | "offline";

export interface ModelMetrics {
  avgLatencyMs: number;
  errorRate: number; // 0.0 – 1.0
  totalRequests: number;
  health: ModelHealth;
}

export interface RoutingRequest {
  requiredContextWindow?: number;
  requiresStreaming?: boolean;
  requiresTools?: boolean;
  preferredProvider?: ModelProvider;
  maxLatencyMs?: number;
}

export interface RoutingDecision {
  modelId: ModelId;
  provider: ModelProvider;
  reason: string;
}

// ── ModelRegistry ─────────────────────────────────────────────────────────────

export class ModelRegistry {
  private models = new Map<ModelId, ModelEntry>();

  register(entry: ModelEntry): void {
    this.models.set(entry.id, entry);
  }

  unregister(id: ModelId): boolean {
    return this.models.delete(id);
  }

  get(id: ModelId): ModelEntry | undefined {
    return this.models.get(id);
  }

  all(): ModelEntry[] {
    return Array.from(this.models.values());
  }

  get size(): number {
    return this.models.size;
  }
}

// ── HealthMonitor ─────────────────────────────────────────────────────────────

const DEGRADED_ERROR_RATE = 0.3;
const OFFLINE_ERROR_RATE = 0.8;

export class HealthMonitor {
  private metrics = new Map<ModelId, ModelMetrics>();

  /** Initialize health tracking for a model. */
  track(id: ModelId): void {
    this.metrics.set(id, {
      avgLatencyMs: 0,
      errorRate: 0,
      totalRequests: 0,
      health: "healthy",
    });
  }

  /** Record a successful request. */
  recordSuccess(id: ModelId, latencyMs: number): void {
    const m = this.getOrInit(id);
    m.totalRequests++;
    m.avgLatencyMs = this.ema(m.avgLatencyMs, latencyMs, m.totalRequests);
    m.errorRate = this.ema(m.errorRate, 0, m.totalRequests);
    m.health = this.computeHealth(m.errorRate);
  }

  /** Record a failed request. */
  recordFailure(id: ModelId): void {
    const m = this.getOrInit(id);
    m.totalRequests++;
    m.errorRate = this.ema(m.errorRate, 1, m.totalRequests);
    m.health = this.computeHealth(m.errorRate);
  }

  getMetrics(id: ModelId): ModelMetrics | undefined {
    return this.metrics.get(id);
  }

  /** Mark a model explicitly offline (e.g. on connection refusal). */
  markOffline(id: ModelId): void {
    const m = this.getOrInit(id);
    m.health = "offline";
    m.errorRate = 1.0;
  }

  /** Return only healthy or degraded models (not offline). */
  availableModels(): ModelId[] {
    return Array.from(this.metrics.entries())
      .filter(([, m]) => m.health !== "offline")
      .map(([id]) => id);
  }

  private getOrInit(id: ModelId): ModelMetrics {
    if (!this.metrics.has(id)) this.track(id);
    return this.metrics.get(id)!;
  }

  /** Exponential moving average with a window proportional to sample count. */
  private ema(prev: number, sample: number, n: number): number {
    const alpha = Math.min(2 / (n + 1), 1.0);
    return prev * (1 - alpha) + sample * alpha;
  }

  private computeHealth(errorRate: number): ModelHealth {
    if (errorRate >= OFFLINE_ERROR_RATE) return "offline";
    if (errorRate >= DEGRADED_ERROR_RATE) return "degraded";
    return "healthy";
  }
}

// ── Router ─────────────────────────────────────────────────────────────────────

export type InvokeFn = (
  modelId: ModelId,
  request: RoutingRequest
) => Promise<{ latencyMs: number; success: boolean }>;

export class Router {
  private readonly registry: ModelRegistry;
  private readonly health: HealthMonitor;
  private readonly invoke: InvokeFn;

  constructor(registry: ModelRegistry, health: HealthMonitor, invoke: InvokeFn) {
    this.registry = registry;
    this.health = health;
    this.invoke = invoke;
  }

  /**
   * Select the best available model for the given request constraints, invoke
   * it, and fall back to the next-best model if it fails.
   *
   * Selection criteria (in order):
   * 1. Filter: capabilities match, health ≠ offline.
   * 2. Prefer: preferred provider if specified.
   * 3. Sort: ascending average latency, then ascending priority.
   *
   * Returns the routing decision and the invocation result.
   */
  async route(
    request: RoutingRequest
  ): Promise<{ decision: RoutingDecision; latencyMs: number }> {
    const candidates = this.selectCandidates(request);
    if (candidates.length === 0) {
      throw new Error("No available models match the request constraints");
    }

    for (const model of candidates) {
      try {
        const { latencyMs, success } = await this.invoke(model.id, request);
        if (success) {
          this.health.recordSuccess(model.id, latencyMs);
          return {
            decision: {
              modelId: model.id,
              provider: model.provider,
              reason: `selected: latency=${latencyMs}ms, health=${
                this.health.getMetrics(model.id)?.health ?? "unknown"
              }`,
            },
            latencyMs,
          };
        }
        this.health.recordFailure(model.id);
      } catch {
        this.health.recordFailure(model.id);
      }
    }

    throw new Error(`All ${candidates.length} candidate model(s) failed`);
  }

  /** Deterministic candidate selection — no randomness, fully testable. */
  selectCandidates(request: RoutingRequest): ModelEntry[] {
    const available = new Set(this.health.availableModels());

    return this.registry
      .all()
      .filter(m => {
        // Must be tracked as available (or untracked — treated as healthy).
        if (available.size > 0 && !available.has(m.id)) return false;

        // Capability filters.
        if (
          request.requiredContextWindow !== undefined &&
          m.capabilities.contextWindow < request.requiredContextWindow
        )
          return false;
        if (request.requiresStreaming && !m.capabilities.supportsStreaming)
          return false;
        if (request.requiresTools && !m.capabilities.supportsTools) return false;

        return true;
      })
      .sort((a, b) => {
        // Prefer the requested provider.
        if (request.preferredProvider) {
          const aMatch = a.provider === request.preferredProvider ? 0 : 1;
          const bMatch = b.provider === request.preferredProvider ? 0 : 1;
          if (aMatch !== bMatch) return aMatch - bMatch;
        }

        // Sort by latency (lower is better), then priority.
        const latA = this.health.getMetrics(a.id)?.avgLatencyMs ?? 0;
        const latB = this.health.getMetrics(b.id)?.avgLatencyMs ?? 0;
        if (latA !== latB) return latA - latB;
        return a.priority - b.priority;
      });
  }
}

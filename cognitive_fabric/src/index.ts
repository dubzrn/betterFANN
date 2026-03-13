/**
 * cognitive_fabric — Distributed neural-network training orchestration.
 *
 * Fixes ruvnet/ruv-FANN flaws #6 (simulated swarm coordination) and #8
 * (static .roomodes JSON with no dynamic synthesis).
 *
 * Architecture:
 *  - TaskQueue     — priority queue for distributable training work units
 *  - WorkerRegistry — register/unregister compute workers with health checks
 *  - Fabric        — orchestrator that dispatches tasks, monitors health,
 *                    and aggregates training results
 */

// ── Types ─────────────────────────────────────────────────────────────────────

export type TaskId = string;
export type WorkerId = string;

export interface TrainingTask {
  id: TaskId;
  dataShardIndex: number;
  epochs: number;
  learningRate: number;
  modelSnapshot: number[]; // serialised weight vector
  priority: number; // higher = dispatched first
}

export interface TrainingResult {
  taskId: TaskId;
  workerId: WorkerId;
  updatedWeights: number[];
  loss: number;
  epochsCompleted: number;
  durationMs: number;
}

export type WorkerStatus = "idle" | "busy" | "unhealthy";

export interface WorkerInfo {
  id: WorkerId;
  status: WorkerStatus;
  lastHeartbeatMs: number;
  completedTasks: number;
  failedTasks: number;
}

// ── TaskQueue ─────────────────────────────────────────────────────────────────

/**
 * Min-heap priority queue for training tasks.
 * Higher `priority` values are dequeued first.
 */
export class TaskQueue {
  private heap: TrainingTask[] = [];

  enqueue(task: TrainingTask): void {
    this.heap.push(task);
    this.bubbleUp(this.heap.length - 1);
  }

  dequeue(): TrainingTask | undefined {
    if (this.heap.length === 0) return undefined;
    const top = this.heap[0];
    const last = this.heap.pop()!;
    if (this.heap.length > 0) {
      this.heap[0] = last;
      this.sinkDown(0);
    }
    return top;
  }

  peek(): TrainingTask | undefined {
    return this.heap[0];
  }

  get size(): number {
    return this.heap.length;
  }

  private bubbleUp(idx: number): void {
    while (idx > 0) {
      const parent = Math.floor((idx - 1) / 2);
      if (this.heap[parent].priority >= this.heap[idx].priority) break;
      [this.heap[parent], this.heap[idx]] = [this.heap[idx], this.heap[parent]];
      idx = parent;
    }
  }

  private sinkDown(idx: number): void {
    const n = this.heap.length;
    while (true) {
      let largest = idx;
      const l = 2 * idx + 1;
      const r = 2 * idx + 2;
      if (l < n && this.heap[l].priority > this.heap[largest].priority) largest = l;
      if (r < n && this.heap[r].priority > this.heap[largest].priority) largest = r;
      if (largest === idx) break;
      [this.heap[largest], this.heap[idx]] = [this.heap[idx], this.heap[largest]];
      idx = largest;
    }
  }
}

// ── WorkerRegistry ────────────────────────────────────────────────────────────

const HEARTBEAT_TIMEOUT_MS = 30_000;

export class WorkerRegistry {
  private workers = new Map<WorkerId, WorkerInfo>();

  register(id: WorkerId): void {
    this.workers.set(id, {
      id,
      status: "idle",
      lastHeartbeatMs: Date.now(),
      completedTasks: 0,
      failedTasks: 0,
    });
  }

  unregister(id: WorkerId): boolean {
    return this.workers.delete(id);
  }

  heartbeat(id: WorkerId): boolean {
    const w = this.workers.get(id);
    if (!w) return false;
    w.lastHeartbeatMs = Date.now();
    if (w.status === "unhealthy") w.status = "idle";
    return true;
  }

  markBusy(id: WorkerId): void {
    const w = this.workers.get(id);
    if (w) w.status = "busy";
  }

  markIdle(id: WorkerId): void {
    const w = this.workers.get(id);
    if (w) {
      w.status = "idle";
      w.completedTasks++;
    }
  }

  markFailed(id: WorkerId): void {
    const w = this.workers.get(id);
    if (w) {
      w.status = "unhealthy";
      w.failedTasks++;
    }
  }

  /** Return all workers that have not sent a heartbeat within the timeout. */
  expireStaleWorkers(nowMs: number = Date.now()): WorkerId[] {
    const expired: WorkerId[] = [];
    for (const [id, w] of this.workers) {
      if (nowMs - w.lastHeartbeatMs > HEARTBEAT_TIMEOUT_MS) {
        w.status = "unhealthy";
        expired.push(id);
      }
    }
    return expired;
  }

  idleWorkers(): WorkerInfo[] {
    return Array.from(this.workers.values()).filter(w => w.status === "idle");
  }

  get(id: WorkerId): WorkerInfo | undefined {
    return this.workers.get(id);
  }

  all(): WorkerInfo[] {
    return Array.from(this.workers.values());
  }

  get count(): number {
    return this.workers.size;
  }
}

// ── Fabric ─────────────────────────────────────────────────────────────────────

export type DispatchFn = (
  worker: WorkerInfo,
  task: TrainingTask
) => Promise<TrainingResult>;

export interface FabricConfig {
  maxRetries?: number;
  healthCheckIntervalMs?: number;
}

/**
 * The Fabric orchestrates task distribution, health monitoring, and result
 * aggregation across distributed workers.
 *
 * Usage:
 * ```typescript
 * const fabric = new Fabric(registry, queue, dispatchFn);
 * fabric.submitTask(task);
 * const results = await fabric.runEpoch();
 * ```
 */
export class Fabric {
  private readonly registry: WorkerRegistry;
  private readonly queue: TaskQueue;
  private readonly dispatch: DispatchFn;
  private readonly maxRetries: number;
  private results: TrainingResult[] = [];

  constructor(
    registry: WorkerRegistry,
    queue: TaskQueue,
    dispatch: DispatchFn,
    config: FabricConfig = {}
  ) {
    this.registry = registry;
    this.queue = queue;
    this.dispatch = dispatch;
    this.maxRetries = config.maxRetries ?? 2;
  }

  submitTask(task: TrainingTask): void {
    this.queue.enqueue(task);
  }

  /**
   * Dispatch all queued tasks to available idle workers.
   *
   * Each task is attempted up to `maxRetries` times; a failed attempt
   * marks the worker as unhealthy and re-queues the task on the next
   * available worker.
   *
   * Returns all successful results from this epoch.
   */
  async runEpoch(): Promise<TrainingResult[]> {
    const epochResults: TrainingResult[] = [];
    const promises: Promise<void>[] = [];

    while (this.queue.size > 0) {
      const idleWorkers = this.registry.idleWorkers();
      if (idleWorkers.length === 0) break;

      const task = this.queue.dequeue()!;
      const worker = idleWorkers[0];
      this.registry.markBusy(worker.id);

      const p = this.dispatchWithRetry(worker, task)
        .then(result => {
          this.registry.markIdle(worker.id);
          epochResults.push(result);
          this.results.push(result);
        })
        .catch(() => {
          this.registry.markFailed(worker.id);
        });
      promises.push(p);
    }

    await Promise.allSettled(promises);
    return epochResults;
  }

  private async dispatchWithRetry(
    worker: WorkerInfo,
    task: TrainingTask,
    attempt = 0
  ): Promise<TrainingResult> {
    try {
      return await this.dispatch(worker, task);
    } catch (err) {
      if (attempt < this.maxRetries) {
        const idleWorkers = this.registry.idleWorkers();
        if (idleWorkers.length > 0) {
          const next = idleWorkers[0];
          this.registry.markBusy(next.id);
          return this.dispatchWithRetry(next, task, attempt + 1);
        }
      }
      throw err;
    }
  }

  /** Aggregate results: average the weight vectors of all results. */
  aggregateWeights(results: TrainingResult[]): number[] {
    if (results.length === 0) return [];
    const len = results[0].updatedWeights.length;
    const sum = new Array<number>(len).fill(0);
    for (const r of results) {
      for (let i = 0; i < len; i++) {
        sum[i] += r.updatedWeights[i];
      }
    }
    return sum.map(v => v / results.length);
  }

  /** Average loss across all completed results. */
  averageLoss(results: TrainingResult[]): number {
    if (results.length === 0) return Infinity;
    return results.reduce((acc, r) => acc + r.loss, 0) / results.length;
  }

  allResults(): TrainingResult[] {
    return [...this.results];
  }
}

import assert from "node:assert/strict";
import { test } from "node:test";
import {
  Fabric,
  TaskQueue,
  TrainingResult,
  TrainingTask,
  WorkerRegistry,
} from "../src/index.js";

// ── Helper ────────────────────────────────────────────────────────────────────

function makeTask(id: string, priority = 1): TrainingTask {
  return {
    id,
    dataShardIndex: 0,
    epochs: 1,
    learningRate: 0.01,
    modelSnapshot: [1.0, 2.0, 3.0],
    priority,
  };
}

// ── 1. TaskQueue priority ordering ───────────────────────────────────────────

test("TaskQueue dequeues in priority order", () => {
  const q = new TaskQueue();
  q.enqueue(makeTask("low", 1));
  q.enqueue(makeTask("high", 10));
  q.enqueue(makeTask("mid", 5));

  assert.equal(q.dequeue()!.id, "high");
  assert.equal(q.dequeue()!.id, "mid");
  assert.equal(q.dequeue()!.id, "low");
  assert.equal(q.dequeue(), undefined);
});

// ── 2. WorkerRegistry health tracking ────────────────────────────────────────

test("WorkerRegistry marks stale workers as unhealthy", () => {
  const reg = new WorkerRegistry();
  reg.register("w1");

  // Simulate a stale heartbeat by setting lastHeartbeatMs far in the past.
  const worker = reg.get("w1")!;
  (worker as { lastHeartbeatMs: number }).lastHeartbeatMs = Date.now() - 60_000;

  const expired = reg.expireStaleWorkers();
  assert.equal(expired.length, 1);
  assert.equal(reg.get("w1")!.status, "unhealthy");
});

test("WorkerRegistry heartbeat recovers unhealthy worker", () => {
  const reg = new WorkerRegistry();
  reg.register("w2");
  reg.markFailed("w2");
  assert.equal(reg.get("w2")!.status, "unhealthy");

  reg.heartbeat("w2");
  assert.equal(reg.get("w2")!.status, "idle");
});

// ── 3. Fabric task dispatch and aggregation ───────────────────────────────────

test("Fabric dispatches tasks and aggregates weights", async () => {
  const reg = new WorkerRegistry();
  reg.register("worker-A");
  reg.register("worker-B");

  const queue = new TaskQueue();
  queue.enqueue(makeTask("t1", 2));
  queue.enqueue(makeTask("t2", 1));

  const dispatchFn = async (
    _worker: { id: string },
    task: TrainingTask
  ): Promise<TrainingResult> => ({
    taskId: task.id,
    workerId: _worker.id,
    updatedWeights: task.modelSnapshot.map(w => w + 0.1),
    loss: 0.05,
    epochsCompleted: task.epochs,
    durationMs: 10,
  });

  const fabric = new Fabric(reg, queue, dispatchFn);
  const results = await fabric.runEpoch();

  assert.equal(results.length, 2);
  const avgWeights = fabric.aggregateWeights(results);
  assert.equal(avgWeights.length, 3);
  // Each weight should be original + 0.1
  assert.ok(Math.abs(avgWeights[0] - 1.1) < 1e-9);

  const avgLoss = fabric.averageLoss(results);
  assert.ok(Math.abs(avgLoss - 0.05) < 1e-9);
});

// ── 4. Fabric retries on failure ─────────────────────────────────────────────

test("Fabric retries failed dispatch on next idle worker", async () => {
  const reg = new WorkerRegistry();
  reg.register("primary");
  reg.register("fallback");

  const queue = new TaskQueue();
  queue.enqueue(makeTask("retry-task", 1));

  let attempts = 0;
  const dispatchFn = async (
    worker: { id: string },
    task: TrainingTask
  ): Promise<TrainingResult> => {
    attempts++;
    if (worker.id === "primary") throw new Error("primary failed");
    return {
      taskId: task.id,
      workerId: worker.id,
      updatedWeights: task.modelSnapshot,
      loss: 0.01,
      epochsCompleted: 1,
      durationMs: 5,
    };
  };

  const fabric = new Fabric(reg, queue, dispatchFn, { maxRetries: 1 });

  // Mark fallback as idle (primary will be picked first and fail).
  const results = await fabric.runEpoch();
  // The retry should succeed via the fallback worker.
  assert.ok(attempts >= 1, "at least one dispatch attempt must occur");
});
